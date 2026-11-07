"""myAISky backend - 기상청 초단기실황(getUltraSrtNcst) 연동 + SQLite 적재.

Run (프로젝트 루트에서):
    uvicorn backend.main:app --reload --port 8001
Run (backend/ 폴더에서):
    uvicorn main:app --reload --port 8001

인증키는 환경변수 KMA_SERVICE_KEY 또는 프로젝트 루트/backend 의 .env 파일에서 읽는다.
"""
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager, closing
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
DB_PATH = BACKEND_DIR / "weather_logs.db"
KST = ZoneInfo("Asia/Seoul")
log = logging.getLogger("uvicorn.error")
KMA_TIMEOUT = 10        # 초
KMA_RETRIES = 2         # 재시도 상한 (총 3회 시도)
KMA_RETRY_DELAY = 1.0   # 초, 시도 횟수에 비례
CACHE_TTL = 600         # 초, 메모리 캐시 TTL (초과 시 DB 최신 행을 우선, 둘 다 없으면 오래된 캐시라도 사용)
_cache: dict[tuple[int, int], tuple[float, dict]] = {}  # (nx,ny) -> (저장시각, 응답)


def _load_dotenv() -> None:
    """의존성 없이 .env 를 읽어 os.environ 에 채운다(이미 있는 값은 덮어쓰지 않음)."""
    for path in (ROOT_DIR / ".env", BACKEND_DIR / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

BASE_URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0"
NCST_URL = f"{BASE_URL}/getUltraSrtNcst"   # 초단기실황
FCST_URL = f"{BASE_URL}/getUltraSrtFcst"   # 초단기예보 (하늘상태 SKY 용)
SEOUL_NX, SEOUL_NY = 55, 127

PTY_TEXT = {  # 강수형태 코드
    "1": "비", "2": "비/눈", "3": "눈", "5": "빗방울",
    "6": "빗방울/눈날림", "7": "눈날림",
}
SKY_TEXT = {"1": "맑음", "3": "구름많음", "4": "흐림"}  # 하늘상태 코드

SCHEDULER_INTERVAL_MIN = 10
_scheduler: BackgroundScheduler | None = None
_last_run: dict = {"time": None, "ok": None, "message": None}


def scheduled_fetch() -> None:
    """스케줄러 잡: 실황 조회 + DB 적재 + 메모리 캐시 갱신. 모든 예외는 로깅만 하고 삼킨다.
    실패 시 기존 캐시/DB 는 그대로 남아 api_weather 의 폴백이 계속 동작한다."""
    nx, ny = SEOUL_NX, SEOUL_NY
    try:
        w = fetch_current_weather(nx, ny)
        saved = save_log(w)
        _cache[(nx, ny)] = (time.time(), dict(w, saved=saved, location="서울"))
        _last_run.update(time=datetime.now(KST).isoformat(timespec="seconds"),
                         ok=True, message=f"saved={saved}")
    except Exception as e:  # 스케줄러가 죽지 않도록 전부 포착
        msg = e.detail if isinstance(e, HTTPException) else f"{type(e).__name__}: {e}"
        log.error("스케줄 수집 실패(캐시/DB 폴백 유지): %s", msg)
        _last_run.update(time=datetime.now(KST).isoformat(timespec="seconds"),
                         ok=False, message=str(msg))


@asynccontextmanager
async def lifespan(_: FastAPI):
    """주의: 다중 워커(--workers N)는 워커마다 스케줄러가 떠 중복 수집된다(INSERT OR IGNORE 로 DB 중복은 없음).
    워커 1개만 쓰거나 나머지는 KMA_SCHEDULER=0 으로 끈다. --reload 는 자식 프로세스 1개만 app 을 import 하므로 안전."""
    global _scheduler
    if os.environ.get("KMA_SCHEDULER", "1") != "0":
        _scheduler = BackgroundScheduler(timezone=KST)
        _scheduler.add_job(
            scheduled_fetch, IntervalTrigger(minutes=SCHEDULER_INTERVAL_MIN),
            id="weather_fetch", max_instances=1, coalesce=True, misfire_grace_time=300,
            next_run_time=datetime.now(KST))  # 시작 직후 1회 즉시 실행
        _scheduler.start()
    try:
        yield
    finally:
        if _scheduler:
            _scheduler.shutdown(wait=False)
            _scheduler = None


app = FastAPI(title="myAISky Weather", lifespan=lifespan)


# ───────────────────────── SQLite ─────────────────────────
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with closing(get_db()) as conn, conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS weather_logs (
                   id          INTEGER PRIMARY KEY AUTOINCREMENT,
                   timestamp   TEXT    NOT NULL,   -- 관측 기준 시각 (KST, ISO)
                   temperature REAL    NOT NULL,   -- 기온 ℃
                   condition   TEXT    NOT NULL,   -- 날씨 상태
                   humidity    REAL,
                   wind_speed  REAL,
                   rain_1h     REAL,
                   nx          INTEGER NOT NULL,
                   ny          INTEGER NOT NULL,
                   created_at  TEXT    NOT NULL,
                   UNIQUE (timestamp, nx, ny)
               )"""
        )


def save_log(w: dict) -> bool:
    """같은 관측시각·격자는 중복 저장하지 않는다. 새로 저장했으면 True."""
    with closing(get_db()) as conn, conn:
        cur = conn.execute(
            """INSERT OR IGNORE INTO weather_logs
               (timestamp, temperature, condition, humidity, wind_speed, rain_1h, nx, ny, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (w["timestamp"], w["temperature"], w["condition"], w["humidity"],
             w["wind_speed"], w["rain_1h"], w["nx"], w["ny"],
             datetime.now(KST).isoformat(timespec="seconds")),
        )
        return cur.rowcount == 1


init_db()


# ───────────────────────── 기상청 API ─────────────────────────
def _service_key() -> str:
    key = os.environ.get("KMA_SERVICE_KEY", "").strip()
    if not key:
        raise HTTPException(500, "KMA_SERVICE_KEY 가 설정되지 않았습니다 (.env 확인).")
    return key


def _kma_get(url: str, params: dict) -> list:
    """기상청 API 호출 후 items 리스트 반환. 오류는 HTTPException 으로 변환.

    타임아웃/연결오류/5xx 만 짧게 재시도(최대 KMA_RETRIES 회). 인증 오류(401/403)는 재시도하지 않는다.
    """
    q = {"serviceKey": _service_key(), "dataType": "JSON", "pageNo": 1,
         "numOfRows": 100, **params}
    last = "알 수 없는 오류"
    for attempt in range(KMA_RETRIES + 1):
        if attempt:
            time.sleep(KMA_RETRY_DELAY * attempt)
        try:
            r = requests.get(url, params=q, timeout=KMA_TIMEOUT)
        except requests.RequestException as e:  # 타임아웃·네트워크 (URL 의 키가 노출되지 않게 타입만 기록)
            last = f"기상청 API 네트워크 오류({type(e).__name__})"
            log.warning("%s attempt=%d", last, attempt + 1)
            continue
        if r.status_code in (401, 403):
            log.error("기상청 API 인증 거부 HTTP %d - 서비스키 만료/미승인/인코딩 확인 필요", r.status_code)
            raise HTTPException(502, f"기상청 API 인증 거부(HTTP {r.status_code}): 서비스키/활용신청 상태를 확인하세요.")
        if r.status_code >= 500 or r.status_code == 429:
            last = f"기상청 API HTTP {r.status_code}"
            log.warning("%s attempt=%d", last, attempt + 1)
            continue
        try:
            r.raise_for_status()
            body = r.json()["response"]
            code = body["header"]["resultCode"]
        except requests.RequestException as e:
            raise HTTPException(502, f"기상청 API 호출 실패(HTTP {r.status_code})")
        except (ValueError, KeyError, TypeError):
            # 인증 오류 등은 JSON 이 아닌 XML 로 내려오는 경우가 있다.
            log.error("기상청 API 응답 해석 실패: %s", r.text[:200].replace(q["serviceKey"], "***"))
            raise HTTPException(502, "기상청 API 응답 해석 실패(JSON 아님)")
        if code != "00":
            log.error("기상청 API resultCode=%s %s", code, body["header"].get("resultMsg"))
            raise HTTPException(502, f"기상청 API 오류 {code}: {body['header'].get('resultMsg')}")
        try:
            items = body["body"]["items"]["item"]
        except (KeyError, TypeError):
            raise HTTPException(502, "기상청 응답에 items 가 없습니다.")
        return [items] if isinstance(items, dict) else items
    raise HTTPException(502, f"{last} (재시도 {KMA_RETRIES}회 후 실패)")


def _base_time(now: datetime) -> datetime:
    """초단기실황은 매시 정각 기준, 약 40분 뒤부터 제공 → 40분 전으로 내린 뒤 정시로 절삭."""
    return (now - timedelta(minutes=40)).replace(minute=0, second=0, microsecond=0)


def _sky_condition(nx: int, ny: int) -> str | None:
    """초단기예보에서 가장 가까운 시각의 하늘상태(SKY)를 가져온다."""
    # 초단기예보는 매시 30분 발표, 45분 이후 제공 → 제공 가능한 가장 최근 발표분 사용
    t = datetime.now(KST) - timedelta(minutes=45)
    fb = t.replace(minute=30, second=0, microsecond=0)
    if fb > t:
        fb -= timedelta(hours=1)
    items = _kma_get(FCST_URL, {"base_date": fb.strftime("%Y%m%d"),
                                "base_time": fb.strftime("%H%M"),
                                "nx": nx, "ny": ny})
    sky = sorted((i for i in items if i["category"] == "SKY"),
                 key=lambda i: (i["fcstDate"], i["fcstTime"]))
    return SKY_TEXT.get(sky[0]["fcstValue"]) if sky else None


def fetch_current_weather(nx: int = SEOUL_NX, ny: int = SEOUL_NY) -> dict:
    """지정 격자의 실시간 기온/날씨 상태 (초단기실황)."""
    now = datetime.now(KST)
    base = _base_time(now)
    items = _kma_get(NCST_URL, {"base_date": base.strftime("%Y%m%d"),
                                "base_time": base.strftime("%H%M"), "nx": nx, "ny": ny})
    v = {i["category"]: i["obsrValue"] for i in items}
    if "T1H" not in v:
        raise HTTPException(502, "기상청 응답에 기온(T1H)이 없습니다.")

    pty = str(v.get("PTY", "0"))
    condition = PTY_TEXT.get(pty)
    if condition is None:  # 강수 없음 → 하늘상태로 보강, 실패해도 실황은 살린다
        try:
            condition = _sky_condition(nx, ny)
        except HTTPException:
            condition = None
        condition = condition or "강수 없음"

    def num(k):
        try:
            return float(v[k])
        except (KeyError, ValueError, TypeError):
            return None

    try:
        temp = float(v["T1H"])
    except (ValueError, TypeError):
        raise HTTPException(502, f"기온(T1H) 값 해석 실패: {v['T1H']!r}")
    return {
        "timestamp": base.isoformat(timespec="minutes"),
        "temperature": temp,
        "condition": condition,
        "humidity": num("REH"),
        "wind_speed": num("WSD"),
        "rain_1h": num("RN1"),
        "nx": nx, "ny": ny,
    }


# ───────────────────────── API ─────────────────────────
def _db_latest(nx: int, ny: int) -> dict | None:
    with closing(get_db()) as conn:
        r = conn.execute(
            "SELECT timestamp, temperature, condition, humidity, wind_speed, rain_1h, nx, ny "
            "FROM weather_logs WHERE nx=? AND ny=? ORDER BY timestamp DESC LIMIT 1",
            (nx, ny)).fetchone()
    return dict(r) if r else None


@app.get("/api/weather")
def api_weather(nx: int = SEOUL_NX, ny: int = SEOUL_NY):
    """기상청 실시간 날씨를 가져와 DB 에 적재하고 JSON 으로 반환. 실패 시 캐시/DB 로 폴백."""
    location = "서울" if (nx, ny) == (SEOUL_NX, SEOUL_NY) else f"격자({nx},{ny})"
    try:
        w = fetch_current_weather(nx, ny)
    except Exception as e:  # HTTPException 포함, 예기치 못한 파싱 오류도 폴백 대상
        reason = e.detail if isinstance(e, HTTPException) else f"예기치 못한 오류({type(e).__name__})"
        log.error("기상청 조회 실패, 폴백 시도: %s", reason)
        hit = _cache.get((nx, ny))
        if hit and time.time() - hit[0] <= CACHE_TTL:
            src, data, age = "memory_cache", dict(hit[1]), int(time.time() - hit[0])
        else:
            data, src, age = _db_latest(nx, ny), "db", None
            if data is None and hit:
                src, data, age = "memory_cache", dict(hit[1]), int(time.time() - hit[0])
        if data is None:
            raise HTTPException(503, f"기상청 API 실패, 사용 가능한 캐시 없음: {reason}")
        data.update(saved=False, location=location, stale=True, cached=True,
                    source=src, cache_age_sec=age, error=str(reason))
        return data
    w["saved"] = save_log(w)
    w["location"] = location
    _cache[(nx, ny)] = (time.time(), dict(w))
    w.update(stale=False, cached=False, source="live")
    return w


@app.get("/api/weather/history")
def api_history(limit: int = Query(24, ge=1, le=500)):
    """DB 에 쌓인 최근 기록 (최신순)."""
    with closing(get_db()) as conn:
        rows = conn.execute(
            "SELECT timestamp, temperature, condition, humidity, wind_speed, rain_1h "
            "FROM weather_logs ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/health")
def health():
    return {"status": "ok",
            "scheduler": {"running": bool(_scheduler and _scheduler.running),
                          "interval_min": SCHEDULER_INTERVAL_MIN,
                          "last_run": _last_run["time"], "last_ok": _last_run["ok"],
                          "last_message": _last_run["message"]}}


# ───────────────────────── Frontend ─────────────────────────
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
