"""기상청(KMA) 단기예보 조회서비스 연동 + SQLite(weather_logs.db) 적재.

- 지금 ~ 직전 3시간 : 초단기실황 getUltraSrtNcst (관측값)
- 이후 ~ 약 2.5일   : 단기예보 getVilageFcst
인증키는 환경변수 KMA_SERVICE_KEY 또는 프로젝트 루트의 .env 에서 읽는다.
"""
import math
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "weather_logs.db"
KST = ZoneInfo("Asia/Seoul")

BASE = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0"
NCST_URL = f"{BASE}/getUltraSrtNcst"   # 초단기실황
FCST_URL = f"{BASE}/getVilageFcst"     # 단기예보
SEOUL_NX, SEOUL_NY = 55, 127

PTY_TEXT = {"1": "비", "2": "비/눈", "3": "눈", "4": "소나기",
            "5": "빗방울", "6": "빗방울/눈날림", "7": "눈날림"}
SKY_TEXT = {"1": "맑음", "3": "구름많음", "4": "흐림"}
BASE_HOURS = (2, 5, 8, 11, 14, 17, 20, 23)   # 단기예보 발표 시각


class KMAError(RuntimeError):
    """기상청 조회 실패 / 범위 밖. 호출 측에서 다른 소스로 대체한다."""


def _load_dotenv() -> None:
    p = ROOT / ".env"
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


# ───────────────────────── 좌표 변환 (위경도 → 기상청 격자) ─────────────────────────
def latlon_to_grid(lat: float, lon: float) -> tuple[int, int]:
    """기상청 공식 Lambert Conformal Conic 변환."""
    if not (32.0 <= lat <= 39.5 and 123.5 <= lon <= 132.0):
        raise KMAError("한반도 밖 좌표는 기상청 격자를 지원하지 않습니다.")
    RE, GRID, SLAT1, SLAT2, OLON, OLAT, XO, YO = 6371.00877, 5.0, 30.0, 60.0, 126.0, 38.0, 43, 136
    d = math.pi / 180.0
    re = RE / GRID
    slat1, slat2, olon, olat = SLAT1 * d, SLAT2 * d, OLON * d, OLAT * d
    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = math.pow(sf, sn) * math.cos(slat1) / sn
    ro = re * sf / math.pow(math.tan(math.pi * 0.25 + olat * 0.5), sn)
    ra = re * sf / math.pow(math.tan(math.pi * 0.25 + lat * d * 0.5), sn)
    theta = lon * d - olon
    theta = (theta + math.pi) % (2 * math.pi) - math.pi
    theta *= sn
    return int(ra * math.sin(theta) + XO + 0.5), int(ro - ra * math.cos(theta) + YO + 0.5)


def grid_for(lat: float, lon: float) -> tuple[int, int]:
    """서울 도심 좌표는 요구사항의 격자(nx=55, ny=127)로, 그 외는 위경도에서 변환."""
    if abs(lat - 37.5665) < 0.02 and abs(lon - 126.9780) < 0.02:
        return SEOUL_NX, SEOUL_NY
    return latlon_to_grid(lat, lon)


# ───────────────────────── API 호출 ─────────────────────────
def _call(url: str, params: dict, rows: int = 100) -> list:
    key = os.environ.get("KMA_SERVICE_KEY", "").strip()
    if not key:
        raise KMAError("KMA_SERVICE_KEY 가 설정되지 않았습니다 (.env 확인).")
    q = {"serviceKey": key, "dataType": "JSON", "pageNo": 1, "numOfRows": rows, **params}
    try:
        r = requests.get(url, params=q, timeout=10)
        r.raise_for_status()
        body = r.json()["response"]
    except requests.RequestException as e:
        raise KMAError(f"기상청 API 호출 실패: {e}")
    except (ValueError, KeyError):
        raise KMAError(f"기상청 API 응답 해석 실패: {r.text[:150]}")
    if body["header"]["resultCode"] != "00":
        raise KMAError(f"기상청 API 오류 {body['header']['resultCode']}: {body['header']['resultMsg']}")
    return body["body"]["items"]["item"]


@lru_cache(maxsize=64)
def _ncst(nx: int, ny: int, base_date: str, base_time: str) -> dict:
    items = _call(NCST_URL, {"base_date": base_date, "base_time": base_time, "nx": nx, "ny": ny})
    return {i["category"]: i["obsrValue"] for i in items}


@lru_cache(maxsize=64)
def _vilage(nx: int, ny: int, base_date: str, base_time: str) -> dict:
    """{(fcstDate, fcstTime): {category: value}}"""
    items = _call(FCST_URL, {"base_date": base_date, "base_time": base_time, "nx": nx, "ny": ny}, rows=1500)
    out: dict = {}
    for i in items:
        out.setdefault((i["fcstDate"], i["fcstTime"]), {})[i["category"]] = i["fcstValue"]
    return out


def _vilage_base(now: datetime) -> datetime:
    """제공 가능한 가장 최근 단기예보 발표 시각 (발표 후 약 10분 뒤부터 조회 가능)."""
    t = now - timedelta(minutes=10)
    hours = [h for h in BASE_HOURS if h <= t.hour]
    if hours:
        return t.replace(hour=hours[-1], minute=0, second=0, microsecond=0)
    return (t - timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)


def _condition(sky, pty) -> str:
    return PTY_TEXT.get(str(pty)) or SKY_TEXT.get(str(sky)) or "정보 없음"


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fetch_weather(nx: int, ny: int, when: datetime | None = None) -> dict:
    """`when`(KST naive, 기본=지금) 시각의 격자 날씨.

    반환: {temp_c, humidity, wind_ms, rain_prob, condition, timestamp, kind('observed'|'forecast')}
    """
    now = datetime.now(KST).replace(tzinfo=None)
    when = when or now
    delta_h = (when - now).total_seconds() / 3600

    # 1) 실황 (관측값): 매시 정각 기준, 약 40분 뒤 제공
    if -3 <= delta_h <= 1:
        base = (now - timedelta(minutes=40)).replace(minute=0, second=0, microsecond=0)
        v = _ncst(nx, ny, base.strftime("%Y%m%d"), base.strftime("%H%M"))
        if "T1H" not in v:
            raise KMAError("실황 응답에 기온(T1H)이 없습니다.")
        sky, pop = None, None
        try:  # 하늘상태·강수확률은 단기예보에서 보강 (실패해도 실황은 유지)
            vb = _vilage_base(now)
            row = _vilage(nx, ny, vb.strftime("%Y%m%d"), vb.strftime("%H%M")).get(
                (now.strftime("%Y%m%d"), now.strftime("%H00")), {})
            sky, pop = row.get("SKY"), _f(row.get("POP"))
        except KMAError:
            pass
        pty = v.get("PTY", "0")
        return {
            "temp_c": float(v["T1H"]), "humidity": _f(v.get("REH")), "wind_ms": _f(v.get("WSD")) or 0.0,
            "rain_prob": 100.0 if str(pty) != "0" else (pop if pop is not None else 0.0),
            "condition": _condition(sky, pty) if (str(pty) != "0" or sky) else "강수 없음",
            "timestamp": base.isoformat(timespec="minutes"), "kind": "observed",
        }

    # 2) 단기예보
    if 1 < delta_h <= 60:
        vb = _vilage_base(now)
        row = _vilage(nx, ny, vb.strftime("%Y%m%d"), vb.strftime("%H%M")).get(
            (when.strftime("%Y%m%d"), when.strftime("%H00")))
        if not row or "TMP" not in row:
            raise KMAError("해당 시각의 단기예보가 없습니다.")
        return {
            "temp_c": float(row["TMP"]), "humidity": _f(row.get("REH")), "wind_ms": _f(row.get("WSD")) or 0.0,
            "rain_prob": _f(row.get("POP")) or 0.0, "condition": _condition(row.get("SKY"), row.get("PTY")),
            "timestamp": when.replace(minute=0).isoformat(timespec="minutes"), "kind": "forecast",
        }

    raise KMAError("기상청 조회 가능 범위(직전 3시간 ~ 약 2.5일)를 벗어났습니다.")


def apparent_temp(t: float, rh: float, ws_ms: float) -> float:
    """체감온도(Steadman/BoM 근사)."""
    e = (rh / 100.0) * 6.105 * math.exp(17.27 * t / (237.7 + t))
    return round(t + 0.33 * e - 0.70 * ws_ms - 4.00, 1)


# ───────────────────────── SQLite ─────────────────────────
def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with closing(_db()) as c, c:
        c.execute("""CREATE TABLE IF NOT EXISTS weather_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL, temperature REAL NOT NULL, condition TEXT NOT NULL,
            humidity REAL, wind_speed REAL, rain_prob REAL,
            nx INTEGER NOT NULL, ny INTEGER NOT NULL, created_at TEXT NOT NULL,
            UNIQUE (timestamp, nx, ny))""")


def save_log(w: dict, nx: int, ny: int) -> bool:
    """실황(관측값)만 적재. 같은 관측시각·격자는 중복 저장하지 않는다."""
    if w.get("kind") != "observed":
        return False
    with closing(_db()) as c, c:
        cur = c.execute(
            "INSERT OR IGNORE INTO weather_logs (timestamp,temperature,condition,humidity,wind_speed,"
            "rain_prob,nx,ny,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (w["timestamp"], w["temp_c"], w["condition"], w["humidity"], w["wind_ms"], w["rain_prob"],
             nx, ny, datetime.now(KST).isoformat(timespec="seconds")))
        return cur.rowcount == 1


def recent_logs(limit: int = 24) -> list[dict]:
    with closing(_db()) as c:
        rows = c.execute("SELECT timestamp,temperature,condition,humidity,wind_speed,rain_prob "
                         "FROM weather_logs ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


init_db()
