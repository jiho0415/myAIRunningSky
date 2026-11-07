"""폴백/예외 처리 테스트. 실행: python backend/test_fallback.py  (외부 API 는 mock, 임시 DB 사용)"""
import sys
import tempfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend import main  # noqa: E402

tmp = Path(tempfile.mkdtemp()) / "test.db"
main.DB_PATH = tmp
main.init_db()
main.KMA_RETRY_DELAY = 0
client = TestClient(main.app)
calls = {"n": 0}


class Resp:
    def __init__(self, status=200, js=None, text="x"):
        self.status_code, self._js, self.text = status, js, text

    def json(self):
        if self._js is None:
            raise ValueError("not json")
        return self._js

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


def ok(items):
    return Resp(js={"response": {"header": {"resultCode": "00", "resultMsg": "OK"},
                                 "body": {"items": {"item": items}}}})


GOOD = ok([{"category": c, "obsrValue": v} for c, v in
           [("T1H", "9.0"), ("REH", "16"), ("WSD", "3.7"), ("RN1", "0"), ("PTY", "1")]])


def set_get(fn):
    def g(*a, **k):
        calls["n"] += 1
        return fn()
    main.requests.get = g


main.os.environ["KMA_SERVICE_KEY"] = "dummy-test-key"


def get():
    calls["n"] = 0
    return client.get("/api/weather")


# 1) DB/캐시 없음 + 403 -> 503, 재시도 안 함
set_get(lambda: Resp(403, text="Forbidden"))
r = get(); assert r.status_code == 503, r.text; assert calls["n"] == 1
assert "dummy-test-key" not in r.text

# 2) 정상 -> live, 캐시/DB 저장
set_get(lambda: GOOD)
r = get(); assert r.status_code == 200 and r.json()["stale"] is False and r.json()["source"] == "live"
assert r.json()["condition"] == "비"

# 3) 403 -> 메모리 캐시 폴백
set_get(lambda: Resp(403, text="Forbidden"))
r = get(); j = r.json()
assert r.status_code == 200 and j["stale"] and j["cached"] and j["source"] == "memory_cache" and "403" in j["error"]
assert j["temperature"] == 9.0

# 4) 캐시 만료 -> DB 폴백
main._cache.clear()
r = get(); assert r.json()["source"] == "db" and r.json()["stale"]

# 5) 타임아웃 -> 재시도 상한(3회) 후 폴백
def boom(): raise requests.Timeout("t")
set_get(boom)
r = get(); assert r.status_code == 200 and calls["n"] == main.KMA_RETRIES + 1 and r.json()["stale"]

# 6) resultCode != 00 / XML 응답 / items 누락 / 5xx
set_get(lambda: Resp(js={"response": {"header": {"resultCode": "03", "resultMsg": "NODATA"}}}))
assert "03" in get().json()["error"]
set_get(lambda: Resp(text="<xml>SERVICE_KEY_IS_NOT_REGISTERED</xml>"))
assert "해석" in get().json()["error"]
set_get(lambda: Resp(js={"response": {"header": {"resultCode": "00", "resultMsg": "OK"}, "body": {}}}))
assert "items" in get().json()["error"]
set_get(lambda: Resp(500))
r = get(); assert r.json()["stale"] and calls["n"] == main.KMA_RETRIES + 1

# 7) T1H 누락 / 비정상 값
set_get(lambda: ok([{"category": "REH", "obsrValue": "10"}]))
assert get().json()["stale"]
set_get(lambda: ok([{"category": "T1H", "obsrValue": "-"}]))
assert get().json()["stale"]

# 8) 다른 격자(캐시/DB 없음) + 실패 -> 503
set_get(boom)
assert client.get("/api/weather?nx=1&ny=1").status_code == 503

# 9) 스케줄러 잡: 성공 시 기록, 실패/예외 시 삼키고 last_ok False
main._cache.clear()
set_get(lambda: GOOD)
main.scheduled_fetch()
assert main._last_run["ok"] is True and (main.SEOUL_NX, main.SEOUL_NY) in main._cache
set_get(lambda: Resp(403, text="Forbidden"))
main.scheduled_fetch()  # 예외 전파 없어야 함
assert main._last_run["ok"] is False and "403" in main._last_run["message"]
set_get(boom)
main.scheduled_fetch()
assert main._last_run["ok"] is False
r = get(); assert r.json()["stale"] and r.json()["source"] == "memory_cache"  # 잡 실패 후에도 캐시 폴백

# 10) lifespan: 스케줄러 시작/즉시 실행/health/종료
set_get(lambda: GOOD)
with TestClient(main.app) as c:
    h = c.get("/api/health").json()
    assert h["status"] == "ok" and h["scheduler"]["running"] is True
    import time as _t; _t.sleep(1.5)
    assert c.get("/api/health").json()["scheduler"]["last_ok"] is True
assert main._scheduler is None

print("ALL PASSED")
