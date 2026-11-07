"""RunWeather Agent - Runner's Weather & Pace Weather Matching Dashboard (Session 1)

Run:  uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""
import math
import threading
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import kma
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
# Windows is case-insensitive, Linux sandboxes are not: accept either folder name.
IMG_DIR = next((p for p in (BASE_DIR / "images", BASE_DIR / "Images") if p.is_dir()), BASE_DIR / "images")
IMG_DIR.mkdir(exist_ok=True)

app = FastAPI(title="RunWeather Agent")
app.mount("/images", StaticFiles(directory=str(IMG_DIR)), name="images")

SOURCE = "Korea Meteorological Administration (KMA)"   # 날씨 데이터 출처 표기
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"  # 장소명 -> 좌표 검색 전용 (날씨 데이터 아님)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"     # 도로명/지번 주소 -> 좌표 (OpenStreetMap)
OSRM_FOOT_URL = "https://routing.openstreetmap.de/routed-foot/route/v1/foot/"  # 도보 경로 (FOSSGIS OSRM)
HTTP_HEADERS = {"User-Agent": "RunWeatherAgent/0.1 (running weather dashboard)"}


# ───────────────────────── Models ─────────────────────────
class Checkpoint(BaseModel):
    time: str = Field(..., description="ISO local time, e.g. 2026-11-08T06:30")
    location: str = Field(..., description="Place name, e.g. 'Seoul' (ignored if lat/lon given)")
    distance_km: float = Field(0, ge=0, description="Cumulative distance at this checkpoint")
    lat: Optional[float] = None
    lon: Optional[float] = None


class RunPlan(BaseModel):
    checkpoints: List[Checkpoint]


class RouteRequest(BaseModel):
    points: List[Tuple[float, float]] = Field(..., description="[[lat, lon], ...] 출발→도착 순서")
    follow_roads: bool = True


# ───────────────────────── Weather helpers ─────────────────────────
# Open-Meteo 지오코딩은 한글 지명("서울" 등)에서 결과가 없거나 엉뚱한 곳을 돌려주는 경우가 있어
# 주요 도시는 좌표를 직접 둔다. 목록에 없으면 아래 지오코딩으로 조회한다.
KR_CITIES = {
    "서울": (37.5665, 126.9780), "인천": (37.4563, 126.7052), "수원": (37.2636, 127.0286),
    "성남": (37.4200, 127.1267), "고양": (37.6584, 126.8320), "용인": (37.2411, 127.1776),
    "춘천": (37.8813, 127.7298), "강릉": (37.7519, 128.8761), "대전": (36.3504, 127.3845),
    "세종": (36.4800, 127.2890), "청주": (36.6424, 127.4890), "전주": (35.8242, 127.1480),
    "광주": (35.1595, 126.8526), "대구": (35.8714, 128.6014), "울산": (35.5384, 129.3114),
    "부산": (35.1796, 129.0756), "창원": (35.2280, 128.6811), "제주": (33.4996, 126.5312),
}


_nominatim_lock = threading.Lock()
_nominatim_last = 0.0


def _nominatim(q: str) -> Optional[dict]:
    """주소 검색. Nominatim 이용 정책(초당 1회)을 지키도록 호출 간격을 둔다."""
    global _nominatim_last
    with _nominatim_lock:
        wait = 1.0 - (time.monotonic() - _nominatim_last)
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(NOMINATIM_URL, headers=HTTP_HEADERS, timeout=10, params={
                "q": q, "format": "jsonv2", "limit": 1, "accept-language": "ko"})
        finally:
            _nominatim_last = time.monotonic()
    r.raise_for_status()
    res = r.json()
    if not res:
        return None
    return {"lat": float(res[0]["lat"]), "lon": float(res[0]["lon"]), "display": res[0].get("display_name", q)}


@lru_cache(maxsize=256)
def geocode(name: str) -> dict:
    """Place name/address -> {lat, lon, name, display}.

    주요 도시는 내장 좌표 → 주소·장소명은 Nominatim(OpenStreetMap) → 그래도 없으면 Open-Meteo 지명 검색.
    """
    q = name.strip()
    key = q.removesuffix("특별시").removesuffix("광역시").removesuffix("시")
    if key in KR_CITIES:
        lat, lon = KR_CITIES[key]
        return {"lat": lat, "lon": lon, "name": key, "display": key}
    hit = _nominatim(q)
    if not hit and " " in q:  # '여의도 한강공원' → OSM 표기 '여의도한강공원'
        hit = _nominatim(q.replace(" ", ""))
    if hit:
        return {**hit, "name": q}
    r = requests.get(GEOCODE_URL, params={"name": q, "count": 1, "language": "ko"}, timeout=10)
    r.raise_for_status()
    results = r.json().get("results")
    if not results:
        raise ValueError(f"장소를 찾을 수 없습니다: {name}")
    g = results[0]
    return {"lat": g["latitude"], "lon": g["longitude"], "name": g.get("name", q), "display": g.get("name", q)}


def _haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    (lat1, lon1), (lat2, lon2) = a, b
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


@lru_cache(maxsize=128)
def plan_route(points: Tuple[Tuple[float, float], ...], follow_roads: bool) -> dict:
    """[(lat, lon), ...] 를 잇는 런닝 경로. 도보 도로를 따르고, 실패하면 직선으로 대체.

    반환: {coordinates: [[lat, lon], ...], legs_km: [구간별 km], distance_km, mode: 'road'|'straight'}
    """
    if follow_roads:
        try:
            coords = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in points)
            r = requests.get(OSRM_FOOT_URL + coords, headers=HTTP_HEADERS, timeout=15,
                             params={"overview": "full", "geometries": "geojson"})
            r.raise_for_status()
            body = r.json()
            if body.get("code") == "Ok" and body.get("routes"):
                rt = body["routes"][0]
                legs = [round(leg["distance"] / 1000, 3) for leg in rt["legs"]]
                return {"coordinates": [[la, lo] for lo, la in rt["geometry"]["coordinates"]],
                        "legs_km": legs, "distance_km": round(sum(legs), 2), "mode": "road"}
        except (requests.RequestException, ValueError, KeyError):
            pass
    legs = [round(_haversine_km(points[i], points[i + 1]), 3) for i in range(len(points) - 1)]
    return {"coordinates": [list(p) for p in points], "legs_km": legs,
            "distance_km": round(sum(legs), 2), "mode": "straight"}


def fetch_forecast(lat: float, lon: float, when: datetime) -> dict:
    """`when` 시각의 날씨 (기상청 KMA). 서울은 nx=55, ny=127 기준.

    지금~직전 3시간은 초단기실황(getUltraSrtNcst), 이후 약 2.5일은 단기예보.
    조회 실패/범위 밖이면 kma.KMAError 를 그대로 올린다 (다른 소스로 대체하지 않음).
    """
    nx, ny = kma.grid_for(lat, lon)
    k = kma.fetch_weather(nx, ny, when)
    kma.save_log(k, nx, ny)  # 실황(관측값)이면 weather_logs 에 적재
    humidity = k["humidity"] if k["humidity"] is not None else 50.0
    return {
        "temp_c": k["temp_c"],
        "feels_like_c": kma.apparent_temp(k["temp_c"], humidity, k["wind_ms"]),
        "humidity": humidity,
        "wind_kmh": round(k["wind_ms"] * 3.6, 1),
        "rain_prob": k["rain_prob"],
        "condition": k["condition"],
        "source": SOURCE,
        "grid": {"nx": nx, "ny": ny},
    }


def running_index(w: dict) -> dict:
    """0-100 score: penalise deviation from ~10°C feels-like, humidity, wind, rain."""
    score = 100.0
    score -= abs(w["feels_like_c"] - 10) * 2.5
    score -= max(0, w["humidity"] - 60) * 0.4
    score -= max(0, w["wind_kmh"] - 10) * 1.2
    score -= (w["rain_prob"] or 0) * 0.5
    score = max(0, min(100, round(score)))
    label = "최적" if score >= 80 else "좋음" if score >= 60 else "보통" if score >= 40 else "주의"
    return {"score": score, "label": label}


def outfit_guide(feels: float, rain: float, wind: float) -> str:
    if feels >= 25: base = "민소매 + 반바지, 수분 보충 필수"
    elif feels >= 18: base = "반팔 + 반바지"
    elif feels >= 10: base = "반팔 + 얇은 긴팔/윈드 베스트, 러닝 타이츠 선택"
    elif feels >= 3: base = "긴팔 + 바람막이, 장갑"
    else: base = "기모 상의 + 방풍 재킷 + 타이츠, 장갑/넥워머"
    if rain >= 50: base += " · 방수 캡/재킷"
    if wind >= 25: base += " · 맞바람 구간 주의"
    return base


# ───────────────────────── API ─────────────────────────
@app.post("/api/run-weather")
def run_weather(plan: RunPlan):
    if not plan.checkpoints:
        raise HTTPException(400, "checkpoints is empty")
    out = []
    for cp in plan.checkpoints:
        try:
            when = datetime.fromisoformat(cp.time)
        except ValueError:
            raise HTTPException(422, f"시간 형식이 올바르지 않습니다: '{cp.time}' (예: 2026-11-08T06:30)")
        if when.tzinfo is not None:  # '+09:00' 등 시간대가 붙은 입력은 KST 로 맞춘다
            when = when.astimezone(kma.KST).replace(tzinfo=None)
        try:
            if cp.lat is not None and cp.lon is not None:
                place = {"lat": cp.lat, "lon": cp.lon, "name": cp.location}
            else:
                place = geocode(cp.location)
            w = fetch_forecast(place["lat"], place["lon"], when)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except kma.KMAError as e:
            raise HTTPException(502, f"기상청(KMA) 조회 실패: {e}")
        except requests.RequestException as e:
            raise HTTPException(502, f"장소 검색 API 오류: {e}")
        out.append({
            "time": cp.time, "location": place["name"], "distance_km": cp.distance_km,
            "lat": place["lat"], "lon": place["lon"], "weather": w,
            "run_index": running_index(w),
            "outfit": outfit_guide(w["feels_like_c"], w["rain_prob"] or 0, w["wind_kmh"]),
        })
    avg = round(sum(c["run_index"]["score"] for c in out) / len(out))
    return {"count": len(out), "avg_run_index": avg, "source": SOURCE, "checkpoints": out}


@app.get("/api/weather")
def api_weather(nx: int = kma.SEOUL_NX, ny: int = kma.SEOUL_NY):
    """기상청 실황(기본: 서울 nx=55, ny=127)을 가져와 weather_logs 에 적재하고 JSON 으로 반환."""
    try:
        k = kma.fetch_weather(nx, ny)
    except kma.KMAError as e:
        raise HTTPException(502, str(e))
    return {**k, "nx": nx, "ny": ny, "source": SOURCE, "saved": kma.save_log(k, nx, ny)}


@app.get("/api/weather/history")
def api_weather_history(limit: int = 24):
    return kma.recent_logs(max(1, min(limit, 500)))


@app.get("/api/geocode")
def api_geocode(q: str):
    if not q.strip():
        raise HTTPException(400, "q is empty")
    try:
        return geocode(q.strip())
    except ValueError as e:
        raise HTTPException(404, str(e))
    except requests.RequestException as e:
        raise HTTPException(502, f"장소 검색 API 오류: {e}")


@app.post("/api/route")
def api_route(req: RouteRequest):
    if len(req.points) < 2:
        raise HTTPException(400, "경로에는 2개 이상의 지점이 필요합니다.")
    if len(req.points) > 50:
        raise HTTPException(400, "지점은 최대 50개까지 지원합니다.")
    pts = tuple((round(la, 6), round(lo, 6)) for la, lo in req.points)
    return plan_route(pts, req.follow_roads)


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ───────────────────────── Dashboard ─────────────────────────
PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RunWeather Agent</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{box-sizing:border-box}
body{margin:0;min-height:100vh;font-family:'Segoe UI',system-ui,sans-serif;color:#e2e8f0;padding:32px 16px;
 /* running_bg.jpg first; falls back to any existing image if the file is missing */
 background-image:url('/images/running_bg.jpg'),url('/images/Running.png');
 background-color:#0f172a;background-size:cover;background-position:center;background-attachment:fixed}
.container{max-width:1200px;margin:0 auto;background:rgba(15,23,42,.85);border-radius:20px;
 padding:28px;box-shadow:0 20px 60px rgba(0,0,0,.5);backdrop-filter:blur(6px)}
header{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;
 border-bottom:1px solid rgba(148,163,184,.2);padding-bottom:18px;margin-bottom:22px}
h1{margin:0;font-size:28px;letter-spacing:.5px;color:#fff}
.sub{color:#94a3b8;font-size:14px;margin-top:4px}
.badge{background:rgba(34,197,94,.15);color:#4ade80;border:1px solid rgba(34,197,94,.4);
 padding:6px 14px;border-radius:999px;font-size:13px;font-weight:600}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
.panel{background:rgba(30,41,59,.7);border-radius:14px;padding:20px;margin-bottom:22px}
.panel h2{margin:0 0 14px;font-size:16px;color:#f97316;text-transform:uppercase;letter-spacing:1px}
label{display:block;font-size:12px;color:#94a3b8;margin:10px 0 4px}
input{width:100%;padding:9px 11px;border-radius:8px;border:1px solid #334155;background:#0f172a;color:#e2e8f0}
.row{display:grid;grid-template-columns:minmax(13.5rem,1.5fr) 1.3fr 5.5rem auto;gap:8px;align-items:end;margin-bottom:6px}
button{cursor:pointer;border:0;border-radius:8px;padding:10px 16px;font-weight:600;color:#fff;background:#f97316}
button.sec{background:#334155}
button:hover{filter:brightness(1.1)}
.tl{border-left:3px solid #f97316;padding:8px 0 8px 14px;margin:0 0 12px 6px;cursor:pointer}
.tl:hover,.tl.sel{background:rgba(249,115,22,.08)}
.tl b{color:#fff}.tl small{color:#94a3b8}
.score{float:right;font-weight:700;color:#4ade80}
.stat{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(148,163,184,.12)}
.muted{color:#94a3b8;font-size:14px}
.controls{display:flex;gap:10px;flex-wrap:wrap}
pre{white-space:pre-wrap;margin:0;font-family:inherit;line-height:1.6}
.row.bad .loc input{border-color:#ef4444}
.pin{font-size:11px}
.row>div{min-width:0}
@media(max-width:600px){
 body{padding:16px 10px}
 .container{padding:16px}
 .row{grid-template-columns:1fr 5.5rem auto;padding-bottom:10px;border-bottom:1px solid rgba(148,163,184,.12)}
 .row>div:first-child{grid-column:1/-1}
 #map{height:340px}
}
.planbtns{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px}
#mapbox{margin-top:14px}
#map{height:400px;border-radius:10px;border:1px solid #334155}
.mapbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:8px;font-size:13px}
.mapbar input{width:auto;margin:0 4px 0 0}
.mapbar button{padding:6px 10px;font-size:12px}
#mapinfo b{color:#4ade80}
.hint{font-size:12px;color:#94a3b8;margin-top:6px}
.cpm{width:26px;height:26px;border-radius:50%;background:#f97316;color:#fff;font-weight:700;font-size:13px;
 display:flex;align-items:center;justify-content:center;border:2px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,.5)}
.via{width:14px;height:14px;border-radius:50%;background:#fff;border:3px solid #f97316;box-shadow:0 1px 3px rgba(0,0,0,.5)}
</style></head><body>
<div class="container">
 <header>
  <div><h1>RunWeather Agent</h1><div class="sub">Pace &amp; Location Weather Intelligence Dashboard</div></div>
  <span class="badge">⚡ GPS &amp; Weather Sync Active</span>
 </header>

 <div class="grid">
  <section>
   <div class="panel">
    <h2>Run Plan</h2>
    <div id="rows"></div>
    <div class="planbtns">
     <button class="sec" onclick="addSegment()">+ 구간 추가</button>
     <button onclick="toggleMap()">🗺 런닝 구간</button>
    </div>
    <div id="mapbox" hidden>
     <div class="mapbar">
      <label style="display:flex;align-items:center;margin:0;color:#e2e8f0"><input type="checkbox" id="roads" checked onchange="scheduleRoute(0)">도로 따라가기</label>
      <button class="sec" onclick="fitRoute()">🎯 전체 보기</button>
      <button class="sec" onclick="clearVias()">경유지 초기화</button>
      <span id="mapinfo" class="muted"></span>
     </div>
     <div id="map"></div>
     <div class="hint">번호 마커를 끌어 구간 위치를 옮기고, 지도를 클릭하면 경유지가 추가됩니다 (경유지는 끌어서 이동, 더블클릭하면 삭제). 구간별 누적 거리는 Run Plan에 자동 반영됩니다.</div>
    </div>
   </div>
   <div class="panel"><h2>Weather Timeline</h2><div id="timeline" class="muted">날씨 동기화를 실행하세요.</div></div>
  </section>

  <section>
   <div class="panel"><h2>Segment Analysis</h2><div id="analysis" class="muted">타임라인에서 구간을 선택하세요.</div></div>
   <div class="panel"><h2>AI Condition Advisor</h2><pre id="advisor" class="muted">아직 리포트가 없습니다.</pre></div>
  </section>
 </div>

 <div class="panel" style="margin-bottom:0">
  <h2>Manual Control Panel</h2>
  <div class="controls">
   <button onclick="sync()">🔄 날씨 데이터 동기화</button>
   <button class="sec" onclick="report()">📝 런닝 리포트 생성</button>
  </div>
  <div id="status" class="muted" style="margin-top:10px"></div>
 </div>
</div>

<script>
let data=null;
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const rowsEl=()=>[...document.querySelectorAll('#rows .row')];
const hasPos=r=>'lat' in r.dataset;
function setPin(r){
  const p=r.querySelector('.pin');
  p.textContent=hasPos(r)?'📍':'';
  p.title=hasPos(r)?`좌표 ${(+r.dataset.lat).toFixed(5)}, ${(+r.dataset.lon).toFixed(5)}`:'';
}
function addRow(t='',l='서울',d=0,lat=null,lon=null){
  const el=document.createElement('div');el.className='row';
  el.innerHTML=`<div><label>시간</label><input type="datetime-local"></div>
   <div class="loc"><label>장소/주소 <span class="pin"></span></label><input></div>
   <div><label>거리(km)</label><input type="number" step="0.01"></div>
   <button class="sec">✕</button>`;
  const [ti,li,di]=el.querySelectorAll('input');
  ti.value=t;li.value=l;di.value=d;
  if(lat!=null){el.dataset.lat=lat;el.dataset.lon=lon;}
  li.addEventListener('change',()=>{delete el.dataset.lat;delete el.dataset.lon;setPin(el);if(mapOpen())refreshMap();});
  ti.addEventListener('change',()=>{if(mapOpen())drawMarkers();});
  el.querySelector('button').onclick=()=>{el.remove();if(mapOpen())refreshMap();};
  $('rows').appendChild(el);setPin(el);
}
const pad=n=>String(n).padStart(2,'0');
const fmt=d=>`${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
function ts(h){const d=new Date();d.setDate(d.getDate()+1);d.setHours(h,0,0,0);return fmt(d);}
function addSegment(){
  const last=rowsEl().pop();
  if(!last){addRow(ts(6),'서울',0);}
  else{
    // 마지막 구간을 이어받아 30분 뒤, 근처 위치로 새 구간을 만든다 (지도에서 끌어서 조정)
    const [ti,li,di]=last.querySelectorAll('input');
    let t='';if(ti.value){const d=new Date(ti.value);d.setMinutes(d.getMinutes()+30);t=fmt(d);}
    addRow(t,li.value,di.value,hasPos(last)?+last.dataset.lat+0.004:null,hasPos(last)?+last.dataset.lon+0.006:null);
  }
  if(mapOpen()){needFit=true;refreshMap();}
}
addRow(ts(6),'서울',0);addRow(ts(7),'인천',10);

// ───── 런닝 구간 지도 ─────
let map=null,line=null,markers=null,seq=[],routeTimer=null,routeReq=0,refreshReq=0,needFit=true;
const mapOpen=()=>map&&!$('mapbox').hidden;
function toggleMap(){
  const box=$('mapbox');
  if(!box.hidden){box.hidden=true;return;}
  box.hidden=false;
  if(typeof L==='undefined'){$('mapinfo').textContent='지도 라이브러리(Leaflet)를 불러오지 못했습니다.';return;}
  if(!map){
    map=L.map('map',{doubleClickZoom:false}).setView([37.5665,126.978],12);
    // tile.openstreetmap.org 는 Codespaces 등에서 접속 시 403(Access blocked)을 돌려줄 수 있어 Esri 타일을 쓴다 (CARTO 는 API 키 필요)
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
      {maxZoom:19,referrerPolicy:'strict-origin-when-cross-origin',
       attribution:'Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors, and the GIS User Community'}).addTo(map);
    line=L.polyline([],{color:'#f97316',weight:5,opacity:.85}).addTo(map);
    markers=L.layerGroup().addTo(map);
    map.on('click',e=>addVia(e.latlng));
  }
  needFit=true;
  setTimeout(()=>map.invalidateSize(),0);
  refreshMap();
}
// seq: 경로 순서대로의 지점 목록. {kind:'cp',row} = Run Plan 구간, {kind:'via',lat,lon} = 경유지
function pts(){
  return seq.map(it=>it.kind==='cp'
    ?(hasPos(it.row)?{item:it,lat:+it.row.dataset.lat,lon:+it.row.dataset.lon}:null)
    :{item:it,lat:it.lat,lon:it.lon}).filter(Boolean);
}
async function refreshMap(){
  const my=++refreshReq,rs=rowsEl();
  seq=seq.filter(it=>it.kind==='via'||rs.includes(it.row));
  rs.forEach(r=>{if(!seq.some(it=>it.row===r))seq.push({kind:'cp',row:r});});
  const failed=[];
  for(const r of rs){
    if(hasPos(r))continue;
    const q=r.querySelectorAll('input')[1].value.trim();
    if(!q){failed.push('(빈 장소)');continue;}
    $('mapinfo').textContent=`주소 검색 중: ${q}`;
    try{
      const res=await fetch('/api/geocode?q='+encodeURIComponent(q));
      if(!res.ok)throw new Error((await res.json()).detail||res.status);
      const g=await res.json();
      if(!r.isConnected||r.querySelectorAll('input')[1].value.trim()!==q)continue;
      r.dataset.lat=g.lat;r.dataset.lon=g.lon;r.classList.remove('bad');setPin(r);
    }catch(e){r.classList.add('bad');failed.push(q);}
    if(my!==refreshReq)return;
  }
  drawMarkers();
  if(needFit){fitRoute();needFit=false;}
  await updateRoute();
  if(failed.length)$('mapinfo').textContent+=` · 찾지 못한 장소: ${failed.join(', ')}`;
}
function drawMarkers(){
  markers.clearLayers();
  const rs=rowsEl();
  pts().forEach(p=>{
    if(p.item.kind==='cp'){
      const r=p.item.row,n=rs.indexOf(r)+1,[ti,li]=r.querySelectorAll('input');
      L.marker([p.lat,p.lon],{draggable:true,icon:L.divIcon({className:'',html:`<div class="cpm">${n}</div>`,iconSize:[26,26],iconAnchor:[13,13]})})
       .bindTooltip(`<b>${n}. ${esc(li.value)}</b><br>${esc(ti.value.replace('T',' '))}`)
       .on('dragend',e=>{const ll=e.target.getLatLng();r.dataset.lat=ll.lat.toFixed(6);r.dataset.lon=ll.lng.toFixed(6);r.classList.remove('bad');setPin(r);scheduleRoute();})
       .addTo(markers);
    }else{
      const it=p.item;
      L.marker([p.lat,p.lon],{draggable:true,icon:L.divIcon({className:'',html:'<div class="via"></div>',iconSize:[14,14],iconAnchor:[7,7]})})
       .bindTooltip('경유지 · 더블클릭하면 삭제')
       .on('dragend',e=>{const ll=e.target.getLatLng();it.lat=ll.lat;it.lon=ll.lng;scheduleRoute();})
       .on('dblclick',()=>{seq=seq.filter(x=>x!==it);drawMarkers();scheduleRoute(0);})
       .addTo(markers);
    }
  });
}
function addVia(ll){
  const P=pts(),item={kind:'via',lat:ll.lat,lon:ll.lng};
  if(P.length<2){seq.push(item);}
  else{ // 클릭 위치와 가장 가까운 구간 사이에 경유지를 끼워 넣는다
    const c=map.latLngToLayerPoint(ll);let best=0,bd=Infinity;
    for(let i=0;i<P.length-1;i++){
      const d=L.LineUtil.pointToSegmentDistance(c,map.latLngToLayerPoint([P[i].lat,P[i].lon]),map.latLngToLayerPoint([P[i+1].lat,P[i+1].lon]));
      if(d<bd){bd=d;best=i;}
    }
    seq.splice(seq.indexOf(P[best].item)+1,0,item);
  }
  drawMarkers();scheduleRoute(0);
}
function clearVias(){seq=seq.filter(it=>it.kind==='cp');if(map){drawMarkers();scheduleRoute(0);}}
function fitRoute(){
  if(!map)return;
  const ll=[...line.getLatLngs(),...pts().map(p=>L.latLng(p.lat,p.lon))];
  if(ll.length===1)map.setView(ll[0],15);
  else if(ll.length)map.fitBounds(L.latLngBounds(ll),{padding:[30,30]});
}
function scheduleRoute(ms=250){clearTimeout(routeTimer);routeTimer=setTimeout(updateRoute,ms);}
async function updateRoute(){
  const P=pts(),my=++routeReq;
  if(P.length<2){line.setLatLngs([]);
    $('mapinfo').textContent=P.length?'지점이 하나뿐입니다. 지도를 클릭하거나 + 구간 추가로 경로를 만드세요.':'표시할 지점이 없습니다.';return;}
  $('mapinfo').textContent='경로 계산 중...';
  try{
    const res=await fetch('/api/route',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({points:P.map(p=>[p.lat,p.lon]),follow_roads:$('roads').checked})});
    if(!res.ok)throw new Error((await res.json()).detail||res.status);
    const r=await res.json();if(my!==routeReq)return;
    line.setLatLngs(r.coordinates);
    let cum=0;
    P.forEach((p,i)=>{if(i>0)cum+=r.legs_km[i-1];if(p.item.kind==='cp')p.item.row.querySelectorAll('input')[2].value=cum.toFixed(2);});
    $('mapinfo').innerHTML=`총 거리 <b>${r.distance_km.toFixed(2)} km</b> · `+
      (r.mode==='road'?'도보 도로 기준':($('roads').checked?'직선 거리 (도로 경로 조회 실패)':'직선 거리'));
  }catch(e){if(my===routeReq)$('mapinfo').textContent='경로 오류: '+e.message;}
}

async function sync(){
  const checkpoints=rowsEl().map(r=>{
    const i=r.querySelectorAll('input');
    const c={time:i[0].value,location:i[1].value,distance_km:parseFloat(i[2].value)||0};
    if(hasPos(r)){c.lat=+r.dataset.lat;c.lon=+r.dataset.lon;}
    return c;
  }).filter(c=>c.time&&c.location);
  $('status').textContent='동기화 중...';
  try{
    const res=await fetch('/api/run-weather',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({checkpoints})});
    if(!res.ok)throw new Error((await res.json()).detail||res.status);
    data=await res.json();render();
    $('status').textContent='동기화 완료 · '+new Date().toLocaleTimeString()+' · 데이터 출처: '+data.source;
  }catch(e){$('status').textContent='오류: '+e.message;}
}
function render(){
  $('timeline').innerHTML=data.checkpoints.map((c,i)=>`<div class="tl" onclick="pick(${i})">
    <span class="score">${c.run_index.score} ${c.run_index.label}</span>
    <b>${c.time.replace('T',' ')}</b> · ${c.location} <small>(${c.distance_km} km)</small><br>
    <small>${c.weather.temp_c}°C · 습도 ${c.weather.humidity}% · 풍속 ${c.weather.wind_kmh}km/h · 강수 ${c.weather.rain_prob}%</small></div>`).join('');
  pick(0);
}
function pick(i){
  const c=data.checkpoints[i],w=c.weather;
  document.querySelectorAll('.tl').forEach((e,j)=>e.classList.toggle('sel',i===j));
  $('analysis').innerHTML=`<div class="stat"><span>📍 ${c.location}</span><b>${c.distance_km} km</b></div>
   <div class="stat"><span>💨 바람</span><b>${w.wind_kmh} km/h</b></div>
   <div class="stat"><span>🌡 체감온도</span><b>${w.feels_like_c}°C</b></div>
   <div class="stat"><span>💧 습도</span><b>${w.humidity}%</b></div>
   <div class="stat"><span>☔ 강수확률</span><b>${w.rain_prob}%</b></div>
   <div style="margin-top:12px">👕 <b>복장 가이드</b><br><span class="muted">${c.outfit}</span></div>
   <div class="muted" style="margin-top:12px;font-size:12px">📡 ${w.source}</div>`;
}
function report(){
  if(!data){$('status').textContent='먼저 날씨를 동기화하세요.';return;}
  const worst=data.checkpoints.reduce((a,b)=>a.run_index.score<b.run_index.score?a:b);
  $('advisor').className='';
  $('advisor').textContent=`평균 런닝 지수: ${data.avg_run_index}/100\\n`+
   `가장 주의할 구간: ${worst.location} (${worst.time.replace('T',' ')}, ${worst.run_index.score}점)\\n`+
   `추천 복장: ${worst.outfit}\\n데이터 출처: ${data.source}\\n(LLM 기반 어드바이저는 다음 교시에 연동 예정)`;
  $('status').textContent='리포트 생성 완료';
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE
