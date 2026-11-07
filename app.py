"""RunWeather Agent - Runner's Weather & Pace Weather Matching Dashboard (Session 1)

Run:  uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests
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

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = ("temperature_2m,relative_humidity_2m,apparent_temperature,"
               "precipitation_probability,wind_speed_10m,weather_code")


# ───────────────────────── Models ─────────────────────────
class Checkpoint(BaseModel):
    time: str = Field(..., description="ISO local time, e.g. 2026-11-08T06:30")
    location: str = Field(..., description="Place name, e.g. 'Seoul' (ignored if lat/lon given)")
    distance_km: float = Field(0, ge=0, description="Cumulative distance at this checkpoint")
    lat: Optional[float] = None
    lon: Optional[float] = None


class RunPlan(BaseModel):
    checkpoints: List[Checkpoint]


# ───────────────────────── Weather helpers ─────────────────────────
def geocode(name: str) -> dict:
    """Place name -> {lat, lon, name} using Open-Meteo geocoding (no API key)."""
    r = requests.get(GEOCODE_URL, params={"name": name, "count": 1, "language": "ko"}, timeout=10)
    r.raise_for_status()
    results = r.json().get("results")
    if not results:
        raise ValueError(f"장소를 찾을 수 없습니다: {name}")
    g = results[0]
    return {"lat": g["latitude"], "lon": g["longitude"], "name": g.get("name", name)}


def fetch_forecast(lat: float, lon: float, when: datetime) -> dict:
    """Hourly forecast nearest to `when` at (lat, lon).

    Draft using Open-Meteo. To switch to KMA (기상청 단기예보), replace this with a call to
    getVilageFcst (data.go.kr) after converting lat/lon to the KMA nx/ny grid.
    """
    day = when.strftime("%Y-%m-%d")
    r = requests.get(FORECAST_URL, params={
        "latitude": lat, "longitude": lon, "hourly": HOURLY_VARS,
        "timezone": "auto", "start_date": day, "end_date": day,
    }, timeout=10)
    r.raise_for_status()
    h = r.json()["hourly"]
    idx = min(when.hour, len(h["time"]) - 1)
    return {
        "temp_c": h["temperature_2m"][idx],
        "feels_like_c": h["apparent_temperature"][idx],
        "humidity": h["relative_humidity_2m"][idx],
        "wind_kmh": h["wind_speed_10m"][idx],
        "rain_prob": h["precipitation_probability"][idx],
        "weather_code": h["weather_code"][idx],
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
            if cp.lat is not None and cp.lon is not None:
                place = {"lat": cp.lat, "lon": cp.lon, "name": cp.location}
            else:
                place = geocode(cp.location)
            w = fetch_forecast(place["lat"], place["lon"], when)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except requests.RequestException as e:
            raise HTTPException(502, f"날씨 API 오류: {e}")
        out.append({
            "time": cp.time, "location": place["name"], "distance_km": cp.distance_km,
            "lat": place["lat"], "lon": place["lon"], "weather": w,
            "run_index": running_index(w),
            "outfit": outfit_guide(w["feels_like_c"], w["rain_prob"] or 0, w["wind_kmh"]),
        })
    avg = round(sum(c["run_index"]["score"] for c in out) / len(out))
    return {"count": len(out), "avg_run_index": avg, "checkpoints": out}


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ───────────────────────── Dashboard ─────────────────────────
PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RunWeather Agent</title>
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
.row{display:grid;grid-template-columns:1.2fr 1.4fr .8fr auto;gap:8px;align-items:end;margin-bottom:6px}
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
    <button class="sec" onclick="addRow()">+ 구간 추가</button>
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
function addRow(t='',l='서울',d=0){
  const el=document.createElement('div');el.className='row';
  el.innerHTML=`<div><label>시간</label><input type="datetime-local" value="${t}"></div>
   <div><label>장소</label><input value="${l}"></div>
   <div><label>거리(km)</label><input type="number" step="0.1" value="${d}"></div>
   <button class="sec" onclick="this.parentNode.remove()">✕</button>`;
  $('rows').appendChild(el);
}
const pad=n=>String(n).padStart(2,'0');
function ts(h){const d=new Date();d.setDate(d.getDate()+1);d.setHours(h,0,0,0);
  return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}T${pad(d.getHours())}:00`;}
addRow(ts(6),'서울',0);addRow(ts(7),'인천',10);

async function sync(){
  const checkpoints=[...document.querySelectorAll('.row')].map(r=>{
    const i=r.querySelectorAll('input');
    return {time:i[0].value,location:i[1].value,distance_km:parseFloat(i[2].value)||0};
  }).filter(c=>c.time&&c.location);
  $('status').textContent='동기화 중...';
  try{
    const res=await fetch('/api/run-weather',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({checkpoints})});
    if(!res.ok)throw new Error((await res.json()).detail||res.status);
    data=await res.json();render();
    $('status').textContent='동기화 완료 · '+new Date().toLocaleTimeString();
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
   <div style="margin-top:12px">👕 <b>복장 가이드</b><br><span class="muted">${c.outfit}</span></div>`;
}
function report(){
  if(!data){$('status').textContent='먼저 날씨를 동기화하세요.';return;}
  const worst=data.checkpoints.reduce((a,b)=>a.run_index.score<b.run_index.score?a:b);
  $('advisor').className='';
  $('advisor').textContent=`평균 런닝 지수: ${data.avg_run_index}/100\\n`+
   `가장 주의할 구간: ${worst.location} (${worst.time.replace('T',' ')}, ${worst.run_index.score}점)\\n`+
   `추천 복장: ${worst.outfit}\\n(LLM 기반 어드바이저는 다음 교시에 연동 예정)`;
  $('status').textContent='리포트 생성 완료';
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE
