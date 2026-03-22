"""
InVesalius TEP Visualization System - FastAPI Backend
"""
from __future__ import annotations
import asyncio, json, logging, struct, time, uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.signal import butter, filtfilt

DB_PATH = Path(__file__).resolve().parent / "invesalius.db"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.pt"
N_CHANNELS = 19
SRATE = 1000
N_SAMPLES = 701
TRIAL_INTERVAL = 1.5
ARTIFACT_RATE = 0.18
CHANNEL_NAMES = [
    "Fp1","Fp2","F3","F4","C3","C4","P3","P4",
    "O1","O2","F7","F8","T3","T4","T5","T6","Fz","Cz","Pz",
]

logger = logging.getLogger("invesalius")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

db: aiosqlite.Connection | None = None
ml_model: Any = None
ml_model_loaded: bool = False
simulator_task: asyncio.Task | None = None

# ── Pydantic ──
class SessionStart(BaseModel):
    subject_id: str | None = None

class SessionInfo(BaseModel):
    session_id: str
    timestamp: float

class SessionStop(BaseModel):
    n_trials: int
    duration: float

class TrialMeta(BaseModel):
    id: int
    trial_num: int
    timestamp: float
    is_artifact: int | None
    peak_n45_latency: float | None
    peak_n45_amplitude: float | None
    peak_n100_amplitude: float | None
    created_at: str

class MetricsResponse(BaseModel):
    n_trials: int
    n_clean: int
    n_artifact: int
    artifact_rate: float
    mean_n45_latency: float | None
    mean_n45_amplitude: float | None
    mean_n100_amplitude: float | None
    model_loaded: bool
    model_confidence_mean: float | None

class HealthResponse(BaseModel):
    status: str
    n_sessions: int
    model_loaded: bool
    uptime: float

# ── Database ──
SESSIONS_DDL = """CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, subject_id TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    n_trials INTEGER DEFAULT 0, status TEXT DEFAULT 'recording'
)"""
TRIALS_DDL = """CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL, trial_num INTEGER NOT NULL,
    timestamp REAL NOT NULL, data BLOB NOT NULL,
    is_artifact INTEGER,
    peak_n45_latency REAL, peak_n45_amplitude REAL, peak_n100_amplitude REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
)"""

async def init_db() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(str(DB_PATH))
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute(SESSIONS_DDL)
    await conn.execute(TRIALS_DDL)
    await conn.commit()
    logger.info("DB ready at %s", DB_PATH)
    return conn

def ndarray_to_blob(arr: np.ndarray) -> bytes:
    dt = str(arr.dtype).encode()
    sh = struct.pack(f"{len(arr.shape)}I", *arr.shape)
    hdr = struct.pack("BB", len(dt), len(arr.shape))
    return hdr + dt + sh + arr.tobytes()

def blob_to_ndarray(blob: bytes) -> np.ndarray:
    dtl, ndim = struct.unpack("BB", blob[:2])
    off = 2
    dts = blob[off:off+dtl].decode(); off += dtl
    shape = struct.unpack(f"{ndim}I", blob[off:off+ndim*4]); off += ndim*4
    return np.frombuffer(blob[off:], dtype=np.dtype(dts)).reshape(shape).copy()

# ── WebSocket Manager ──
class ConnectionManager:
    def __init__(self):
        self._conns: dict[str, list[WebSocket]] = {}

    async def connect(self, sid: str, ws: WebSocket):
        await ws.accept()
        self._conns.setdefault(sid, []).append(ws)
        logger.info("WS+ session=%s clients=%d", sid, len(self._conns[sid]))

    def disconnect(self, sid: str, ws: WebSocket):
        c = self._conns.get(sid, [])
        if ws in c:
            c.remove(ws)
        if not c:
            self._conns.pop(sid, None)

    async def broadcast(self, sid: str, msg: dict):
        payload = json.dumps(msg, default=_json_ser)
        dead = []
        for ws in self._conns.get(sid, []):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(sid, ws)

    @property
    def session_count(self):
        return len(self._conns)

def _json_ser(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.int32, np.int64)):
        return int(o)
    raise TypeError

manager = ConnectionManager()

# ── EEG Generator ──
def _bp(lo, hi, fs, order=4):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band")
    return b, a

def generate_tep_trial(n_ch=N_CHANNELS, n_s=N_SAMPLES, sr=SRATE, artifact=False):
    rng = np.random.default_rng()
    t = np.arange(n_s) / sr
    si = 100
    data = np.zeros((n_ch, n_s), dtype=np.float64)
    ba, aa = _bp(8.0, 13.0, sr)
    for ch in range(n_ch):
        wh = rng.standard_normal(n_s + 200)
        alpha = filtfilt(ba, aa, wh)[100:100+n_s]
        pk = np.cumsum(rng.standard_normal(n_s))
        pk -= pk.mean()
        pk /= max(pk.std(), 1e-9)
        data[ch] = alpha * rng.uniform(3, 8) + pk * 0.5 + rng.standard_normal(n_s) * 1.5

    def bump(ctr_ms, w_ms, amp):
        t_post = t - (si / sr)
        return amp * np.exp(-0.5 * ((t_post - ctr_ms/1000) / (w_ms/1000))**2)

    comps = [(15,4,-8,3),(30,5,12,4),(45,8,-25,8),(60,6,10,4),(100,15,-30,10),(180,20,15,6)]
    for c, w, ma, sa in comps:
        for ch in range(n_ch):
            data[ch] += bump(c + rng.normal(0, w*0.15), w + rng.uniform(-1,1), rng.normal(ma, sa))

    pw = 3
    for ch in range(n_ch):
        amp = rng.uniform(100, 300)
        lo = max(si-pw, 0)
        hi = min(si+pw+1, n_s)
        tl = np.arange(lo, hi) - si
        data[ch, lo:hi] += amp * tl * np.exp(-0.5 * (tl/1.0)**2)

    if artifact:
        ae = min(si + int(sr * 0.012), n_s)
        for ch in range(n_ch):
            burst = rng.standard_normal(ae - si) * rng.uniform(80, 250)
            env = np.linspace(1, 0, ae - si)**0.5
            data[ch, si:ae] += burst * env

    return data.astype(np.float32)

def compute_tep_metrics(data):
    si = 100
    m = data.mean(axis=0)
    n45w = m[si+30:si+60]
    n45i = int(np.argmin(n45w))
    n100w = m[si+80:si+130]
    n100i = int(np.argmin(n100w))
    return {
        "peak_n45_latency": round(float(30 + n45i), 2),
        "peak_n45_amplitude": round(float(n45w[n45i]), 4),
        "peak_n100_amplitude": round(float(n100w[n100i]), 4),
    }

# ── ML ──
def _load_model():
    try:
        import torch
        if MODEL_PATH.exists():
            m = torch.jit.load(str(MODEL_PATH), map_location="cpu")
            m.eval()
            logger.info("Model loaded: %s", MODEL_PATH)
            return m
        logger.warning("No model at %s — heuristic mode", MODEL_PATH)
    except Exception as e:
        logger.warning("Model load failed: %s — heuristic mode", e)
    return None

def classify_trial_sync(data: np.ndarray) -> dict:
    if ml_model is None:
        si = 100
        mx = float(np.max(np.abs(data[:, si:si+15])))
        ia = mx > 120
        conf = min(mx/250, 1.0) if ia else max(1 - mx/150, 0.0)
        return {"is_artifact": ia, "confidence": round(float(conf), 4)}
    import torch
    with torch.no_grad():
        tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        p = torch.sigmoid(ml_model(tensor)).item()
    return {"is_artifact": p > 0.5, "confidence": round(float(p if p > 0.5 else 1-p), 4)}

async def classify_trial(data: np.ndarray) -> dict:
    return await asyncio.get_running_loop().run_in_executor(None, classify_trial_sync, data)

# ── Simulator ──
class EEGDeviceSimulator:
    def __init__(self):
        self._running = False
        self._session_id: str | None = None
        self._trial_num = 0
        self._t0 = 0.0
        self._clean: list[np.ndarray] = []

    @property
    def session_id(self):
        return self._session_id

    @property
    def running(self):
        return self._running

    async def start(self, sid: str):
        self._session_id = sid
        self._trial_num = 0
        self._t0 = time.time()
        self._clean = []
        self._running = True
        logger.info("Sim start session=%s", sid)

    async def stop(self):
        self._running = False
        dur = time.time() - self._t0
        n = self._trial_num
        logger.info("Sim stop: %d trials %.1fs", n, dur)
        return {"n_trials": n, "duration": round(dur, 2)}

    async def run_forever(self):
        while True:
            if not self._running or not self._session_id:
                await asyncio.sleep(0.25)
                continue

            sid = self._session_id
            self._trial_num += 1
            tnum = self._trial_num
            ts = time.time()

            has_art = np.random.random() < ARTIFACT_RATE
            data = generate_tep_trial(artifact=has_art)
            result = await classify_trial(data)
            ia = int(result["is_artifact"])
            met = compute_tep_metrics(data)

            blob = ndarray_to_blob(data)
            async with db.cursor() as cur:
                await cur.execute(
                    "INSERT INTO trials (session_id,trial_num,timestamp,data,is_artifact,"
                    "peak_n45_latency,peak_n45_amplitude,peak_n100_amplitude) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (sid, tnum, ts, blob, ia, met["peak_n45_latency"],
                     met["peak_n45_amplitude"], met["peak_n100_amplitude"]))
                await cur.execute(
                    "UPDATE sessions SET n_trials=? WHERE id=?", (tnum, sid))
            await db.commit()

            if not ia:
                self._clean.append(data)

            await manager.broadcast(sid, {
                "type": "trial", "trial_num": tnum, "data": data,
                "is_artifact": ia, "confidence": result["confidence"],
                "metrics": met, "timestamp": ts,
            })

            logger.info("Trial %d art=%d conf=%.3f N45=%.1fms/%.1fuV N100=%.1fuV",
                         tnum, ia, result["confidence"],
                         met["peak_n45_latency"], met["peak_n45_amplitude"],
                         met["peak_n100_amplitude"])

            if tnum % 10 == 0 and self._clean:
                ev = np.mean(self._clean, axis=0)
                em = compute_tep_metrics(ev)
                await manager.broadcast(sid, {
                    "type": "evoked", "data": ev,
                    "n_clean": len(self._clean),
                    "metrics": em, "timestamp": time.time(),
                })

            if tnum % 20 == 0:
                nc = len(self._clean)
                na = tnum - nc
                am = await _agg_metrics(sid, tnum, nc, na)
                await manager.broadcast(sid, {
                    "type": "metrics", **am,
                    "timestamp": time.time(),
                })

            await asyncio.sleep(TRIAL_INTERVAL)

simulator = EEGDeviceSimulator()

async def _agg_metrics(sid, total, nc, na):
    async with db.cursor() as cur:
        await cur.execute(
            "SELECT AVG(peak_n45_latency) a, AVG(peak_n45_amplitude) b, "
            "AVG(peak_n100_amplitude) c FROM trials "
            "WHERE session_id=? AND is_artifact=0", (sid,))
        r = await cur.fetchone()
    return {
        "n_trials": total, "n_clean": nc, "n_artifact": na,
        "artifact_rate": round(na/max(total,1), 4),
        "mean_n45_latency": round(r["a"],2) if r["a"] else None,
        "mean_n45_amplitude": round(r["b"],4) if r["b"] else None,
        "mean_n100_amplitude": round(r["c"],4) if r["c"] else None,
        "model_loaded": ml_model_loaded, "model_confidence_mean": None,
    }

# ── Lifespan ──
_boot = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, ml_model, ml_model_loaded, simulator_task
    db = await init_db()
    ml_model = _load_model()
    ml_model_loaded = ml_model is not None
    simulator_task = asyncio.create_task(simulator.run_forever())
    logger.info("Backend up (model=%s)", ml_model_loaded)
    yield
    simulator._running = False
    if simulator_task:
        simulator_task.cancel()
        try:
            await simulator_task
        except asyncio.CancelledError:
            pass
    if db:
        await db.close()
    logger.info("Backend down")

# ── App ──
app = FastAPI(title="InVesalius TEP", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ── REST endpoints ──
@app.get("/health", response_model=HealthResponse)
async def health():
    async with db.cursor() as cur:
        await cur.execute("SELECT COUNT(*) cnt FROM sessions")
        r = await cur.fetchone()
    return HealthResponse(status="ok", n_sessions=r["cnt"],
                          model_loaded=ml_model_loaded,
                          uptime=round(time.time()-_boot, 2))

@app.post("/sessions/start", response_model=SessionInfo)
async def start_session(body: SessionStart | None = None):
    sid = str(uuid.uuid4())
    ts = time.time()
    subj = body.subject_id if body else None
    async with db.cursor() as cur:
        await cur.execute("INSERT INTO sessions (id,subject_id) VALUES (?,?)", (sid, subj))
    await db.commit()
    await simulator.start(sid)
    logger.info("Session %s started subject=%s", sid, subj)
    return SessionInfo(session_id=sid, timestamp=ts)

@app.post("/sessions/{session_id}/stop", response_model=SessionStop)
async def stop_session(session_id: str):
    async with db.cursor() as cur:
        await cur.execute("SELECT id FROM sessions WHERE id=?", (session_id,))
        if not await cur.fetchone():
            raise HTTPException(404, "Session not found")
        await cur.execute("UPDATE sessions SET status='stopped' WHERE id=?", (session_id,))
    await db.commit()
    return SessionStop(**(await simulator.stop()))

@app.get("/sessions/{session_id}/trials", response_model=list[TrialMeta])
async def list_trials(session_id: str, limit: int = Query(100, ge=1, le=1000),
                      offset: int = Query(0, ge=0)):
    async with db.cursor() as cur:
        await cur.execute("SELECT id FROM sessions WHERE id=?", (session_id,))
        if not await cur.fetchone():
            raise HTTPException(404, "Session not found")
        await cur.execute(
            "SELECT id,trial_num,timestamp,is_artifact,peak_n45_latency,"
            "peak_n45_amplitude,peak_n100_amplitude,created_at "
            "FROM trials WHERE session_id=? ORDER BY trial_num DESC LIMIT ? OFFSET ?",
            (session_id, limit, offset))
        rows = await cur.fetchall()
    return [TrialMeta(id=r["id"], trial_num=r["trial_num"], timestamp=r["timestamp"],
                      is_artifact=r["is_artifact"], peak_n45_latency=r["peak_n45_latency"],
                      peak_n45_amplitude=r["peak_n45_amplitude"],
                      peak_n100_amplitude=r["peak_n100_amplitude"],
                      created_at=r["created_at"]) for r in rows]

@app.get("/sessions/{session_id}/evoked")
async def get_evoked(session_id: str):
    async with db.cursor() as cur:
        await cur.execute("SELECT id FROM sessions WHERE id=?", (session_id,))
        if not await cur.fetchone():
            raise HTTPException(404, "Session not found")
        await cur.execute(
            "SELECT data FROM trials WHERE session_id=? AND is_artifact=0 "
            "ORDER BY trial_num", (session_id,))
        rows = await cur.fetchall()
    if not rows:
        raise HTTPException(404, "No clean trials yet")
    arrays = [blob_to_ndarray(r["data"]) for r in rows]
    ev = np.mean(arrays, axis=0)
    met = compute_tep_metrics(ev)
    return {"data": ev.tolist(), "n_clean": len(arrays),
            "metrics": met, "channels": CHANNEL_NAMES}

@app.get("/sessions/{session_id}/metrics", response_model=MetricsResponse)
async def get_metrics(session_id: str):
    async with db.cursor() as cur:
        await cur.execute("SELECT id FROM sessions WHERE id=?", (session_id,))
        if not await cur.fetchone():
            raise HTTPException(404, "Session not found")
        await cur.execute(
            "SELECT COUNT(*) t FROM trials WHERE session_id=?", (session_id,))
        total = (await cur.fetchone())["t"]
        await cur.execute(
            "SELECT COUNT(*) c FROM trials WHERE session_id=? AND is_artifact=0",
            (session_id,))
        nc = (await cur.fetchone())["c"]
        await cur.execute(
            "SELECT AVG(peak_n45_latency) a, AVG(peak_n45_amplitude) b, "
            "AVG(peak_n100_amplitude) c FROM trials "
            "WHERE session_id=? AND is_artifact=0", (session_id,))
        r = await cur.fetchone()
    na = total - nc
    return MetricsResponse(
        n_trials=total, n_clean=nc, n_artifact=na,
        artifact_rate=round(na/max(total,1), 4),
        mean_n45_latency=round(r["a"],2) if r["a"] else None,
        mean_n45_amplitude=round(r["b"],4) if r["b"] else None,
        mean_n100_amplitude=round(r["c"],4) if r["c"] else None,
        model_loaded=ml_model_loaded, model_confidence_mean=None)

@app.post("/sessions/{session_id}/analyze")
async def analyze_session(session_id: str):
    async with db.cursor() as cur:
        await cur.execute("SELECT id,n_trials FROM sessions WHERE id=?", (session_id,))
        sess = await cur.fetchone()
        if not sess:
            raise HTTPException(404, "Session not found")
        await cur.execute(
            "SELECT data,is_artifact FROM trials WHERE session_id=? ORDER BY trial_num",
            (session_id,))
        rows = await cur.fetchall()
    if not rows:
        raise HTTPException(400, "No trials recorded")

    clean = [blob_to_ndarray(r["data"]) for r in rows if r["is_artifact"] == 0]
    all_arr = [blob_to_ndarray(r["data"]) for r in rows]
    if not clean:
        raise HTTPException(400, "No clean trials available")

    ga = np.mean(clean, axis=0)
    gm = compute_tep_metrics(ga)

    qs = max(len(clean) // 4, 1)
    studies = []
    for i in range(4):
        s = i * qs
        e = s + qs if i < 3 else len(clean)
        sub = clean[s:e]
        if not sub:
            continue
        avg = np.mean(sub, axis=0)
        studies.append({
            "study": i+1, "n_trials": len(sub),
            "evoked": avg.tolist(), "metrics": compute_tep_metrics(avg),
        })

    if len(clean) > 1:
        sig = np.mean(clean, axis=0)
        nv = np.mean([np.var(t - sig) for t in clean])
        sv = np.var(sig)
        snr = float(10 * np.log10(sv / max(nv, 1e-12)))
    else:
        snr = 0.0

    return {
        "session_id": session_id, "n_total": len(all_arr),
        "n_clean": len(clean), "n_artifact": len(all_arr)-len(clean),
        "grand_average": ga.tolist(), "grand_metrics": gm,
        "studies": studies, "snr_db": round(snr, 2),
        "channels": CHANNEL_NAMES,
    }

# ── WebSocket ──
@app.websocket("/ws/{session_id}")
async def ws_endpoint(ws: WebSocket, session_id: str):
    await manager.connect(session_id, ws)
    try:
        async with db.cursor() as cur:
            await cur.execute(
                "SELECT trial_num,timestamp,data,is_artifact,"
                "peak_n45_latency,peak_n45_amplitude,peak_n100_amplitude "
                "FROM trials WHERE session_id=? ORDER BY trial_num DESC LIMIT 10",
                (session_id,))
            rows = await cur.fetchall()

        init = []
        for r in reversed(list(rows)):
            arr = blob_to_ndarray(r["data"])
            init.append({
                "type": "trial", "trial_num": r["trial_num"],
                "data": arr.tolist(), "is_artifact": r["is_artifact"],
                "metrics": {
                    "peak_n45_latency": r["peak_n45_latency"],
                    "peak_n45_amplitude": r["peak_n45_amplitude"],
                    "peak_n100_amplitude": r["peak_n100_amplitude"],
                },
                "timestamp": r["timestamp"],
            })
        await ws.send_text(json.dumps({
            "type": "initial_state", "trials": init,
            "session_id": session_id,
        }, default=_json_ser))

        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
                if msg.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong", "ts": time.time()}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WS err session=%s: %s", session_id, e)
    finally:
        manager.disconnect(session_id, ws)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
