import os, sys, io, time, threading
from collections import deque
from datetime import datetime

# ── Tcl/Tk path fix for Windows venv ──────────────────────
def _fix_tcl():
    candidates = []
    exe  = sys.executable
    base = os.path.dirname(os.path.dirname(exe))
    cfg  = os.path.join(base, 'pyvenv.cfg')
    if os.path.exists(cfg):
        with open(cfg) as f:
            for line in f:
                if line.lower().startswith('home'):
                    home = line.split('=', 1)[1].strip()
                    candidates += [home, os.path.dirname(home)]
                    break
    candidates += [
        os.path.dirname(exe),
        os.path.dirname(os.path.dirname(exe)),
        # full installer path (has tcl even when base install doesn't)
        r"C:\Users\baris\AppData\Local\Programs\Python\Python313",
    ]
    for python_dir in candidates:
        tcl = os.path.join(python_dir, 'tcl', 'tcl8.6')
        tk_ = os.path.join(python_dir, 'tcl', 'tk8.6')
        if os.path.exists(tcl):
            os.environ['TCL_LIBRARY'] = tcl
            os.environ['TK_LIBRARY']  = tk_
            return
_fix_tcl()

import tkinter as tk
from tkinter import ttk, messagebox

import cv2
from PIL import Image, ImageTk

# ── YOLO ──────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    MODEL_PATH = r"C:\Users\baris\Downloads\best (3).pt"
    model = YOLO(MODEL_PATH)
    MODEL_OK = True
    print("[OK] YOLO model loaded")
except Exception as e:
    model = None
    MODEL_OK = False
    print(f"[WARN] YOLO failed: {e}")

# ── MediaPipe ──────────────────────────────────────────────
HAND_LM = None
_mp     = None
try:
    import mediapipe as _mp
    from mediapipe.tasks import python as _mp_py
    from mediapipe.tasks.python import vision as _mp_vis
    HAND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
    if os.path.exists(HAND_PATH):
        _base = _mp_py.BaseOptions(model_asset_path=HAND_PATH)
        _opts = _mp_vis.HandLandmarkerOptions(
            base_options=_base, num_hands=4,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            running_mode=_mp_vis.RunningMode.IMAGE,
        )
        HAND_LM = _mp_vis.HandLandmarker.create_from_options(_opts)
        print("[OK] MediaPipe loaded")
    else:
        print("[INFO] hand_landmarker.task not found")
except Exception as e:
    print(f"[INFO] MediaPipe not available: {e}")

# ── Excel ──────────────────────────────────────────────────
try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    EXCEL_OK = True
except ImportError:
    EXCEL_OK = False

# ── Camera source ──────────────────────────────────────────
# Webcam          : CAMERA_SOURCE = 0
# IP Webcam       : CAMERA_SOURCE = "http://192.168.3.45:8080/video"
# DroidCam        : CAMERA_SOURCE = 1
CAMERA_SOURCE = "http://192.168.3.45:8080/video"

# ── Tools ─────────────────────────────────────────────────
TOOLS = [
    "army_navy","bulldog","castroviejo","clamp","forceps","frazier",
    "hemostat","iris","mayo_metz","needle","potts","richardson",
    "scalpel","towel_clip","weitlaner","yankauer","scissors",
]

TOOL_LABELS = {t: t.replace("_", " ").title() for t in TOOLS}

# ── State ──────────────────────────────────────────────────
_lock = threading.Lock()
S = {
    'surgery_active':  False,
    'surgery_start':   None,
    'tool_timers':     {t: 0 for t in TOOLS},
    'tool_states':     {t: 'on_table' for t in TOOLS},
    'tool_pick_count': {t: 0 for t in TOOLS},
    'prev_in_hand':    {t: False for t in TOOLS},
    'held_history':    {t: deque(maxlen=15) for t in TOOLS},
    'pre_tools':       {},
    'last_detected':   set(),
    'events':          [],
    'event_seq':       0,
    'fps':             0,
    '_fps_cnt':        0,
    '_fps_t':          time.time(),
    'hold_threshold':  0.20,
    'tool_scores':     {t: 0.0 for t in TOOLS},
}

# Ham kare (display thread okur — YOLO beklemeden)
_raw_frame  = None
_raw_lock   = threading.Lock()

# YOLO sonuçları (detect thread yazar, display thread çizer)
# Her eleman: (x1, y1, x2, y2, label, bgr_color, conf)
_det_boxes  = []
_det_lock   = threading.Lock()

# ── Helpers ────────────────────────────────────────────────
def _evt(msg, level="ok"):
    S['event_seq'] += 1
    ts = datetime.now().strftime("%H:%M:%S")
    S['events'].append({'id': S['event_seq'], 'ts': ts, 'level': level, 'msg': msg})
    if len(S['events']) > 120:
        del S['events'][:60]

def _score(bh, bt):
    hx1,hy1,hx2,hy2 = bh;  tx1,ty1,tx2,ty2 = bt
    ix1=max(hx1,tx1); iy1=max(hy1,ty1); ix2=min(hx2,tx2); iy2=min(hy2,ty2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    ta    = max((tx2-tx1)*(ty2-ty1), 1)
    ha    = max((hx2-hx1)*(hy2-hy1), 1)
    union = ta+ha-inter
    cont  = inter/ta
    tc_x  = (tx1+tx2)/2;  tc_y = (ty1+ty2)/2
    cin   = float(hx1<=tc_x<=hx2 and hy1<=tc_y<=hy2)
    iou   = inter/union if union else 0
    hc_x  = (hx1+hx2)/2;  hc_y = (hy1+hy2)/2
    hd    = max(((hx2-hx1)**2+(hy2-hy1)**2)**.5, 1)
    prox  = max(0, 1-((tc_x-hc_x)**2+(tc_y-hc_y)**2)**.5/(hd*1.5))
    return 0.40*cont + 0.30*cin + 0.20*iou + 0.10*prox

STATUS_BGR = {
    'on_table': (80,  200,  80),
    'in_hand':  (73,   81, 248),
    'missing':  (34,  153, 210),
}

# ── Thread 1: Sadece kamera okuma (mümkün olan en hızlı) ──
def _open_cap():
    cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_FFMPEG)
    if isinstance(CAMERA_SOURCE, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def _capture_loop():
    """Kamerayı sürekli okur; sadece en son kareyi _raw_frame'de tutar."""
    global _raw_frame
    cap        = _open_cap()
    fail_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            if fail_count > 30:
                print(f"[WARN] Reconnecting: {CAMERA_SOURCE}")
                cap.release()
                time.sleep(1.0)
                cap = _open_cap()
                fail_count = 0
            else:
                time.sleep(0.03)
            continue
        fail_count = 0
        # IP kamera buffer birikimini temizle: birkaç eski kareyi at
        for _ in range(2):
            ret2, f2 = cap.read()
            if ret2:
                frame = f2
        with _raw_lock:
            _raw_frame = frame
    cap.release()


# ── Thread 2: YOLO + MediaPipe (ağır iş, display'i bloklamaz) ─
def _detect_loop():
    """_raw_frame üzerinde YOLO+MP çalıştırır, sonuçları _det_boxes'a yazar."""
    global _det_boxes
    last_id = None
    while True:
        with _raw_lock:
            frame = _raw_frame
        if frame is None or id(frame) == last_id:
            time.sleep(0.01)
            continue
        last_id   = id(frame)
        work      = frame.copy()
        h_f, w_f  = work.shape[:2]

        # FPS sayacı
        with _lock:
            S['_fps_cnt'] += 1
            now = time.time()
            if now - S['_fps_t'] >= 1.0:
                S['fps']      = S['_fps_cnt']
                S['_fps_cnt'] = 0
                S['_fps_t']   = now

        # MediaPipe el algılama
        detected_hands = []
        if HAND_LM and _mp:
            try:
                rgb    = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
                mp_img = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
                hr     = HAND_LM.detect(mp_img)
                if hr.hand_landmarks:
                    for lms in hr.hand_landmarks:
                        xs = [l.x*w_f for l in lms]
                        ys = [l.y*h_f for l in lms]
                        p  = 25
                        hx1=max(0,int(min(xs))-p);   hy1=max(0,int(min(ys))-p)
                        hx2=min(w_f,int(max(xs))+p); hy2=min(h_f,int(max(ys))+p)
                        detected_hands.append([hx1,hy1,hx2,hy2])
            except Exception:
                pass

        # YOLO algılama
        detected_tools = {}
        detected_conf  = {}
        new_boxes      = []
        if MODEL_OK and model:
            try:
                conf_thresh = 0.22 if detected_hands else 0.35
                res = model(work, conf=conf_thresh, verbose=False, imgsz=640)[0]
                for box in res.boxes:
                    c   = box.xyxy[0].tolist()
                    lbl = model.names[int(box.cls[0])].lower()
                    cf  = float(box.conf[0])
                    if lbl in TOOLS:
                        if lbl not in detected_conf or cf > detected_conf[lbl]:
                            detected_tools[lbl] = c
                            detected_conf[lbl]  = cf
                    with _lock:
                        bc = STATUS_BGR.get(S['tool_states'].get(lbl,'on_table'), (88,166,255))
                    x1,y1,x2,y2 = int(c[0]),int(c[1]),int(c[2]),int(c[3])
                    new_boxes.append((x1, y1, x2, y2, lbl, bc, cf))
            except Exception:
                pass

        with _det_lock:
            _det_boxes = new_boxes

        # Durum güncelleme
        with _lock:
            S['last_detected'] = set(detected_tools.keys())
            s_active  = S['surgery_active']
            threshold = S['hold_threshold']

        live_scores = {}
        for tool in TOOLS:
            if tool in detected_tools and detected_hands:
                live_scores[tool] = round(
                    max(_score(h, detected_tools[tool]) for h in detected_hands), 3)
            else:
                live_scores[tool] = 0.0

        held_raw = {tool: live_scores[tool] >= threshold for tool in TOOLS} if s_active else {}

        with _lock:
            S['tool_scores'].update(live_scores)

        if s_active:
            with _lock:
                for tool in TOOLS:
                    in_view  = tool in detected_tools
                    raw_held = held_raw.get(tool, False)
                    S['held_history'][tool].append(raw_held)
                    is_held = sum(S['held_history'][tool]) > len(S['held_history'][tool]) / 2
                    prev    = S['prev_in_hand'][tool]
                    if in_view:
                        if is_held:
                            S['tool_timers'][tool] += 1
                            S['tool_states'][tool]  = 'in_hand'
                            if not prev:
                                S['tool_pick_count'][tool] += 1
                                _evt(f"{TOOL_LABELS[tool]} picked up "
                                     f"(#{S['tool_pick_count'][tool]})", "ok")
                        else:
                            if prev:
                                _evt(f"{TOOL_LABELS[tool]} released", "")
                            S['tool_states'][tool] = 'on_table'
                    else:
                        if S['tool_states'][tool] == 'in_hand':
                            S['tool_states'][tool] = 'missing'
                            _evt(f"WARNING: {TOOL_LABELS[tool]} disappeared!", "warn")
                    S['prev_in_hand'][tool] = is_held


# ── Colours ────────────────────────────────────────────────
BG       = '#0d1117'
BG_PANEL = '#161b22'
BG_CARD  = '#1c2128'
ACCENT   = '#58a6ff'
GREEN    = '#3fb950'
RED      = '#f85149'
YELLOW   = '#d29922'
CYAN     = '#39d0d8'
TEXT     = '#e6edf3'
TEXT_DIM = '#8b949e'
BORDER   = '#30363d'

# ── GUI ────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart Surgical Assistant — Near East University")
        self.configure(bg=BG)
        self.minsize(1100, 640)
        self.resizable(True, True)

        self._imgtk      = None
        self._last_evt   = 0

        self._build()
        self._tick_video()
        self._tick_state()

    # ── Layout ────────────────────────────────────────────
    def _build(self):
        # Video pane
        self.canvas = tk.Canvas(self, bg='#000000', highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right panel (fixed width)
        panel = tk.Frame(self, bg=BG_PANEL, width=310)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)

        # Title
        tk.Label(panel, text="Smart Surgical Assistant",
                 bg=BG_PANEL, fg=ACCENT, font=('Segoe UI', 11, 'bold')
                 ).pack(pady=(14,1))
        tk.Label(panel, text="Near East University",
                 bg=BG_PANEL, fg=TEXT_DIM, font=('Segoe UI', 8)
                 ).pack(pady=(0,10))
        self._sep(panel)

        # Surgery control
        self._section(panel, "SURGERY CONTROL")
        btn_row = tk.Frame(panel, bg=BG_PANEL)
        btn_row.pack(fill=tk.X, padx=12, pady=(0,4))
        self.btn_start = tk.Button(
            btn_row, text="▶  Start", bg=GREEN, fg='#0d1117',
            font=('Segoe UI', 9, 'bold'), relief=tk.FLAT,
            padx=10, pady=7, cursor='hand2', command=self._start)
        self.btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,4))
        self.btn_end = tk.Button(
            btn_row, text="■  End", bg=BG_CARD, fg=TEXT_DIM,
            font=('Segoe UI', 9, 'bold'), relief=tk.FLAT,
            padx=10, pady=7, cursor='hand2', command=self._end,
            state=tk.DISABLED)
        self.btn_end.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._timer_var = tk.StringVar(value="--:--:--")
        tk.Label(panel, textvariable=self._timer_var,
                 bg=BG_PANEL, fg=CYAN, font=('Consolas', 20, 'bold')
                 ).pack(pady=6)
        self._sep(panel)

        # Sensitivity
        self._section(panel, "DETECTION SENSITIVITY")
        sens = tk.Frame(panel, bg=BG_PANEL)
        sens.pack(fill=tk.X, padx=12, pady=(0,2))
        tk.Label(sens, text="Loose", bg=BG_PANEL, fg=TEXT_DIM,
                 font=('Segoe UI', 7)).pack(side=tk.LEFT)
        self._thresh = tk.DoubleVar(value=0.20)
        s = ttk.Scale(sens, from_=0.05, to=0.80, variable=self._thresh,
                      orient=tk.HORIZONTAL, command=self._on_thresh)
        s.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        tk.Label(sens, text="Strict", bg=BG_PANEL, fg=TEXT_DIM,
                 font=('Segoe UI', 7)).pack(side=tk.LEFT)
        self._thresh_lbl = tk.Label(panel, text="0.20",
                                     bg=BG_PANEL, fg=ACCENT, font=('Consolas', 9, 'bold'))
        self._thresh_lbl.pack()
        self._sep(panel)

        # Tool status
        self._section(panel, "TOOL STATUS")
        tool_frame = tk.Frame(panel, bg=BG_PANEL)
        tool_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,4))
        vsb = tk.Scrollbar(tool_frame, bg=BG_PANEL, troughcolor=BG_CARD, width=8)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tool_txt = tk.Text(
            tool_frame, bg=BG_CARD, fg=TEXT, font=('Consolas', 8),
            relief=tk.FLAT, state=tk.DISABLED, yscrollcommand=vsb.set,
            cursor='arrow', wrap=tk.NONE)
        self.tool_txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.config(command=self.tool_txt.yview)
        for tag, col in (('g', GREEN),('r', RED),('y', YELLOW),('d', TEXT_DIM)):
            self.tool_txt.tag_config(tag, foreground=col)
        self._sep(panel)

        # Event log
        self._section(panel, "EVENT LOG")
        log_frame = tk.Frame(panel, bg=BG_PANEL)
        log_frame.pack(fill=tk.BOTH, padx=8, pady=(0,4))
        lsb = tk.Scrollbar(log_frame, bg=BG_PANEL, troughcolor=BG_CARD, width=8)
        lsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.evt_txt = tk.Text(
            log_frame, bg=BG_CARD, fg=TEXT_DIM, font=('Consolas', 7),
            relief=tk.FLAT, state=tk.DISABLED, yscrollcommand=lsb.set,
            height=8, cursor='arrow', wrap=tk.WORD)
        self.evt_txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lsb.config(command=self.evt_txt.yview)
        for tag, col in (('ok', GREEN),('warn', YELLOW),('err', RED),('ai', CYAN),('ts', TEXT_DIM)):
            self.evt_txt.tag_config(tag, foreground=col)

        # Export
        tk.Button(
            panel, text="⬇  Export Excel Report (.xlsx)",
            bg=BG_CARD, fg=ACCENT, font=('Segoe UI', 8, 'bold'),
            relief=tk.FLAT, pady=8, cursor='hand2', command=self._export
        ).pack(fill=tk.X, padx=12, pady=(4,10))

    def _sep(self, p):
        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X, padx=8, pady=3)

    def _section(self, p, text):
        tk.Label(p, text=text, bg=BG_PANEL, fg=TEXT_DIM,
                 font=('Segoe UI', 7, 'bold')).pack(anchor='w', padx=12, pady=(6,3))

    # ── Video tick (fast — YOLO'yu beklemiyor) ────────────
    def _tick_video(self):
        with _raw_lock:
            frame = _raw_frame
        if frame is not None:
            display = frame.copy()
            fh, fw  = display.shape[:2]

            # YOLO kutularını çiz (en son sonuç, arka planda güncelleniyor)
            with _det_lock:
                boxes = list(_det_boxes)
            for (x1, y1, x2, y2, lbl, bc, cf) in boxes:
                cv2.rectangle(display, (x1,y1),(x2,y2), bc, 2)
                cv2.putText(display, f"{lbl} {cf:.2f}", (x1, max(y1-6,12)),
                            0, 0.42, (230,237,243), 1)

            # HUD
            with _lock:
                fps_v = S['fps']
                s_act = S['surgery_active']
            cv2.putText(display, f"FPS:{fps_v}", (fw-80, 22), 0, 0.55, (57,208,216), 2)
            s_txt = "SURGERY ACTIVE" if s_act else "STANDBY"
            s_col = (80,200,80) if s_act else (120,120,120)
            cv2.putText(display, s_txt, (10, fh-12), 0, 0.48, s_col, 1)
            SZ=22; T=2
            for px,py,dx,dy in [(0,0,1,1),(fw,0,-1,1),(0,fh,1,-1),(fw,fh,-1,-1)]:
                cv2.line(display,(px,py),(px+dx*SZ,py),(88,166,255),T)
                cv2.line(display,(px,py),(px,py+dy*SZ),(88,166,255),T)

            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw > 10 and ch > 10:
                scale  = min(cw/fw, ch/fh)
                nw, nh = int(fw*scale), int(fh*scale)
                resized = cv2.resize(display, (nw, nh), interpolation=cv2.INTER_LINEAR)
                rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pil     = Image.fromarray(rgb)
                self._imgtk = ImageTk.PhotoImage(pil)
                self.canvas.delete('all')
                x = (cw - nw) // 2
                y = (ch - nh) // 2
                self.canvas.create_image(x, y, anchor=tk.NW, image=self._imgtk)
        self.after(16, self._tick_video)   # ~60 FPS hedef

    # ── State tick (slow) ─────────────────────────────────
    def _tick_state(self):
        with _lock:
            s_active = S['surgery_active']
            elapsed  = int(time.time() - S['surgery_start']) \
                       if s_active and S['surgery_start'] else 0
            states   = dict(S['tool_states'])
            timers   = dict(S['tool_timers'])
            picks    = dict(S['tool_pick_count'])
            events   = list(S['events'])

        # Timer label
        if s_active:
            h = elapsed//3600; m = (elapsed%3600)//60; s = elapsed%60
            self._timer_var.set(f"{h:02d}:{m:02d}:{s:02d}")

        # Buttons
        if s_active:
            self.btn_start.config(state=tk.DISABLED, bg=BG_CARD, fg=TEXT_DIM)
            self.btn_end.config(state=tk.NORMAL, bg=RED, fg='white')
        else:
            self.btn_start.config(state=tk.NORMAL, bg=GREEN, fg='#0d1117')
            self.btn_end.config(state=tk.DISABLED, bg=BG_CARD, fg=TEXT_DIM)

        # Tool list
        self.tool_txt.config(state=tk.NORMAL)
        self.tool_txt.delete('1.0', tk.END)
        for tool in TOOLS:
            st  = states.get(tool, 'on_table')
            sec = timers.get(tool, 0) // 10
            pk  = picks.get(tool, 0)
            tag = 'g' if st == 'on_table' else ('r' if st == 'in_hand' else 'y')
            badge = {'on_table': 'TABLE  ', 'in_hand': 'IN HAND', 'missing': 'MISSING'}.get(st, 'TABLE  ')
            line = f"  {TOOL_LABELS[tool]:<17} [{badge}] {sec:>4}s  ×{pk}\n"
            self.tool_txt.insert(tk.END, line, tag)
        self.tool_txt.config(state=tk.DISABLED)

        # Event log
        new_evts = [e for e in events if e['id'] > self._last_evt]
        if new_evts:
            self.evt_txt.config(state=tk.NORMAL)
            for e in new_evts:
                self._last_evt = max(self._last_evt, e['id'])
                lvl = e['level'] if e['level'] in ('ok','warn','err','ai') else 'ts'
                self.evt_txt.insert(tk.END, f"[{e['ts']}] ", 'ts')
                self.evt_txt.insert(tk.END, e['msg'] + "\n", lvl)
            self.evt_txt.see(tk.END)
            self.evt_txt.config(state=tk.DISABLED)

        self.after(400, self._tick_state)

    # ── Threshold ─────────────────────────────────────────
    def _on_thresh(self, val):
        v = round(float(val), 2)
        self._thresh_lbl.config(text=f"{v:.2f}")
        with _lock:
            S['hold_threshold'] = v

    # ── Surgery control ───────────────────────────────────
    def _start(self):
        with _lock:
            if S['surgery_active']:
                return
            S['surgery_active'] = True
            S['surgery_start']  = time.time()
            for t in TOOLS:
                S['tool_timers'][t]     = 0
                S['tool_states'][t]     = 'on_table'
                S['tool_pick_count'][t] = 0
                S['prev_in_hand'][t]    = False
                S['held_history'][t].clear()
            S['pre_tools'] = {t: (t in S['last_detected']) for t in TOOLS}
            present = [t for t in TOOLS if S['pre_tools'].get(t)]
            _evt("=== SURGERY STARTED ===", "ok")
            _evt(f"Initial tools: {len(present)} detected", "ok")

    def _end(self):
        with _lock:
            if not S['surgery_active']:
                return
            S['surgery_active'] = False
            post    = {t: (t in S['last_detected']) for t in TOOLS}
            missing = [t for t in TOOLS if S['pre_tools'].get(t) and not post[t]]
            elapsed = int(time.time() - S['surgery_start']) if S['surgery_start'] else 0
            _evt("=== SURGERY ENDED ===", "ok")
            if missing:
                _evt(f"MISSING TOOLS: {', '.join(missing)}", "err")
            else:
                _evt("All tools verified. None missing.", "ok")
            top3 = sorted(TOOLS, key=lambda t: S['tool_timers'][t], reverse=True)[:3]
            top3 = [f"{t}:{S['tool_timers'][t]//10}s" for t in top3 if S['tool_timers'][t] > 0]
            if top3:
                _evt("Most used: " + ", ".join(top3), "ai")
        self._timer_var.set("--:--:--")
        if missing:
            messagebox.showwarning("Missing Tools",
                                   "Missing tools detected:\n" + "\n".join(
                                       TOOL_LABELS.get(t, t) for t in missing))

    # ── Excel export ──────────────────────────────────────
    def _export(self):
        if not EXCEL_OK:
            messagebox.showerror("Error", "openpyxl not installed.\npip install openpyxl")
            return
        with _lock:
            timers   = dict(S['tool_timers'])
            states   = dict(S['tool_states'])
            picks    = dict(S['tool_pick_count'])
            events   = list(S['events'])
            s_active = S['surgery_active']
            s_start  = S['surgery_start']
            elapsed  = int(time.time() - s_start) if s_active and s_start else 0

        wb      = openpyxl.Workbook()
        ts_str  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        hdr_fill   = PatternFill("solid", fgColor="1C2128")
        title_fill = PatternFill("solid", fgColor="161B22")
        dark_fill  = PatternFill("solid", fgColor="0D1117")
        mid_fill   = PatternFill("solid", fgColor="161B22")

        title_font  = Font(name="Calibri", bold=True, color="E6EDF3", size=13)
        hdr_font    = Font(name="Calibri", bold=True, color="58A6FF", size=10)
        normal_font = Font(name="Calibri", color="E6EDF3", size=10)
        dim_font    = Font(name="Calibri", color="8B949E", size=9)
        green_font  = Font(name="Calibri", bold=True, color="3FB950", size=10)
        red_font    = Font(name="Calibri", bold=True, color="F85149", size=10)
        yellow_font = Font(name="Calibri", bold=True, color="D29922", size=10)

        green_fill  = PatternFill("solid", fgColor="0D3320")
        red_fill    = PatternFill("solid", fgColor="3D1210")
        yellow_fill = PatternFill("solid", fgColor="3D2C00")

        thin = Border(
            left=Side(style="thin", color="30363D"),
            right=Side(style="thin", color="30363D"),
            top=Side(style="thin", color="30363D"),
            bottom=Side(style="thin", color="30363D"),
        )
        center = Alignment(horizontal="center", vertical="center")
        left   = Alignment(horizontal="left",   vertical="center")

        ws1 = wb.active
        ws1.title = "Tool Summary"
        ws1.merge_cells("A1:F1")
        ws1["A1"] = "Smart Surgical Assistant — Session Report"
        ws1["A1"].font = title_font; ws1["A1"].fill = title_fill; ws1["A1"].alignment = center
        ws1.row_dimensions[1].height = 28
        ws1.merge_cells("A2:F2")
        ws1["A2"] = f"Generated: {ts_str}   |   Surgery active: {'Yes' if s_active else 'No'}"
        ws1["A2"].font = dim_font; ws1["A2"].fill = title_fill; ws1["A2"].alignment = left

        for col, h in enumerate(["Tool Name","Use Time (s)","Pick Count","Status","Notes"], 1):
            cell = ws1.cell(row=4, column=col, value=h)
            cell.font = hdr_font; cell.fill = hdr_fill; cell.border = thin; cell.alignment = center
        ws1.row_dimensions[4].height = 20

        status_map = {
            'on_table': ("ON TABLE", green_font,  green_fill),
            'in_hand':  ("IN HAND",  red_font,    red_fill),
            'missing':  ("MISSING",  yellow_font, yellow_fill),
        }
        for i, tool in enumerate(TOOLS):
            r   = i + 5
            sec = timers[tool] // 10
            bg  = dark_fill if i % 2 == 0 else mid_fill
            stxt, sfont, sfill = status_map.get(states[tool], status_map['on_table'])
            for col, val in enumerate([TOOL_LABELS[tool], sec, picks[tool], stxt, ""], 1):
                cell = ws1.cell(row=r, column=col, value=val)
                cell.border = thin
                cell.alignment = left if col == 1 else center
                if col == 4:
                    cell.font = sfont; cell.fill = sfill
                else:
                    cell.font = normal_font; cell.fill = bg
        for col, w in enumerate([22,14,12,14,20], 1):
            ws1.column_dimensions[get_column_letter(col)].width = w

        ws2 = wb.create_sheet("Event Log")
        ws2.merge_cells("A1:C1")
        ws2["A1"] = "Event Log"
        ws2["A1"].font = title_font; ws2["A1"].fill = title_fill; ws2["A1"].alignment = center
        ws2.row_dimensions[1].height = 26
        for col, h in enumerate(["Timestamp","Level","Message"], 1):
            cell = ws2.cell(row=2, column=col, value=h)
            cell.font = hdr_font; cell.fill = hdr_fill; cell.border = thin; cell.alignment = center

        lvl_colors = {
            "ok":  ("3FB950","0D3320"), "warn": ("D29922","3D2C00"),
            "err": ("F85149","3D1210"), "ai":   ("39D0D8","0D2A2C"),
            "":    ("8B949E","161B22"),
        }
        for i, ev in enumerate(events):
            r = i + 3
            fg, bg_hex = lvl_colors.get(ev['level'], ("8B949E","161B22"))
            ef = Font(name="Calibri", color=fg, size=9)
            efill = PatternFill("solid", fgColor=bg_hex)
            for col, val in enumerate([ev['ts'], ev['level'].upper(), ev['msg']], 1):
                cell = ws2.cell(row=r, column=col, value=val)
                cell.font = ef; cell.fill = efill; cell.border = thin
                cell.alignment = Alignment(horizontal="left", vertical="center",
                                           wrap_text=(col == 3))
        ws2.column_dimensions["A"].width = 14
        ws2.column_dimensions["B"].width = 10
        ws2.column_dimensions["C"].width = 70

        fname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"surgical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        wb.save(fname)
        messagebox.showinfo("Export Complete", f"Saved:\n{fname}")
        os.startfile(fname)


# ── Entry point ────────────────────────────────────────────
if __name__ == '__main__':
    threading.Thread(target=_capture_loop, daemon=True).start()   # kamera okuma
    threading.Thread(target=_detect_loop,  daemon=True).start()   # YOLO işleme
    print("\n" + "="*52)
    print("  Smart Surgical Assistant — Tkinter GUI")
    print("="*52 + "\n")
    App().mainloop()
