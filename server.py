import os, io, time, threading, traceback
import csv
import cv2
from collections import deque
from datetime import datetime
from flask import Flask, Response, jsonify, send_from_directory, send_file, request

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
    print(f"[WARN] YOLO failed to load: {e}")

# ── MediaPipe (optional) ───────────────────────────────────
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
        print("[OK] MediaPipe Hand Landmarker loaded")
    else:
        print("[INFO] hand_landmarker.task not found; using IoU-based hand detection")
except Exception as e:
    print(f"[INFO] MediaPipe not available: {e}")

# ── Excel support (optional) ───────────────────────────────
try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    EXCEL_OK = True
except ImportError:
    EXCEL_OK = False

# ── Camera source ──────────────────────────────────────────
# Bilgisayar webcam : CAMERA_SOURCE = 0
# IP Webcam Android : CAMERA_SOURCE = "http://192.168.3.45:8080/video"
# DroidCam sanal    : CAMERA_SOURCE = 1
CAMERA_SOURCE = "http://192.168.3.45:8080/video"

# ── Constants ──────────────────────────────────────────────
TOOLS = [
    "army_navy","bulldog","castroviejo","clamp","forceps","frazier",
    "hemostat","iris","mayo_metz","needle","potts","richardson",
    "scalpel","towel_clip","weitlaner","yankauer","scissors",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')

# ── Global state ───────────────────────────────────────────
_lock = threading.Lock()
S = {
    'surgery_active':  False,
    'surgery_start':   None,
    'tool_timers':     {t: 0 for t in TOOLS},
    'tool_states':     {t: 'on_table' for t in TOOLS},
    'tool_pick_count': {t: 0 for t in TOOLS},
    'prev_in_hand':    {t: False for t in TOOLS},
    'held_history':    {t: deque(maxlen=7) for t in TOOLS},
    'pre_tools':       {},
    'last_detected':   set(),
    'events':          [],
    'event_seq':       0,
    'fps':             0,
    '_fps_cnt':        0,
    '_fps_t':          time.time(),
    'hold_threshold':  0.20,          # hand-tool score threshold (0.05 – 0.90)
    'tool_scores':     {t: 0.0 for t in TOOLS},  # latest overlap score per tool
}

_latest_jpeg = None
_jpeg_lock   = threading.Lock()

# ── Helper functions ───────────────────────────────────────
def _evt(msg, level="ok"):
    """Must be called while holding _lock."""
    S['event_seq'] += 1
    ts = datetime.now().strftime("%H:%M:%S")
    S['events'].append({'id': S['event_seq'], 'ts': ts, 'level': level, 'msg': msg})
    if len(S['events']) > 120:
        del S['events'][:60]

def _score(bh, bt):
    """Hand-tool overlap score."""
    hx1,hy1,hx2,hy2 = bh;  tx1,ty1,tx2,ty2 = bt
    ix1=max(hx1,tx1); iy1=max(hy1,ty1); ix2=min(hx2,tx2); iy2=min(hy2,ty2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    ta = max((tx2-tx1)*(ty2-ty1),1)
    ha = max((hx2-hx1)*(hy2-hy1),1)
    union = ta+ha-inter
    cont  = inter/ta
    tc_x  = (tx1+tx2)/2;  tc_y=(ty1+ty2)/2
    cin   = float(hx1<=tc_x<=hx2 and hy1<=tc_y<=hy2)
    iou   = inter/union if union else 0
    hc_x  = (hx1+hx2)/2;  hc_y=(hy1+hy2)/2
    hd    = max(((hx2-hx1)**2+(hy2-hy1)**2)**.5, 1)
    prox  = max(0, 1-((tc_x-hc_x)**2+(tc_y-hc_y)**2)**.5/(hd*1.5))
    return 0.40*cont + 0.30*cin + 0.20*iou + 0.10*prox

STATUS_BGR = {
    'on_table': (80,  200,  80),   # green
    'in_hand':  (73,   81, 248),   # red
    'missing':  (34,  153, 210),   # yellow/orange
}

# ── Camera processing thread ───────────────────────────────
def _open_cap():
    cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_FFMPEG)
    if isinstance(CAMERA_SOURCE, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def _process_loop():
    global _latest_jpeg
    import numpy as np

    cap = _open_cap()
    fail_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            if fail_count > 30:
                print(f"[WARN] Stream lost, reconnecting: {CAMERA_SOURCE}")
                cap.release()
                time.sleep(1.0)
                cap = _open_cap()
                fail_count = 0
            else:
                time.sleep(0.05)
            continue
        fail_count = 0

        h_f, w_f = frame.shape[:2]

        # FPS counter
        with _lock:
            S['_fps_cnt'] += 1
            now = time.time()
            if now - S['_fps_t'] >= 1.0:
                S['fps']      = S['_fps_cnt']
                S['_fps_cnt'] = 0
                S['_fps_t']   = now

        # ── Hand detection (MediaPipe) ────────────────────
        detected_hands = []
        if HAND_LM and _mp:
            try:
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

        # ── YOLO tool detection ───────────────────────────
        detected_tools = {}
        detected_conf  = {}
        if MODEL_OK and model:
            try:
                res = model(frame, conf=0.35, verbose=False, imgsz=416)[0]
                for box in res.boxes:
                    c   = box.xyxy[0].tolist()
                    lbl = model.names[int(box.cls[0])].lower()
                    cf  = float(box.conf[0])
                    if lbl in TOOLS:
                        if lbl not in detected_conf or cf > detected_conf[lbl]:
                            detected_tools[lbl] = c
                            detected_conf[lbl]  = cf
                    x1,y1,x2,y2 = int(c[0]),int(c[1]),int(c[2]),int(c[3])
                    with _lock:
                        bc = STATUS_BGR.get(S['tool_states'].get(lbl,'on_table'), (88,166,255))
                    cv2.rectangle(frame,(x1,y1),(x2,y2),bc,2)
                    cv2.putText(frame,f"{lbl} {cf:.2f}",(x1,max(y1-6,12)),0,0.42,(230,237,243),1)
            except Exception as ex:
                cv2.putText(frame,f"YOLO error: {ex}",(10,50),0,0.45,(0,0,200),1)

        with _lock:
            S['last_detected'] = set(detected_tools.keys())
            s_active           = S['surgery_active']
            threshold          = S['hold_threshold']

        # Hand-tool scores computed outside lock (always, for live UI display)
        live_scores = {}
        for tool in TOOLS:
            if tool in detected_tools and detected_hands:
                live_scores[tool] = round(
                    max(_score(h, detected_tools[tool]) for h in detected_hands), 3)
            else:
                live_scores[tool] = 0.0

        # Apply threshold to determine held state
        held_raw = {}
        if s_active:
            for tool in TOOLS:
                held_raw[tool] = live_scores[tool] >= threshold

        # Store latest scores for API
        with _lock:
            S['tool_scores'].update(live_scores)

        # ── State update (under lock) ─────────────────────
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
                                _evt(f"{tool.replace('_',' ').title()} picked up "
                                     f"(#{S['tool_pick_count'][tool]})", "ok")
                        else:
                            if prev:
                                _evt(f"{tool.replace('_',' ').title()} released", "")
                            S['tool_states'][tool] = 'on_table'
                    else:
                        if S['tool_states'][tool] == 'in_hand':
                            S['tool_states'][tool] = 'missing'
                            _evt(f"WARNING: {tool.replace('_',' ').title()} disappeared!", "warn")

                    S['prev_in_hand'][tool] = is_held

        # ── HUD overlay ───────────────────────────────────
        with _lock:
            fps_v = S['fps']
            s_act = S['surgery_active']

        cv2.putText(frame, f"FPS:{fps_v}", (w_f-80, 22), 0, 0.55, (57,208,216), 2)
        s_txt = "SURGERY ACTIVE" if s_act else "STANDBY"
        s_col = (80,200,80) if s_act else (120,120,120)
        cv2.putText(frame, s_txt, (10, h_f-12), 0, 0.48, s_col, 1)

        SZ=22; T=2
        for px,py,dx,dy in [(0,0,1,1),(w_f,0,-1,1),(0,h_f,1,-1),(w_f,h_f,-1,-1)]:
            cv2.line(frame,(px,py),(px+dx*SZ,py),(88,166,255),T)
            cv2.line(frame,(px,py),(px,py+dy*SZ),(88,166,255),T)

        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if ok:
            with _jpeg_lock:
                _latest_jpeg = buf.tobytes()

    cap.release()

# ── MJPEG stream ───────────────────────────────────────────
def _gen_frames():
    while True:
        with _jpeg_lock:
            fb = _latest_jpeg
        if fb:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + fb + b'\r\n'
        time.sleep(0.013)

# ── Flask routes ───────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/video_feed')
def video_feed():
    return Response(_gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/state')
def api_state():
    with _lock:
        elapsed = 0
        if S['surgery_active'] and S['surgery_start']:
            elapsed = int(time.time() - S['surgery_start'])
        data = {
            'surgery_active':  S['surgery_active'],
            'elapsed':         elapsed,
            'fps':             S['fps'],
            'tool_timers':     dict(S['tool_timers']),
            'tool_states':     dict(S['tool_states']),
            'tool_pick_count': dict(S['tool_pick_count']),
            'tool_scores':     dict(S['tool_scores']),
            'hold_threshold':  S['hold_threshold'],
            'events':          list(S['events'][-40:]),
        }
    return jsonify(data)

@app.route('/api/start', methods=['POST'])
def api_start():
    with _lock:
        if S['surgery_active']:
            return jsonify({'ok': False, 'reason': 'already active'})
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
    return jsonify({'ok': True})

@app.route('/api/end', methods=['POST'])
def api_end():
    with _lock:
        if not S['surgery_active']:
            return jsonify({'ok': False, 'reason': 'not active'})
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
    return jsonify({'ok': True, 'missing': missing, 'elapsed': elapsed})

@app.route('/api/set_threshold', methods=['POST'])
def api_set_threshold():
    """Update the hand-tool overlap score threshold (0.05 – 0.90)."""
    body = request.get_json(silent=True) or {}
    try:
        val = float(body.get('threshold', 0.20))
    except (TypeError, ValueError):
        return jsonify({'ok': False, 'reason': 'invalid value'}), 400
    val = round(max(0.05, min(0.90, val)), 3)
    with _lock:
        S['hold_threshold'] = val
    return jsonify({'ok': True, 'hold_threshold': val})

@app.route('/api/config')
def api_config():
    with _lock:
        return jsonify({'hold_threshold': S['hold_threshold']})

@app.route('/api/export_excel')
def api_export_excel():
    """Generate and return an Excel report of the current session."""
    if not EXCEL_OK:
        return jsonify({'error': 'openpyxl not installed on server'}), 500

    with _lock:
        timers     = dict(S['tool_timers'])
        states     = dict(S['tool_states'])
        picks      = dict(S['tool_pick_count'])
        events     = list(S['events'])
        s_active   = S['surgery_active']
        s_start    = S['surgery_start']
        elapsed    = int(time.time() - s_start) if s_active and s_start else 0

    wb  = openpyxl.Workbook()
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Styles ────────────────────────────────────────────
    hdr_fill   = PatternFill("solid", fgColor="1C2128")
    title_fill = PatternFill("solid", fgColor="161B22")

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
    dark_fill   = PatternFill("solid", fgColor="0D1117")
    mid_fill    = PatternFill("solid", fgColor="161B22")

    thin = Border(
        left=Side(style="thin", color="30363D"),
        right=Side(style="thin", color="30363D"),
        top=Side(style="thin", color="30363D"),
        bottom=Side(style="thin", color="30363D"),
    )
    center = Alignment(horizontal="center", vertical="center")
    left   = Alignment(horizontal="left",   vertical="center")

    # ── Sheet 1: Tool Summary ─────────────────────────────
    ws1 = wb.active
    ws1.title = "Tool Summary"

    ws1.merge_cells("A1:F1")
    ws1["A1"] = "Smart Surgical Assistant — Session Report"
    ws1["A1"].font = title_font
    ws1["A1"].fill = title_fill
    ws1["A1"].alignment = center
    ws1.row_dimensions[1].height = 28

    ws1.merge_cells("A2:F2")
    ws1["A2"] = f"Generated: {ts_str}   |   Surgery active: {'Yes' if s_active else 'No'}"
    ws1["A2"].font = dim_font
    ws1["A2"].fill = title_fill
    ws1["A2"].alignment = left

    for col, h in enumerate(["Tool Name","Use Time (s)","Pick Count","Status","Notes"], 1):
        cell = ws1.cell(row=4, column=col, value=h)
        cell.font = hdr_font; cell.fill = hdr_fill
        cell.border = thin; cell.alignment = center
    ws1.row_dimensions[4].height = 20

    status_map = {
        'on_table': ("ON TABLE", green_font, green_fill),
        'in_hand':  ("IN HAND",  red_font,   red_fill),
        'missing':  ("MISSING",  yellow_font, yellow_fill),
    }

    for i, tool in enumerate(TOOLS):
        r   = i + 5
        sec = timers[tool] // 10
        bg  = dark_fill if i % 2 == 0 else mid_fill
        stxt, sfont, sfill = status_map.get(states[tool], status_map['on_table'])

        for col, val in enumerate([tool.replace("_"," ").title(), sec, picks[tool], stxt, ""], 1):
            cell = ws1.cell(row=r, column=col, value=val)
            cell.border = thin
            cell.alignment = left if col == 1 else center
            if col == 4:
                cell.font = sfont; cell.fill = sfill
            else:
                cell.font = normal_font; cell.fill = bg

    for col, w in enumerate([22,14,12,14,20], 1):
        ws1.column_dimensions[get_column_letter(col)].width = w

    # ── Sheet 2: Event Log ────────────────────────────────
    ws2 = wb.create_sheet("Event Log")
    ws2.merge_cells("A1:C1")
    ws2["A1"] = "Event Log"
    ws2["A1"].font = title_font; ws2["A1"].fill = title_fill
    ws2["A1"].alignment = center; ws2.row_dimensions[1].height = 26

    for col, h in enumerate(["Timestamp","Level","Message"], 1):
        cell = ws2.cell(row=2, column=col, value=h)
        cell.font = hdr_font; cell.fill = hdr_fill
        cell.border = thin; cell.alignment = center

    lvl_colors = {
        "ok":    ("3FB950","0D3320"),
        "warn":  ("D29922","3D2C00"),
        "err":   ("F85149","3D1210"),
        "ai":    ("39D0D8","0D2A2C"),
        "info":  ("8B949E","161B22"),
        "":      ("8B949E","161B22"),
    }

    for i, ev in enumerate(events):
        r = i + 3
        fg, bg = lvl_colors.get(ev['level'], ("8B949E","161B22"))
        efont = Font(name="Calibri", color=fg, size=9)
        efill = PatternFill("solid", fgColor=bg)
        for col, val in enumerate([ev['ts'], ev['level'].upper(), ev['msg']], 1):
            cell = ws2.cell(row=r, column=col, value=val)
            cell.font = efont; cell.fill = efill; cell.border = thin
            cell.alignment = Alignment(horizontal="left", vertical="center",
                                       wrap_text=(col == 3))

    ws2.column_dimensions["A"].width = 14
    ws2.column_dimensions["B"].width = 10
    ws2.column_dimensions["C"].width = 70

    # ── Return as download ────────────────────────────────
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    fname = f"surgical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return send_file(
        buf,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=fname,
    )

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(BASE_DIR, path)

# ── Startup ────────────────────────────────────────────────
if __name__ == '__main__':
    threading.Thread(target=_process_loop, daemon=True).start()
    print("\n" + "="*52)
    print("  Smart Surgical Assistant  |  Flask Server")
    print("  Open in browser: http://localhost:5000")
    print("="*52 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
