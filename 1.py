import os
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
import requests
import time
import threading

# --- 1. TKINTER AND PATH SETTINGS ---
os.environ['TCL_LIBRARY'] = r'C:\Users\baris\AppData\Local\Programs\Python\Python313\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Users\baris\AppData\Local\Programs\Python\Python313\tcl\tk8.6'

# --- 2. MODEL AND API KEY (EDIT HERE) ---
MODEL_PATH = r"C:\Users\baris\Downloads\best (3).pt"
API_KEY = "YOUR_OPENROUTER_API_KEY_HERE"

class SurgicalAssistantApp:
    def __init__(self, window):
        self.window = window
        self.window.title("AI-Powered Smart Surgical Assistant")
        self.window.configure(bg="#0a0a0a")

        # Model loading
        self.model = YOLO(MODEL_PATH)
        self.class_names = self.model.names

        # Tracking variables
        self.tool_timers = {name: 0.0 for name in self.class_names.values() if "hand" not in name.lower() and "el" not in name.lower()}
        self.tool_states = {name: "on_table" for name in self.tool_timers.keys()}
        self.last_api_call = time.time()

        self.setup_ui()
        self.setup_camera()

# ... (Previous imports and Tkinter path fixes remain the same) ...

    def update_loop(self):
        ret, frame = self.cap.read()
        if not ret: return

        # CHANGE 1: Lower confidence threshold to 0.20 for more sensitive detection
        results = self.model(frame, conf=0.20, verbose=False)[0]

        # CHANGE 2: Print how many objects the model found this frame (for diagnostics)
        if len(results.boxes) > 0:
            print(f"Detected: {len(results.boxes)} object(s) found.")

        hands = []
        current_tools = {}

        for box in results.boxes:
            b = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            label = self.class_names[cls_id]

            # CHANGE 3: Draw EVERYTHING without conditions
            # If boxes appear now, the problem is in the 'if' checks below.
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {box.conf[0]:.2f}", (int(b[0]), int(b[1]-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Logical grouping
            clean_label = label.lower()
            if "hand" in clean_label or "el" in clean_label:
                hands.append(b)
            elif clean_label in [t.lower() for t in self.tool_timers.keys()]:
                current_tools[label] = b

        # ... (Remaining timer and AI logic unchanged) ...

    def setup_ui(self):
        """UI Layout"""
        # Left Panel: Camera and Log
        self.left_frame = tk.Frame(self.window, bg="#0a0a0a")
        self.left_frame.grid(row=0, column=0, padx=20, pady=20)

        self.cam_label = tk.Label(self.left_frame, bg="black", borderwidth=2, relief="solid")
        self.cam_label.pack()

        self.log_text = tk.Text(self.left_frame, height=10, width=65, bg="#121212", fg="#00ff00", font=("Consolas", 10))
        self.log_text.pack(pady=10)
        self.log_text.insert(tk.END, ">>> System Ready. Processing video...\n")

        # Right Panel: Timers and Statuses
        self.right_frame = tk.Frame(self.window, bg="#0a0a0a")
        self.right_frame.grid(row=0, column=1, sticky="ns", padx=10)

        tk.Label(self.right_frame, text="TOOL TRACKING", fg="white", bg="#0a0a0a", font=("Arial", 14, "bold")).pack(pady=10)

        self.timer_widgets = {}
        for tool in self.tool_timers.keys():
            frame = tk.Frame(self.right_frame, bg="#1a1a1a", pady=5)
            frame.pack(fill="x", pady=5)

            name_lbl = tk.Label(frame, text=tool.upper(), fg="#00ccff", bg="#1a1a1a", font=("Arial", 10, "bold"))
            name_lbl.pack()

            time_lbl = tk.Label(frame, text="0.0s", fg="white", bg="#1a1a1a", font=("Arial", 12))
            time_lbl.pack()

            status_lbl = tk.Label(frame, text="STATUS: STANDBY", fg="gray", bg="#1a1a1a", font=("Arial", 8))
            status_lbl.pack()

            self.timer_widgets[tool] = {"time": time_lbl, "status": status_lbl}

    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.update_loop()

    def call_llm(self, report_text):
        """Get AI report via OpenRouter"""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "nousresearch/hermes-3-l3.1-405b",
            "messages": [
                {"role": "system", "content": "You are an expert surgical assistant. Analyze the given tool usage times and statuses, and provide the surgeon with professional, concise, and clear feedback."},
                {"role": "user", "content": report_text}
            ]
        }
        try:
            response = requests.post(url, json=data, headers=headers, timeout=10)
            ai_msg = response.json()['choices'][0]['message']['content']
            self.log_text.insert(tk.END, f"\n[AI REPORT - {time.strftime('%H:%M')}]: {ai_msg}\n")
            self.log_text.see(tk.END)
        except Exception as e:
            self.log_text.insert(tk.END, f"\n[ERROR]: Could not reach AI API. {e}\n")

    def is_touching(self, b1, b2):
        return not (b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1])

    def update_loop(self):
        ret, frame = self.cap.read()
        if not ret: return

        # 1. YOLO inference
        results = self.model(frame, conf=0.35, verbose=False)[0]

        hands = []
        current_tools = {}

        for box in results.boxes:
            b = box.xyxy[0].tolist()
            label = self.class_names[int(box.cls[0])]

            # Visualization
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(b[0]), int(b[1]-10)), 0, 0.5, (255,255,255), 2)

            if "hand" in label.lower() or "el" in label.lower():
                hands.append(b)
            elif label in self.tool_timers:
                current_tools[label] = b

        # 2. Logic and timers
        for tool, timer in self.tool_timers.items():
            is_visible = tool in current_tools
            is_held = False

            if is_visible:
                for h in hands:
                    if self.is_touching(h, current_tools[tool]):
                        is_held = True
                        break

                if is_held:
                    self.tool_timers[tool] += 0.1
                    self.tool_states[tool] = "in_hand"
                    self.timer_widgets[tool]["time"].config(text=f"{self.tool_timers[tool]:.1f}s", fg="#ff4444")
                    self.timer_widgets[tool]["status"].config(text="STATUS: IN USE", fg="#ff4444")
                else:
                    self.tool_states[tool] = "on_table"
                    self.timer_widgets[tool]["time"].config(fg="#44ff44")
                    self.timer_widgets[tool]["status"].config(text="STATUS: ON TABLE", fg="#44ff44")
            else:
                # Tool not visible and was last in hand: MISSING
                if self.tool_states[tool] == "in_hand":
                    self.tool_states[tool] = "missing"
                    self.timer_widgets[tool]["status"].config(text="STATUS: MISSING/LOST!", fg="yellow")
                    self.log_text.insert(tk.END, f"!!! WARNING: {tool.upper()} MISSING !!!\n")

        # 3. AI Reporting (every 20 seconds)
        if time.time() - self.last_api_call > 20:
            summary = f"Timers: {self.tool_timers}, States: {self.tool_states}"
            threading.Thread(target=self.call_llm, args=(summary,)).start()
            self.last_api_call = time.time()

        # Tkinter render
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(img))
        self.cam_label.configure(image=img)
        self.cam_label.image = img
        self.window.after(100, self.update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = SurgicalAssistantApp(root)
    root.mainloop()
