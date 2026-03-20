import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import subprocess
from pathlib import Path

class PulmonaryTrioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pulmonary Trio Tracer - AI Controller")
        self.root.geometry("750x700")
        
        # 取得目前 main_gui.py 所在的根目錄
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 變數儲存
        self.ct_dir = tk.StringVar()
        self.seed_a = tk.StringVar()
        self.seed_v = tk.StringVar()
        self.seed_b = tk.StringVar()
        self.mode = tk.StringVar(value="global") # 預設全域模式
        
        self.setup_ui()

    def setup_ui(self):
        # --- 1. CT 影像載入區 ---
        frame_input = tk.LabelFrame(self.root, text=" 1. CT Data Input ", padx=10, pady=10)
        frame_input.pack(fill="x", padx=20, pady=10)

        tk.Label(frame_input, text="CT Series (JPGs):").grid(row=0, column=0, sticky="w")
        tk.Entry(frame_input, textvariable=self.ct_dir, width=55).grid(row=0, column=1, padx=5)
        tk.Button(frame_input, text="Browse", command=self.browse_ct).grid(row=0, column=2)

        # --- 2. 模式選擇 ---
        frame_mode = tk.LabelFrame(self.root, text=" 2. Inference Mode ", padx=10, pady=10)
        frame_mode.pack(fill="x", padx=20, pady=10)

        tk.Radiobutton(frame_mode, text="Global (No Seed)", variable=self.mode, value="global", 
                       command=self.toggle_seed_ui).pack(side="left", padx=20)
        tk.Radiobutton(frame_mode, text="Seed-based (Expert Tracking)", variable=self.mode, value="seed", 
                       command=self.toggle_seed_ui).pack(side="left", padx=20)

        # --- 3. 種子路徑區 ---
        self.frame_seeds = tk.LabelFrame(self.root, text=" 3. Seed Masks (Required for Seed Mode) ", padx=10, pady=10)
        self.frame_seeds.pack(fill="x", padx=20, pady=10)

        self.setup_seed_row("Artery Seed:", self.seed_a, 0)
        self.setup_seed_row("Vein Seed:", self.seed_v, 1)
        self.setup_seed_row("Bronchi Seed:", self.seed_b, 2)
        
        self.toggle_seed_ui() # 初始狀態檢查

        # --- 4. 執行與進度 ---
        self.btn_run = tk.Button(self.root, text=" RUN PIPELINE", bg="#27ae60", fg="white", 
                                 font=("Arial", 12, "bold"), height=2, command=self.start_thread)
        self.btn_run.pack(fill="x", padx=40, pady=10)

        self.log_text = tk.Text(self.root, height=12, bg="#2c3e50", fg="#ecf0f1", font=("Consolas", 10))
        self.log_text.pack(fill="both", padx=20, pady=10)

    def setup_seed_row(self, label, var, row):
        tk.Label(self.frame_seeds, text=label).grid(row=row, column=0, sticky="w")
        tk.Entry(self.frame_seeds, textvariable=var, width=55).grid(row=row, column=1, padx=5)
        tk.Button(self.frame_seeds, text="Browse", command=lambda: self.browse_seed(var)).grid(row=row, column=2)

    def toggle_seed_ui(self):
        state = "normal" if self.mode.get() == "seed" else "disabled"
        for child in self.frame_seeds.winfo_children():
            child.configure(state=state)

    def browse_ct(self):
        path = filedialog.askdirectory()
        if path: self.ct_dir.set(path)

    def browse_seed(self, var):
        path = filedialog.askdirectory()
        if path: var.set(path)

    def log(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)

    def start_thread(self):
        if not self.ct_dir.get():
            messagebox.showwarning("Warning", "Please select CT directory first!")
            return
        
        # 種子模式檢查
        if self.mode.get() == "seed":
            if not all([self.seed_a.get(), self.seed_v.get(), self.seed_b.get()]):
                messagebox.showwarning("Warning", "Please select all 3 seed directories for Seed Mode!")
                return

        self.btn_run.config(state="disabled", text=" Processing... Window may seem frozen, check logs below")
        self.log_text.delete('1.0', tk.END)
        threading.Thread(target=self.run_pipeline, daemon=True).start()

    def run_pipeline(self):
        try:
            # 1. 配置環境變數，解決 ModuleNotFoundError
            my_env = os.environ.copy()
            # 將根目錄加入 PYTHONPATH
            my_env["PYTHONPATH"] = self.base_dir + os.pathsep + my_env.get("PYTHONPATH", "")
            my_env["PYTHONIOENCODING"] = "utf-8"
            my_env["PYTHONLEGACYWINDOWSSTDIO"] = "1" # 強制 Windows 使用舊版 IO 模式，通常更穩定

            ct_path = self.ct_dir.get()
            output_base = os.path.join(ct_path, "Output_Results")
            
            # 2. 準備執行指令
            if self.mode.get() == "global":
                self.log(" Mode: Global Inference (No Seeds)")
                script_path = os.path.join("scripts", "run_global.py")
                cmd = ["python", script_path, 
                       "--ct_dir", ct_path, 
                       "--output_dir", output_base,
                       "--trio_model", "checkpoint/trio_tracer_fast.pt", 
                       "--bronchi_model", "checkpoint/bronchial_expert.pt"]
            else:
                self.log(" Mode: Seed-based Expert Tracking")
                script_path = os.path.join("scripts", "run_seed_track.py")
                cmd = ["python", script_path, 
                       "--ct_dir", ct_path, 
                       "--output_dir", output_base,
                       "--trio_model", "checkpoint/trio_tracer_fast.pt", 
                       "--bronchi_model", "checkpoint/bronchial_expert.pt",
                       "--seed_a", self.seed_a.get(), 
                       "--seed_v", self.seed_v.get(), 
                       "--seed_b", self.seed_b.get()]

            # 3. 執行推論腳本
            self.log(f" Launching process from: {self.base_dir}")
            # 使用 cwd=self.base_dir 確保腳本內部的相對路徑正確
            result = subprocess.run(cmd, env=my_env, cwd=self.base_dir, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode != 0:
                self.log(f" Script Error Output:\n{result.stderr}")
                raise Exception("Subprocess failed. Check log for details.")

            self.log(" Inference Complete. Starting visualization...")

            # 4. 呼叫視覺化模組 (從根目錄 import)
            # 在這裡我們直接導入腳本中的類別，避免再次調用 subprocess
            sys.path.append(self.base_dir)
            from scripts.visualize_overlay import TrioVisualizer
            
            overlay_dir = os.path.join(output_base, "Overlay_Visualization")
            viz = TrioVisualizer(alpha=0.5)
            viz.process(ct_path, output_base, overlay_dir)
            
            self.log(f" Success! Results saved in: {overlay_dir}")
            
            # 自動開啟資料夾 (Windows)
            os.startfile(overlay_dir)

        except Exception as e:
            self.log(f" Critical Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.btn_run.config(state="normal", text=" RUN PIPELINE")

if __name__ == "__main__":
    root = tk.Tk()
    # 設置視窗圖標 (如果有 icon 檔的話)
    # root.iconbitmap("icon.ico") 
    app = PulmonaryTrioApp(root)
    root.mainloop()