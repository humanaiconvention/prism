import tkinter as tk
from tkinter import scrolledtext
import threading
import subprocess
import queue
import re
import os
import sys

# Import phases from go.py
from go import PHASES

# Hacker Theme + PRISM Spectrum Colors
BG_COLOR = "#050505"
FG_COLOR = "#00FF41"
ERR_COLOR = "#FF003C"
HL_COLOR = "#1A1A1A"
SEL_BG = "#111111"
SEL_FG = "#FFFFFF"

# Newton's Spectrum
SPECTRUM = ["#FF0000", "#FF8C00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#EE82EE"]
FONT = ("Consolas", 10)
BOLD_FONT = ("Consolas", 10, "bold")
HEADER_FONT = ("Consolas", 12, "bold")
LOGO_FONT = ("Consolas", 24, "bold")

def phase_sort_key(pid):
    match = re.search(r'(\d+)([A-Z_0-9]*)', pid)
    if match:
        num = int(match.group(1))
        suffix = match.group(2)
        return (num, suffix)
    return (0, pid)

class PrismGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PRISM /// Spectral Interpretability Terminal /// Opticks v1704")
        self.geometry("1200x850")
        self.configure(bg=BG_COLOR)
        self.process = None
        self.output_queue = queue.Queue()

        self._build_ui()
        self._populate_phases()
        self.after(100, self._process_queue)

    def _build_ui(self):
        # --- HEADER AREA ---
        self.header_frame = tk.Frame(self, bg=BG_COLOR)
        self.header_frame.pack(fill=tk.X, padx=20, pady=(10, 0))

        logo_text = "PRISM"
        for i, char in enumerate(logo_text):
            tk.Label(self.header_frame, text=char, font=LOGO_FONT, bg=BG_COLOR, 
                     fg=SPECTRUM[i % len(SPECTRUM)]).pack(side=tk.LEFT)
        
        tk.Label(self.header_frame, text=" | SPECTRAL INTERPRETABILITY SUITE", font=HEADER_FONT, 
                 bg=BG_COLOR, fg="#666666").pack(side=tk.LEFT, padx=10)
        
        tk.Label(self.header_frame, text="\"Standing on the shoulders of giants\"", font=("Consolas", 9, "italic"), 
                 bg=BG_COLOR, fg="#444444").pack(side=tk.RIGHT, pady=10)

        # Main Container
        self.main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=BG_COLOR, bd=0, sashwidth=2, sashbg="#222222")
        self.main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Frame (Phase List)
        self.left_frame = tk.Frame(self.main_pane, bg=BG_COLOR)
        self.main_pane.add(self.left_frame, minsize=320)

        tk.Label(self.left_frame, text="EXPERIMENTUM CRUCIS", font=HEADER_FONT, bg=BG_COLOR, fg=SPECTRUM[3]).pack(anchor=tk.W, pady=(0, 5))
        
        self.listbox = tk.Listbox(self.left_frame, bg=BG_COLOR, fg="#AAAAAA", font=FONT, 
                                  selectbackground="#222222", selectforeground=SPECTRUM[2],
                                  highlightthickness=1, highlightcolor=SPECTRUM[0], highlightbackground="#111111", bd=0)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.bind('<<ListboxSelect>>', self._on_phase_select)

        # Right Frame (Details + Console)
        self.right_pane = tk.PanedWindow(self.main_pane, orient=tk.VERTICAL, bg=BG_COLOR, bd=0, sashwidth=2, sashbg="#222222")
        self.main_pane.add(self.right_pane, minsize=750)

        # Details Area (Principia Area)
        self.details_frame = tk.Frame(self.right_pane, bg=BG_COLOR)
        self.right_pane.add(self.details_frame, minsize=300)
        
        tk.Label(self.details_frame, text="PRINCIPIA INTEL", font=HEADER_FONT, bg=BG_COLOR, fg=SPECTRUM[1]).pack(anchor=tk.W, pady=(0, 5))
        self.info_text = tk.Text(self.details_frame, bg=BG_COLOR, fg="#999999", font=FONT, wrap=tk.WORD, 
                                 state=tk.DISABLED, bd=0, highlightthickness=1, highlightbackground="#111111")
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Tags for info text
        self.info_text.tag_config('header', font=BOLD_FONT, foreground=SPECTRUM[2])
        self.info_text.tag_config('narrative', foreground="#CCCCCC")
        self.info_text.tag_config('technical', foreground="#888888")
        self.info_text.tag_config('link', foreground=SPECTRUM[4], underline=True)

        # Console Area (The Dark Side)
        self.console_frame = tk.Frame(self.right_pane, bg=BG_COLOR)
        self.right_pane.add(self.console_frame, minsize=350)

        tk.Label(self.console_frame, text="THE DARK SIDE (TERMINAL)", font=HEADER_FONT, bg=BG_COLOR, fg=SPECTRUM[4]).pack(anchor=tk.W, pady=(0, 5))
        self.console_text = scrolledtext.ScrolledText(self.console_frame, bg="#020202", fg=FG_COLOR, font=FONT, 
                                                      wrap=tk.WORD, bd=0, highlightthickness=1, highlightbackground="#111111")
        self.console_text.pack(fill=tk.BOTH, expand=True)
        
        for i, color in enumerate(SPECTRUM):
            self.console_text.tag_config(f'rainbow_{i}', foreground=color)
        self.console_text.tag_config('error', foreground=ERR_COLOR)
        self.console_text.tag_config('system', foreground="#666666")

        # Controls Area (Bottom)
        self.controls_frame = tk.Frame(self, bg=BG_COLOR)
        self.controls_frame.pack(fill=tk.X, padx=10, pady=10)

        btn_style = {"bg": "#0A0A0A", "fg": "#888888", "font": ("Consolas", 10, "bold"), 
                     "activebackground": "#222222", "activeforeground": "#FFFFFF", 
                     "bd": 0, "highlightthickness": 1, "padx": 15, "pady": 5, "cursor": "hand2"}
        
        self.btn_run = tk.Button(self.controls_frame, text="RUN THE RIG", command=self._run_selected, 
                                 highlightbackground=SPECTRUM[3], **btn_style)
        self.btn_run.pack(side=tk.LEFT, padx=5)

        self.btn_clear = tk.Button(self.controls_frame, text="CLEAR GLASS", command=self._clear_console, 
                                   highlightbackground=SPECTRUM[2], **btn_style)
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        self.btn_stop = tk.Button(self.controls_frame, text="ECLIPSE", command=self._stop_process, 
                                  state=tk.DISABLED, highlightbackground=SPECTRUM[0], **btn_style)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(self.controls_frame, text="There is no dark side of the moon really. Matter of fact it's all dark.", 
                                     font=("Consolas", 8), bg=BG_COLOR, fg="#333333")
        self.status_label.pack(side=tk.LEFT, padx=30)
        
        self.btn_quit = tk.Button(self.controls_frame, text="BREATHE (OUT)", command=self.destroy, 
                                  highlightbackground="#444444", **btn_style)
        self.btn_quit.pack(side=tk.RIGHT, padx=5)

    def _populate_phases(self):
        self.sorted_pids = sorted(PHASES.keys(), key=phase_sort_key)
        for pid in self.sorted_pids:
            name = PHASES[pid]['name']
            self.listbox.insert(tk.END, f" {pid.ljust(6)} | {name}")

    def _on_phase_select(self, event):
        selection = self.listbox.curselection()
        if not selection: return
        
        idx = selection[0]
        pid = self.sorted_pids[idx]
        pinfo = PHASES[pid]

        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        
        self.info_text.insert(tk.END, f"PHASE ID:      {pid}\n", 'header')
        self.info_text.insert(tk.END, f"DESIGNATION:   {pinfo['name']}\n", 'header')
        self.info_text.insert(tk.END, f"OBJECTIVE:     {pinfo['description']}\n\n")
        
        self.info_text.insert(tk.END, "--- SIMPLE NARRATIVE ---\n", 'header')
        self.info_text.insert(tk.END, f"{pinfo['narrative']}\n\n", 'narrative')
        
        self.info_text.insert(tk.END, "--- TECHNICAL INTEL ---\n", 'header')
        self.info_text.insert(tk.END, f"{pinfo['technical']}\n\n", 'technical')
        
        self.info_text.insert(tk.END, "--- SYSTEM CONFIG ---\n", 'header')
        self.info_text.insert(tk.END, f"SOURCE:        {pinfo['script']}\n")
        self.info_text.insert(tk.END, f"PARAM:         {pinfo['args']}\n")
        self.info_text.insert(tk.END, f"PREREQ:        {', '.join(pinfo['dependencies']) if pinfo['dependencies'] else 'TABULA RASA'}\n\n")
        
        self.info_text.insert(tk.END, "DEEPER ANSWERS (NotebookLM):\n", 'header')
        self.info_text.insert(tk.END, "https://notebooklm.google.com/notebook/1a68b472-4bac-4293-8a5e-04452633415b\n", 'link')

        self.info_text.config(state=tk.DISABLED)
        self.status_label.config(text=f"Selected: {pinfo['name']} ... any colour you like.", fg="#555555")

    def _log(self, msg, tag=None):
        self.console_text.insert(tk.END, msg, tag)
        self.console_text.see(tk.END)

    def _rainbow_log(self, msg):
        for i, char in enumerate(msg):
            self._log(char, f'rainbow_{i % len(SPECTRUM)}')

    def _clear_console(self):
        self.console_text.delete(1.0, tk.END)

    def _run_selected(self):
        if self.process and self.process.poll() is None:
            self._log("\n[!] A spectral process is already in orbit.\n", 'error')
            return

        selection = self.listbox.curselection()
        if not selection:
            self._log("\n[!] No rig selected for execution.\n", 'error')
            return
        
        idx = selection[0]
        pid = self.sorted_pids[idx]
        
        self._log(f"\n>> INITIATING {PHASES[pid]['name']}... ", 'system')
        self._log(f"Opticks phase {pid}\n")
        
        self.btn_run.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_label.config(text="Ticking away the moments that make up a dull day...", fg=SPECTRUM[3])

        threading.Thread(target=self._exec_process, args=(pid,), daemon=True).start()

    def _exec_process(self, pid):
        try:
            cmd = [sys.executable, "-u", "go.py", pid]
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, startupinfo=startupinfo
            )

            for line in self.process.stdout:
                self.output_queue.put(line)

            self.process.wait()
            if self.process.returncode == 0:
                self.output_queue.put(f"___SUCCESS_{pid}___")
            else:
                self.output_queue.put(f"\n[-] PHASE {pid} ECLIPSED WITH CODE {self.process.returncode}.\n")

        except Exception as e:
            self.output_queue.put(f"\n[!] SYSTEM FAILURE: {str(e)}\n")
        finally:
            self.output_queue.put("___DONE___")

    def _stop_process(self):
        if self.process and self.process.poll() is None:
            self._log("\n[!] TRIGGERING TOTAL ECLIPSE...\n", 'error')
            self.process.terminate()

    def _process_queue(self):
        try:
            while True:
                msg = self.output_queue.get_nowait()
                if msg == "___DONE___":
                    self.btn_run.config(state=tk.NORMAL)
                    self.btn_stop.config(state=tk.DISABLED)
                elif msg.startswith("___SUCCESS_"):
                    pid = msg.split("_")[3]
                    self._rainbow_log(f"\n[+] PHASE {pid} SUCCESSFUL. ALL YOU TOUCH AND ALL YOU SEE.\n")
                    self.status_label.config(text="The sun is the same in a relative way but you're older.", fg=SPECTRUM[1])
                else:
                    self._log(msg)
        except queue.Empty:
            pass
        finally:
            self.after(50, self._process_queue)

if __name__ == "__main__":
    app = PrismGUI()
    app.mainloop()

