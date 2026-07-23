import json
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "pipeline.py"
SETTINGS = ROOT / "settings.json"

# Ordered pipeline stages (mirrors config.STAGE_ORDER). Import is auto-run; the rest are
# manual "Run" buttons that unlock in sequence. QA + Package unlock together after Render.
STAGES = ["import", "detect", "ocr", "translate", "paraphrase", "render", "qa"]
STAGE_LABELS = {
    "import": "0 · Import",
    "detect": "1 · Detect",
    "ocr": "2 · OCR",
    "translate": "3 · Translate",
    "paraphrase": "4 · Paraphrase",
    "render": "5 · Render",
    "qa": "6 · QA",
}
PACKAGE_FORMATS = ["cbz", "zip", "tar", "pdf"]

# Stages that use a prompt/response handoff when backend is manual/mcp.
# (paths mirror config STAGE_FOLDERS + *_PROMPT/RESPONSE_NAME defaults)
HANDOFF_HINTS = {
    "translate": ("stage3_translation/translation_prompt.json",
                  "stage3_translation/translation_response.json"),
    "paraphrase": ("stage4_paraphrase/paraphrase_prompt.json",
                   "stage4_paraphrase/paraphrase_response.json"),
}

# Only surface meaningful lines in the log (warnings, re-runs, overrides, verdicts) -
# hide the per-page INFO spam. Case-insensitive substring match.
SHOW_KEYWORDS = (
    "warning", "error", "failed", "review", "success", "done:",
    "held", "not advancing", "re-run", "rerun", "handoff",
    "override", "glossary", "not translated", "missing", "giving up",
    "completion", "conflict", "skip", "credits", "stitch",
)


def _read_settings() -> dict:
    if SETTINGS.exists():
        try:
            return json.loads(SETTINGS.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _write_setting(key: str, value) -> None:
    """Read-modify-write settings.json so we never clobber pipeline-managed keys."""
    s = _read_settings()
    s[key] = value
    try:
        SETTINGS.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _chapter_name(input_path: str) -> str:
    """Chapter folder name = stem for cbz/zip, folder name otherwise (mirrors import)."""
    p = Path(input_path)
    if p.suffix.lower() in {".cbz", ".zip"}:
        return p.stem
    return p.name


class PipelineGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Manhua Pipeline - Guided Runner")
        root.geometry("640x680")

        self.q: queue.Queue = queue.Queue()
        self.running = False
        self.input_path: str | None = None
        self.chapter: str | None = None
        self.stage_done: dict[str, bool] = {}
        self.stage_btns: dict[str, ttk.Button] = {}

        self._build_ui()
        self._poll_queue()

    def _manifest_current_stage(self) -> str | None:
        """Read <series>/<chapter>/manifest.json 'current_stage'. None if unreadable."""
        try:
            mpath = Path(self.series_var.get().strip()) / self.chapter / "manifest.json"
            return json.loads(mpath.read_text(encoding="utf-8")).get("current_stage")
        except Exception:
            return None

    # ---------------- UI ----------------
    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # Series folder (persisted)
        top = ttk.LabelFrame(self.root, text="Series folder (where chapters are stored)")
        top.pack(fill="x", **pad)
        self.series_var = tk.StringVar(value=_read_settings().get("output_dir", ""))
        ttk.Entry(top, textvariable=self.series_var).pack(side="left", fill="x", expand=True, padx=6, pady=6)
        ttk.Button(top, text="Browse…", command=self._pick_series).pack(side="right", padx=6)

        # Input picker
        inp = ttk.LabelFrame(self.root, text="Chapter input")
        inp.pack(fill="x", **pad)
        row = ttk.Frame(inp); row.pack(fill="x", padx=6, pady=6)
        ttk.Button(row, text="Choose Folder…", command=self._pick_folder).pack(side="left")
        ttk.Button(row, text="Choose CBZ/ZIP…", command=self._pick_archive).pack(side="left", padx=6)
        self.input_lbl = ttk.Label(inp, text="No input selected.", foreground="#888")
        self.input_lbl.pack(anchor="w", padx=6, pady=(0, 6))
        self.fresh_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(inp, text="Fresh  (⚠ wipes prior outputs for this chapter)",
                        variable=self.fresh_var).pack(anchor="w", padx=6, pady=(0, 6))

        self.skip_last_var = tk.IntVar(value=0)
        row2 = ttk.Frame(inp)
        row2.pack(anchor="w", padx=6, pady=(0, 6))
        ttk.Label(row2, text="Skip last").pack(side="left")
        ttk.Spinbox(row2, from_=0, to=20, width=4, textvariable=self.skip_last_var).pack(side="left", padx=4)
        ttk.Label(row2, text="pages (promo/credits at chapter end)").pack(side="left")

        # Stages
        st = ttk.LabelFrame(self.root, text="Stages (unlock in order)")
        st.pack(fill="x", **pad)
        for name in STAGES:
            r = ttk.Frame(st); r.pack(fill="x", padx=6, pady=2)
            btn = ttk.Button(r, text=STAGE_LABELS[name], width=18,
                             command=lambda n=name: self._run_stage(n))
            btn.pack(side="left")
            btn.state(["disabled"])
            self.stage_btns[name] = btn
            if name == "qa":
                # Package sits beside QA (both unlock after Render).
                self.pkg_fmt = tk.StringVar(value=_read_settings().get("gui_default_package", "cbz"))
                self.pkg_btn = ttk.Button(r, text="Package", width=12, command=self._run_package)
                self.pkg_btn.pack(side="left", padx=(12, 4))
                self.pkg_btn.state(["disabled"])
                ttk.Combobox(r, textvariable=self.pkg_fmt, values=PACKAGE_FORMATS,
                             width=6, state="readonly").pack(side="left")

        # Log
        lg = ttk.LabelFrame(self.root, text="Output (warnings · re-runs · overrides · verdicts)")
        lg.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(lg, height=14, wrap="word", state="disabled",
                           font=("Consolas", 9), bg="#111", fg="#ddd")
        self.log.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        sb = ttk.Scrollbar(lg, command=self.log.yview); sb.pack(side="right", fill="y", pady=6)
        self.log.config(yscrollcommand=sb.set)
        self.log.tag_config("err", foreground="#ff6b6b")
        self.log.tag_config("warn", foreground="#ffd166")
        self.log.tag_config("ok", foreground="#8ce99a")
        self.log.tag_config("sys", foreground="#74c0fc")

    # ---------------- pickers ----------------
    def _pick_series(self):
        d = filedialog.askdirectory(title="Select series folder")
        if d:
            self.series_var.set(d)
            self._set_output_dir(d)

    def _pick_folder(self):
        d = filedialog.askdirectory(title="Select chapter folder (images)")
        if d:
            self._on_input_selected(d)

    def _pick_archive(self):
        f = filedialog.askopenfilename(title="Select CBZ/ZIP",
                                       filetypes=[("Comic archives", "*.cbz *.zip")])
        if f:
            self._on_input_selected(f)

    # ---------------- flow ----------------
    def _on_input_selected(self, path: str):
        if self.running:
            messagebox.showinfo("Busy", "A stage is still running - please wait.")
            return
        if not self.series_var.get().strip():
            messagebox.showwarning("Series folder", "Pick a series folder first.")
            return
        self.input_path = path
        self.chapter = _chapter_name(path)
        self.input_lbl.config(text=f"{Path(path).name}   →   chapter '{self.chapter}'", foreground="#ddd")

        # RESET everything for the new chapter
        self.skip_last_var.set(0)
        self.stage_done = {s: False for s in STAGES}
        for n, b in self.stage_btns.items():
            b.state(["disabled"])
            b.configure(text=STAGE_LABELS[n])
        self.pkg_btn.state(["disabled"])
        self._clear_log()

        if self.fresh_var.get():
            if not messagebox.askyesno("Fresh run",
                                       f"Fresh will WIPE prior outputs for '{self.chapter}'.\nContinue?"):
                return
        self._sys(f"Input selected: {self.chapter}. Auto-running Import…")
        self._run_stage("import", auto=True)

    def _run_stage(self, name: str, auto: bool = False):
        if self.running:
            return
        cmd = [sys.executable, str(PIPELINE)]
        if name == "import":
            cmd += ["import", "--input", self.input_path]
            if self.fresh_var.get():
                cmd += ["--fresh"]
            if self.skip_last_var.get() > 0:
                cmd += ["--skip-last", str(self.skip_last_var.get())]
        else:
            cmd += [name, "--chapter", self.chapter]
        self._run_cmd(cmd, on_done=lambda rc: self._stage_finished(name, rc))

    def _run_package(self):
        if self.running:
            return
        fmt = self.pkg_fmt.get()
        _write_setting("gui_default_package", fmt)   # remember choice
        cmd = [sys.executable, str(PIPELINE), "package", "--chapter", self.chapter, "--package", fmt]
        self._run_cmd(cmd, on_done=lambda rc: self._sys(
            f"Package {'done' if rc == 0 else 'FAILED'} ({fmt}).", "ok" if rc == 0 else "err"))

    def _stage_finished(self, name: str, rc: int):
        # Hard failure -> keep this stage enabled so the user can retry.
        if rc != 0:
            self._sys(f"{STAGE_LABELS.get(name, name)} did not complete cleanly - "
                      f"fix the issue above and re-run this stage.", "err")
            self.stage_btns[name].state(["!disabled"])
            return

        cs = self._manifest_current_stage()
        if cs is None:
            self._sys("Couldn't read manifest to confirm the stage advanced - "
                      "re-run this stage if needed.", "warn")
            self.stage_btns[name].state(["!disabled"])
            return

        # Handoff pending: exit code was 0 but the manifest did NOT advance past this stage,
        # i.e. the stage only wrote its prompt and is awaiting a manual/MCP response.
        if cs == name:
            self._handoff_pending(name)
            return

        # Truly complete (manifest advanced past this stage, or reached 'complete').
        self.stage_done[name] = True
        self.stage_btns[name].configure(text=STAGE_LABELS[name])   # clear any "Resume" label
        self._sys(f"{STAGE_LABELS.get(name, name)} complete.", "ok")

        if name == "render":
            self.stage_btns["qa"].state(["!disabled"])
            self.pkg_btn.state(["!disabled"])
            self._sys("QA and Package are now available (run either, or both).", "sys")
        else:
            idx = STAGES.index(name)
            if idx + 1 < len(STAGES):
                self.stage_btns[STAGES[idx + 1]].state(["!disabled"])

    def _handoff_pending(self, name: str):
        """Stage wrote its prompt and is waiting for a manual/MCP response. Keep it active
        as 'Resume', tell the user exactly what to do, and DON'T unlock the next stage."""
        prompt_rel, resp_rel = HANDOFF_HINTS.get(name, ("<prompt>.json", "<response>.json"))
        btn = self.stage_btns[name]
        btn.configure(text=f"Resume ▶ {STAGE_LABELS[name]}")
        btn.state(["!disabled"])
        self._sys(f"{STAGE_LABELS.get(name, name)} needs a manual/MCP handoff:", "warn")
        self._sys(f"   1. Prompt written to:  {self.chapter}/{prompt_rel}", "warn")
        self._sys(f"   2. Get the result - paste it to your chatbot, or run your MCP tool.", "warn")
        self._sys(f"   3. Save the reply as:  {self.chapter}/{resp_rel}", "warn")
        self._sys(f"   4. Click 'Resume ▶ {STAGE_LABELS[name]}' to ingest and continue.", "warn")

    def _set_output_dir(self, path: str):
        self._run_cmd([sys.executable, str(PIPELINE), "--set-output-dir", path],
                      on_done=lambda rc: self._sys(f"Series folder saved.", "ok") if rc == 0 else None,
                      quiet=True)

    # ---------------- subprocess plumbing ----------------
    def _run_cmd(self, cmd: list[str], on_done=None, quiet: bool = False):
        self.running = True
        self._set_buttons_running(True)
        if not quiet:
            self._sys("$ " + " ".join(Path(c).name if c == cmd[1] else c for c in cmd), "sys")

        def worker():
            try:
                proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1,encoding="utf-8", errors="replace")
                for line in proc.stdout:
                    self.q.put(("line", line.rstrip("\n")))
                proc.wait()
                self.q.put(("done", proc.returncode))
            except Exception as exc:
                self.q.put(("line", f"[GUI] Failed to launch: {exc}"))
                self.q.put(("done", 1))
            finally:
                self.q.put(("__cb__", on_done))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self):
        pending_cb = None
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "line":
                    self._maybe_log(payload)
                elif kind == "done":
                    self._last_rc = payload
                elif kind == "__cb__":
                    self.running = False
                    self._set_buttons_running(False)
                    pending_cb = payload
        except queue.Empty:
            pass
        if pending_cb is not None:
            try:
                pending_cb(getattr(self, "_last_rc", 1))
            except Exception:
                pass
        self.root.after(120, self._poll_queue)

    def _set_buttons_running(self, busy: bool):
        # While a stage runs, disable all stage buttons; _stage_finished re-enables the right one.
        for b in self.stage_btns.values():
            if busy:
                b.state(["disabled"])
        if busy:
            self.pkg_btn.state(["disabled"])

    # ---------------- log helpers ----------------
    def _maybe_log(self, line: str):
        low = line.lower()
        if not any(k in low for k in SHOW_KEYWORDS):
            return
        tag = "err" if ("error" in low or "failed" in low or "giving up" in low) else \
              "warn" if ("warning" in low or "held" in low or "missing" in low
                         or "not translated" in low or "conflict" in low) else None
        self._append(line, tag)

    def _sys(self, msg: str, tag: str = "sys"):
        self._append(f"» {msg}", tag)

    def _append(self, text: str, tag=None):
        self.log.config(state="normal")
        self.log.insert("end", text + "\n", (tag,) if tag else ())
        self.log.see("end")
        self.log.config(state="disabled")

    def _clear_log(self):
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")


def main():
    if not PIPELINE.exists():
        print(f"pipeline.py not found next to this GUI ({PIPELINE}). "
              f"Place pipeline_gui.py in the project root.")
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    PipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
