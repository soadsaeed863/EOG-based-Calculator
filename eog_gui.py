"""
EOG Gaze-Controlled Calculator  —  CSV Column Mode  v3
=======================================================
CSV format  : 9 columns, one signal (movement) per column.
              Pattern repeats: [Move1, Move2, Blink] × 3
              e.g.  Up | Right | Blink | Down | Left | Blink | Up | Left | Blink

Highlighting :
  Move1  → the predicted direction square lights up (sky-blue)
  Move2  → ONLY the single resolved button lights up (orange)
  Blink  → Center flashes (violet); confirmed button flashes (green)
            → Digit 1 / Operator / Digit 2 / Result box updates NOW

Calculator logic (from old GUI, unchanged):
  Phase d1  → expects a digit  → fills Digit 1
  Phase op  → expects operator → fills Operator
  Phase d2  → expects a digit  → fills Digit 2 + computes Result
  If the sequence produces a valid expression  → show result
  If anything is invalid (wrong type at wrong phase) → show warning dialog
"""

import importlib, os, threading, json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np


# import joblib, os, traceback

# models_dir = r"C:\Users\Asus\Downloads\EOG_Project\EOG_Project\EOG_Project\Models\saved_models"

# for f in sorted(os.listdir(models_dir)):
#     if not f.endswith('.joblib'): continue
#     try:
#         bundle = joblib.load(os.path.join(models_dir, f))
#         model  = bundle['model'] if isinstance(bundle, dict) else bundle
#         if hasattr(model, 'classes_'):
#             print(f"{f:35s}  classes_ = {list(model.classes_)}")
#         else:
#             print(f"{f:35s}  no classes_ attr")
#     except Exception as e:
#         print(f"{f:35s}  FAILED: {e}")

# ─────────────────────────────────────────────────────────────
#  OPTIONAL HEAVY IMPORTS
# ─────────────────────────────────────────────────────────────
pywt       = None
AutoReg    = None
joblib     = None
pd         = None
butter     = None
filtfilt   = None
_deps_ok   = False
_deps_err  = ""

try:
    joblib       = importlib.import_module('joblib')
    pd           = importlib.import_module('pandas')
    scipy_signal = importlib.import_module('scipy.signal')
    butter       = scipy_signal.butter
    filtfilt     = scipy_signal.filtfilt
    _deps_ok     = True
except ImportError as _ie:
    _deps_ok  = False
    _deps_err = str(_ie)

try:
    pywt = importlib.import_module('pywt')
except ImportError:
    pywt = None

try:
    sm = importlib.import_module('statsmodels.tsa.ar_model')
    AutoReg = sm.AutoReg
except ImportError:
    AutoReg = None


# ─────────────────────────────────────────────────────────────
#  SIGNAL PROCESSING
# ─────────────────────────────────────────────────────────────
FS, BPF_LOW, BPF_HIGH, BPF_ORDER = 176, 0.5, 20.0, 2
DWT_WAV, DWT_LEVEL, AR_LAGS      = 'db4', 2, 6


def _bandpass(signal):
    nyq  = FS / 2.0
    b, a = butter(BPF_ORDER, [BPF_LOW / nyq, BPF_HIGH / nyq], btype='band')
    return filtfilt(b, a, signal)


def extract_features(sig_h, sig_v, ft):
    fh, fv = _bandpass(sig_h), _bandpass(sig_v)
    if ft == 'raw':
        return np.concatenate([fh, fv])
    if ft == 'dwt':
        ch = pywt.wavedec(fh, DWT_WAV, level=DWT_LEVEL)[0]
        cv = pywt.wavedec(fv, DWT_WAV, level=DWT_LEVEL)[0]
        return np.concatenate([ch, cv])
    if ft == 'ar':
        def _ar(s):
            m = AutoReg(s, lags=AR_LAGS).fit()
            return np.array(m.params[:7])
        return np.concatenate([_ar(fh), _ar(fv)])
    raise ValueError(f"Unknown feature type: {ft}")


# ─────────────────────────────────────────────────────────────
#  MODEL STORE
# ─────────────────────────────────────────────────────────────
_loaded_models: dict = {}


def load_models_from_dir(models_dir, status_cb):
    """
    Always scan the folder for every .joblib file and use the filename
    (without extension) as the model key.  This is robust regardless of
    whether models_manifest.json exists, is missing its extension, or
    lists only a subset of the models.
    """
    global _loaded_models

    jfiles = sorted([f for f in os.listdir(models_dir) if f.endswith('.joblib')])
    if not jfiles:
        status_cb("No .joblib models found in the selected folder"); return

    # Build key → path from filenames (e.g. "raw_SVM.joblib" → key "raw_SVM")
    paths = {os.path.splitext(f)[0]: os.path.join(models_dir, f) for f in jfiles}

    _loaded_models = {}
    for key, path in paths.items():
        try:
            _loaded_models[key] = joblib.load(path)
            status_cb(f"Loaded: {key}")
        except Exception as e:
            status_cb(f"Failed to load {key}: {e}")

    status_cb(f"{len(_loaded_models)} model(s) ready  —  keys: {list(_loaded_models)}")


def run_predict_signal(signal_array, model_key):
    """Predict from a 1-D numpy array (used as both H and V channel)."""
    if model_key not in _loaded_models:
        raise KeyError(f"Model '{model_key}' not loaded.")
    bundle = _loaded_models[model_key]
    sig    = signal_array.astype(float)

    # Parse feature type(s) from key, e.g. "raw+ar+dwt_SVM" → ['raw','ar','dwt']
    ft_part = model_key.split('_', 1)[0]   # e.g. "raw+ar+dwt"
    ft_list = ft_part.split('+')           # e.g. ['raw','ar','dwt']

    # ── Trim signal to match training length ──────────────────
    # Model stores n_features_in_ = total features it was trained on.
    # For 'raw': features = 2 * n_samples  →  trim signal to n_expected // 2
    # For 'ar' : features = 2 * 11 = 22   →  signal length doesn't matter
    # For 'dwt': features = 2 * dwt_coeff →  trim signal proportionally
    # Safe approach: trim signal so raw contribution = n_features_in_ // 2,
    # then trim/pad full X to n_features_in_ as a final safety net.
    model      = bundle['model']
    n_expected = getattr(model, 'n_features_in_', None)

    if n_expected is not None:
        # Estimate how many raw samples the model expects per channel
        # raw contributes 2*n_samples, ar contributes 22, dwt contributes ~2*(n//4+1)
        # Simplest robust fix: trim signal so we don't overshoot
        if 'raw' in ft_list:
            raw_contribution = n_expected
            if 'ar'  in ft_list: raw_contribution -= 22
            if 'dwt' in ft_list: raw_contribution -= 2 * (len(sig) // 4 + 1)
            target_len = max(AR_LAGS + 2, raw_contribution // 2)
            if len(sig) > target_len:
                sig = sig[:target_len]
        elif 'dwt' in ft_list:
            # dwt only: trim so wavedec output * 2 = n_expected
            # wavedec level=2 on N samples → approx N//4 + 1 coeffs
            target_len = (n_expected // 2 - 1) * 4
            target_len = max(AR_LAGS + 2, target_len)
            if len(sig) > target_len:
                sig = sig[:target_len]
        # ar-only: AR params are always 11 per channel, signal length irrelevant

    fh, fv = _bandpass(sig), _bandpass(sig)   # filter once, reuse both channels

    parts = []
    for ft in ft_list:
        if ft == 'raw':
            parts.append(np.concatenate([fh, fv]))
        elif ft == 'dwt':
            if pywt is None:
                raise ImportError("pywt not installed — cannot use dwt features")
            ch = pywt.wavedec(fh, DWT_WAV, level=DWT_LEVEL)[0]
            cv = pywt.wavedec(fv, DWT_WAV, level=DWT_LEVEL)[0]
            parts.append(np.concatenate([ch, cv]))
        elif ft == 'ar':
            if AutoReg is None:
                raise ImportError("statsmodels not installed — cannot use ar features")
            def _ar(s):
                m = AutoReg(s, lags=AR_LAGS).fit()
                return np.array(m.params[:11])
            parts.append(np.concatenate([_ar(fh), _ar(fv)]))
        else:
            raise ValueError(f"Unknown feature type: {ft}")

    X = np.concatenate(parts).reshape(1, -1)

    # ── Final safety: trim or pad to exactly n_expected ───────
    if n_expected is not None and X.shape[1] != n_expected:
        if X.shape[1] > n_expected:
            X = X[:, :n_expected]
        else:
            X = np.pad(X, ((0, 0), (0, n_expected - X.shape[1])))

    if bundle.get('needs_scaling', False):
        X = bundle['scaler'].transform(X)

    label = str(bundle['model'].predict(X)[0])

    try:    conf = max(bundle['model'].predict_proba(X)[0]) * 100
    except: conf = None
    return label, conf


# ─────────────────────────────────────────────────────────────
#  GAZE STATE MACHINE  (unchanged from original)
# ─────────────────────────────────────────────────────────────
def op(x, c, z):
    x = x.lower()
    if   x=='up'    and c=='center':       c='center_up'
    elif x=='down'  and c=='center':       c='center_down'
    elif x=='left'  and c=='center':       c='center_left'
    elif x=='right' and c=='center':       c='center_right'
    elif x=='up'    and c=='center_up':    z='4'
    elif x=='down'  and c=='center_up':    z='6'
    elif x=='left'  and c=='center_up':    z='5'
    elif x=='right' and c=='center_up':    z='7'
    elif x=='up'    and c=='center_down':  z='0'
    elif x=='down'  and c=='center_down':  z='2'
    elif x=='left'  and c=='center_down':  z='1'
    elif x=='right' and c=='center_down':  z='3'
    elif x=='up'    and c=='center_left':  z='8'
    elif x=='down'  and c=='center_left':  z='E'
    elif x=='left'  and c=='center_left':  z='9'
    elif x=='right' and c=='center_left':  z='C'
    elif x=='up'    and c=='center_right': z='/'
    elif x=='down'  and c=='center_right': z='+'
    elif x=='left'  and c=='center_right': z='*'
    elif x=='right' and c=='center_right': z='-'
    elif x in ('blink', 'blinking'):       c='center'
    return c, z


# ─────────────────────────────────────────────────────────────
#  LAYOUT CONSTANTS
# ─────────────────────────────────────────────────────────────
CW, CH  = 900, 700
S       = 65
CX      = CW // 2
UY, DY  = 175, 530
LX, RX  = 175, CW - 175
MY      = 352

DIR_XY = {
    "Up":     (CX, UY),
    "Down":   (CX, DY),
    "Left":   (LX, MY),
    "Right":  (RX, MY),
    "Center": (CX, MY),
}

BTN_XY = {
    '4': (CX,      UY - S), '5': (CX - S,  UY),
    '6': (CX,      UY + S), '7': (CX + S,  UY),
    '0': (CX,      DY - S), '1': (CX - S,  DY),
    '2': (CX,      DY + S), '3': (CX + S,  DY),
    '8': (LX,      MY - S), '9': (LX - S,  MY),
    'C': (LX + S,  MY),     'E': (LX,      MY + S),
    '/': (RX,      MY - S), '*': (RX - S,  MY),
    '-': (RX + S,  MY),     '+': (RX,      MY + S),
}

QUADRANT_BTNS = {
    "Up":    ['4','5','6','7'],
    "Down":  ['0','1','2','3'],
    "Left":  ['8','9','C','E'],
    "Right": ['/','*','-','+'],
}

BTN_SIZE = 58

ALL_MODEL_KEYS = [
    f"{ft}_{mn}"
    for ft in ['raw', 'ar', 'dwt']
    for mn in ['SVM', 'ExtraTrees', 'RF', 'GradBoost']
]

# ── Colours ──────────────────────────────────────────────────
BG_BLUE         = '#5b7fa6'
BTN_BG          = '#d4d4d4'
BTN_FG          = '#000000'
DIR_BG          = '#ffffff'
DIR_FG          = '#000000'
ACCENT          = '#7c3aed'
PANEL           = '#1e1e2e'
TEXT            = '#e2e8f0'
MUTED           = '#94a3b8'
SUCCESS         = '#059669'
CLR_CONFIRMED   = '#4ade80'   # green  – blink-confirmed button flash
CLR_RESULT      = '#facc15'   # yellow – result box
CLR_PENDING     = '#f59e0b'   # amber  – pending label
CLR_MOVE1       = '#38bdf8'   # sky-blue – Move1 direction highlight
CLR_MOVE2       = '#fb923c'   # orange   – Move2 button highlight
CLR_BLINK       = '#a78bfa'   # violet   – blink center flash
CLR_STEP_DONE   = '#4ade80'   # green  – step box after confirmation
CLR_STEP_WAIT   = '#64748b'   # slate  – step box before confirmation
CLR_STEP_ACTIVE = '#f59e0b'   # amber  – "Now entering" label


# ─────────────────────────────────────────────────────────────
#  APP
# ─────────────────────────────────────────────────────────────
class EOGApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EOG Gaze Calculator — CSV Mode v3")
        self.geometry("1100x880")
        self.configure(bg=PANEL)
        self.resizable(True, True)

        # ── State vars ──────────────────────────────────────
        self._models_dir  = tk.StringVar(value=os.path.join(os.getcwd(), "saved_models"))
        self._model_key   = tk.StringVar(value="raw_SVM")
        self._status_var  = tk.StringVar(value="Ready — load models, then load a CSV file")
        self._display     = tk.StringVar(value="")
        self._phase_var   = tk.StringVar(value="Digit 1")

        # CSV: list of (col_name, np.array)
        self._csv_signals: list = []
        self._csv_path    = tk.StringVar(value="No file loaded")
        self._col_idx     = 0          # next column to predict

        # Calculator state (same as old GUI)
        self._digit1 = ""
        self._oper   = ""
        self._digit2 = ""
        self._phase  = "d1"            # d1 → op → d2 → result

        # Gaze state machine
        self._gaze_state  = "center"
        self._pending_btn = "_"
        self._move_step   = 1          # 1 = Move1 expected, 2 = Move2 expected

        # Widget refs
        self._dir_widgets: dict = {}
        self._btn_widgets: dict = {}

        self._build_ui()
        self._try_autoload()

    # ══════════════════════════════════════════════════════════
    #  BUILD UI
    # ══════════════════════════════════════════════════════════
    def _build_ui(self):
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)
        self._build_control_bar()
        self._build_scrollable_area()
        self._build_status_bar()

    # ── Control bar ───────────────────────────────────────────
    def _build_control_bar(self):
        bar = tk.Frame(self, bg=PANEL, padx=8, pady=6)
        bar.grid(row=0, column=0, sticky="ew")
        bar.grid_columnconfigure(1, weight=1)

        # Models dir
        r0 = tk.Frame(bar, bg=PANEL)
        r0.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0,3))
        r0.grid_columnconfigure(1, weight=1)
        _lbl(r0, "Models dir:", PANEL).grid(row=0, column=0, sticky="w")
        tk.Entry(r0, textvariable=self._models_dir,
                 bg='#2a2a3e', fg=TEXT, relief='flat',
                 font=('Consolas', 8)).grid(row=0, column=1, sticky="ew", padx=4)
        _btn(r0, "Browse",      self._browse_models).grid(row=0, column=2, padx=(0,3))
        _btn(r0, "Load Models", self._load_models_async,
             bg=SUCCESS).grid(row=0, column=3)

        # Model selector
        r1 = tk.Frame(bar, bg=PANEL)
        r1.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(0,3))
        _lbl(r1, "Model:", PANEL).pack(side='left')
        self._model_combo = ttk.Combobox(
            r1, textvariable=self._model_key,
            values=ALL_MODEL_KEYS, state='readonly',
            width=20, font=('Segoe UI', 9, 'bold'))
        self._model_combo.pack(side='left', padx=6)
        _lbl(r1, "  (feature type encoded in model name)",
             PANEL, fg=MUTED, font=('Segoe UI', 8)).pack(side='left')

        # CSV loader
        r2 = tk.Frame(bar, bg=PANEL)
        r2.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(0,3))
        r2.grid_columnconfigure(1, weight=1)
        _lbl(r2, "CSV / Excel:", PANEL).grid(row=0, column=0, sticky="w")
        tk.Entry(r2, textvariable=self._csv_path,
                 bg='#2a2a3e', fg=TEXT, relief='flat',
                 font=('Consolas', 8),
                 state='readonly').grid(row=0, column=1, sticky="ew", padx=4)
        _btn(r2, "Load CSV", self._load_csv,
             bg='#1e40af').grid(row=0, column=2, padx=(0,3))
        _btn(r2, "Reset",    self._reset_csv,
             bg='#7f1d1d').grid(row=0, column=3)

        # Progress bar
        r3 = tk.Frame(bar, bg=PANEL)
        r3.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(0,3))
        _lbl(r3, "Progress:", PANEL, font=('Segoe UI', 8)).pack(side='left')
        self._progress_var = tk.DoubleVar(value=0)
        self._progress_bar = ttk.Progressbar(
            r3, variable=self._progress_var, maximum=9, length=300)
        self._progress_bar.pack(side='left', padx=6)
        self._progress_lbl = _lbl(r3, "Col 0 / 0", PANEL,
                                   fg=MUTED, font=('Segoe UI', 8))
        self._progress_lbl.pack(side='left', padx=4)

        # Predict buttons + colour legend
        r4 = tk.Frame(bar, bg=PANEL)
        r4.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(2,0))
        _btn(r4, "Predict Next Column", self._run_predict,
             bg=ACCENT, font=('Segoe UI', 10, 'bold')).pack(side='left')
        _btn(r4, "Run All  ▶▶", self._run_all,
             bg='#0f4c81', font=('Segoe UI', 9, 'bold')).pack(side='left', padx=8)
        self._step_lbl = _lbl(r4, "   Next: —", PANEL,
                               fg=MUTED, font=('Segoe UI', 8))
        self._step_lbl.pack(side='left', padx=8)
        # Legend
        for colour, label in [
            (CLR_MOVE1, "Move1"), (CLR_MOVE2, "Move2"),
            (CLR_BLINK, "Blink"), (CLR_CONFIRMED, "Confirmed"),
        ]:
            tk.Label(r4, text="  ", bg=colour, width=2).pack(side='left', padx=(6,1))
            _lbl(r4, label, PANEL, fg=MUTED,
                 font=('Segoe UI', 7)).pack(side='left', padx=(0,4))

    # ── Scrollable area ───────────────────────────────────────
    def _build_scrollable_area(self):
        container = tk.Frame(self, bg=PANEL)
        container.grid(row=1, column=0, sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        canvas  = tk.Canvas(container, bg=PANEL, highlightthickness=0)
        vscroll = ttk.Scrollbar(container, orient='vertical',   command=canvas.yview)
        hscroll = ttk.Scrollbar(container, orient='horizontal', command=canvas.xview)
        canvas.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)

        vscroll.grid(row=0, column=1, sticky='ns')
        hscroll.grid(row=1, column=0, sticky='ew')
        canvas.grid( row=0, column=0, sticky='nsew')

        canvas.bind_all("<MouseWheel>",
            lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        canvas.bind_all("<Shift-MouseWheel>",
            lambda e: canvas.xview_scroll(int(-1*(e.delta/120)), "units"))

        inner  = tk.Frame(canvas, bg=PANEL)
        win_id = canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win_id, width=e.width))

        self._build_inner_content(inner)

    def _build_inner_content(self, parent):
        # ── Display bar ──────────────────────────────────────
        disp_frame = tk.Frame(parent, bg=PANEL)
        disp_frame.pack(fill='x', padx=20, pady=(12, 4))
        tk.Entry(disp_frame, textvariable=self._display,
                 font=('Arial', 26, 'bold'), bd=6, justify='right',
                 bg='white', fg='black', relief='sunken',
                 state='readonly', readonlybackground='white'
                 ).pack(fill='x', ipady=6)

        # ── Step boxes ───────────────────────────────────────
        # These update ONLY when a blink confirms a button press
        step_frame = tk.Frame(parent, bg=PANEL)
        step_frame.pack(fill='x', padx=20, pady=(0, 4))

        self._lbl_d1     = self._step_box(step_frame, "Digit 1",  "—")
        tk.Label(step_frame, text="  ", bg=PANEL).pack(side='left')
        self._lbl_op     = self._step_box(step_frame, "Operator", "—")
        tk.Label(step_frame, text="  ", bg=PANEL).pack(side='left')
        self._lbl_d2     = self._step_box(step_frame, "Digit 2",  "—")
        tk.Label(step_frame, text="  ", bg=PANEL).pack(side='left')
        self._lbl_result = self._step_box(step_frame, "Result",   "—", wide=True)

        # ── "Now entering" label ─────────────────────────────
        phase_frame = tk.Frame(parent, bg=PANEL)
        phase_frame.pack(fill='x', padx=20, pady=(0, 2))
        _lbl(phase_frame, "Now entering:", PANEL,
             fg=MUTED, font=('Segoe UI', 9)).pack(side='left')
        tk.Label(phase_frame, textvariable=self._phase_var,
                 bg=PANEL, fg=CLR_STEP_ACTIVE,
                 font=('Arial', 13, 'bold')).pack(side='left', padx=6)

        # ── Pending indicator ─────────────────────────────────
        pend_frame = tk.Frame(parent, bg=PANEL)
        pend_frame.pack(fill='x', padx=20, pady=(0, 2))
        _lbl(pend_frame, "Pending:", PANEL, fg=MUTED,
             font=('Segoe UI', 9)).pack(side='left')
        self._pending_var = tk.StringVar(value="none")
        self._pending_lbl = tk.Label(
            pend_frame, textvariable=self._pending_var,
            bg=PANEL, fg=CLR_PENDING, font=('Arial', 16, 'bold'))
        self._pending_lbl.pack(side='left', padx=4)
        _lbl(pend_frame, "  blink to confirm",
             PANEL, fg=MUTED, font=('Segoe UI', 8)).pack(side='left')

        # ── Calculator canvas ─────────────────────────────────
        calc_canvas = tk.Canvas(
            parent, width=CW, height=CH,
            bg=BG_BLUE, highlightthickness=0)
        calc_canvas.pack(padx=20, pady=(4, 20))
        self._draw_calculator(calc_canvas)

    def _step_box(self, parent, title, init, wide=False):
        """A step indicator box — value only updated on blink confirmation."""
        f = tk.Frame(parent, bg='#263548', padx=10, pady=4, relief='flat')
        f.pack(side='left')
        tk.Label(f, text=title, bg='#263548', fg=MUTED,
                 font=('Segoe UI', 8)).pack()
        lbl = tk.Label(f, text=init, bg='#263548', fg=CLR_STEP_WAIT,
                       font=('Arial', 22, 'bold'),
                       width=8 if wide else 4)
        lbl.pack()
        return lbl

    def _draw_calculator(self, canvas):
        for name, (cx, cy) in DIR_XY.items():
            frame = tk.Frame(canvas, bg=DIR_BG, bd=2, relief='raised')
            inner = tk.Frame(frame, bg=DIR_BG)
            inner.pack(padx=6, pady=4)
            lbl = tk.Label(inner, text=name, bg=DIR_BG, fg=DIR_FG,
                           font=('Arial', 16, 'bold'), width=7)
            lbl.pack()
            canvas.create_window(cx, cy, window=frame, anchor='center')
            self._dir_widgets[name] = (frame, lbl)

        for char, (cx, cy) in BTN_XY.items():
            btn = tk.Button(
                canvas, text=char,
                font=('Arial', 14, 'bold'),
                bg=BTN_BG, fg=BTN_FG,
                activebackground=ACCENT,
                width=3, height=1,
                relief='raised', bd=3)
            canvas.create_window(cx, cy, window=btn,
                                  width=BTN_SIZE, height=BTN_SIZE,
                                  anchor='center')
            self._btn_widgets[char] = btn

    def _build_status_bar(self):
        sb = tk.Frame(self, bg='#0f172a', height=24)
        sb.grid(row=2, column=0, sticky="ew")
        tk.Label(sb, textvariable=self._status_var,
                 bg='#0f172a', fg=MUTED, font=('Segoe UI', 8),
                 anchor='w').pack(side='left', padx=8, pady=2)

    # ══════════════════════════════════════════════════════════
    #  CSV LOADING  (reads WITH header row)
    # ══════════════════════════════════════════════════════════
    def _load_csv(self):
        path = filedialog.askopenfilename(
            title="Select CSV / Excel with 9 signal columns",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx *.xls"), ("All", "*.*")])
        if not path:
            return
        try:
            df = pd.read_excel(path) if path.lower().endswith(('.xlsx','.xls')) \
                 else pd.read_csv(path)
            if df.shape[1] != 9:
                messagebox.showwarning(
                    "Column count",
                    f"Expected 9 columns, found {df.shape[1]}.\n"
                    "Only the first 9 will be used.")
            n = min(df.shape[1], 9)
            self._csv_signals = [
                (df.columns[i], df.iloc[:, i].dropna().to_numpy())
                for i in range(n)
            ]
            self._col_idx = 0
            self._csv_path.set(os.path.basename(path))
            self._progress_bar.config(maximum=n)
            self._refresh_progress()
            cols = [c for c, _ in self._csv_signals]
            self._status_var.set(
                f"Loaded '{os.path.basename(path)}'  —  "
                f"{n} cols × {df.shape[0]} rows  |  Cols: {cols}")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def _reset_csv(self):
        self._csv_signals = []
        self._col_idx     = 0
        self._csv_path.set("No file loaded")
        self._progress_var.set(0)
        self._progress_lbl.config(text="Col 0 / 0")
        self._step_lbl.config(text="   Next: —")
        self._do_clear()
        self._status_var.set("CSV cleared — load a new file to begin")

    def _refresh_progress(self):
        total = len(self._csv_signals)
        done  = self._col_idx
        self._progress_var.set(done)
        self._progress_lbl.config(text=f"Col {done} / {total}")
        if done < total:
            col_name, _ = self._csv_signals[done]
            step_type   = ["Move1","Move2","Blink"][done % 3]
            self._step_lbl.config(
                text=f"   Next: [{step_type}]  '{col_name}'  col {done+1}")
        else:
            self._step_lbl.config(text="   All columns predicted")

    # ══════════════════════════════════════════════════════════
    #  MODEL LOADING
    # ══════════════════════════════════════════════════════════
    def _browse_models(self):
        d = filedialog.askdirectory(title="Select saved_models folder")
        if d: self._models_dir.set(d)

    def _try_autoload(self):
        d = self._models_dir.get()
        if os.path.isdir(d) and any(f.endswith('.joblib') for f in os.listdir(d)):
            self._load_models_async()

    def _load_models_async(self):
        if not _deps_ok:
            messagebox.showerror("Missing deps", _deps_err); return
        self._status_var.set("Loading models...")
        def wk():
            try:
                load_models_from_dir(
                    self._models_dir.get(),
                    lambda s: self.after(0, self._on_load_status, s))
            except Exception as e:
                self.after(0, self._status_var.set, f"Error: {e}")
        threading.Thread(target=wk, daemon=True).start()

    def _on_load_status(self, msg):
        self._status_var.set(msg)
        keys = sorted(_loaded_models.keys())
        if keys:
            self._model_combo['values'] = keys
            current = self._model_key.get()
            if current not in keys:
                preferred = next((k for k in keys if k == 'raw_SVM'), keys[0])
                self._model_key.set(preferred)

    # ══════════════════════════════════════════════════════════
    #  PREDICTION — one CSV column at a time
    # ══════════════════════════════════════════════════════════
    def _run_predict(self):
        if not _deps_ok:
            messagebox.showerror("Missing deps", _deps_err); return
        if not _loaded_models:
            messagebox.showwarning("No models", "Load models first."); return
        if not self._csv_signals:
            messagebox.showwarning("No data", "Load a CSV file first."); return
        if self._col_idx >= len(self._csv_signals):
            messagebox.showinfo("Done", "All 9 columns have been predicted."); return

        col_name, signal = self._csv_signals[self._col_idx]
        model_key  = self._model_key.get()
        step_type  = ["Move1","Move2","Blink"][self._col_idx % 3]

        self._status_var.set(
            f"Predicting col {self._col_idx+1} [{step_type}] '{col_name}' …")
        self._col_idx += 1
        self._refresh_progress()

        def wk():
            try:
                label, conf = run_predict_signal(signal, model_key)
                self.after(0, self._on_prediction, label, conf, model_key, step_type)
            except Exception as e:
                self.after(0, self._status_var.set, f"Error: {e}")
        threading.Thread(target=wk, daemon=True).start()

    def _run_all(self):
        if not _deps_ok:
            messagebox.showerror("Missing deps", _deps_err); return
        if not _loaded_models:
            messagebox.showwarning("No models", "Load models first."); return
        if not self._csv_signals:
            messagebox.showwarning("No data", "Load a CSV file first."); return
        if self._col_idx >= len(self._csv_signals):
            messagebox.showinfo("Done", "All columns already predicted."); return
        self._auto_step()

    def _auto_step(self):
        if self._col_idx >= len(self._csv_signals):
            return
        self._run_predict()
        self.after(2500, self._auto_step)

    def _on_prediction(self, label, conf, key, step_type):
        """
        step_type is determined by COLUMN POSITION, not by what the model
        predicted.  This is essential: the blink model can mispredict, so
        we NEVER trust the model to tell us it is a blink column.
          step_type == "Blink"  → always treat as confirm, ignore model label
          step_type == "Move1"  → always treat as quadrant select
          step_type == "Move2"  → always treat as button resolve
        """
        conf_str = f"{conf:.1f}%" if conf is not None else "N/A"

        move = {"Up":"up","Down":"down","Left":"left",
                "Right":"right","Blink":"blinking"}.get(label, "center")

        # ── MOVE 1: reset first, then light up direction square ─
        if step_type == "Move1":
            self._reset_all_highlights()
            self._gaze_state, self._pending_btn = op(
                move, self._gaze_state, self._pending_btn)
            self._move_step = 2

            if label in self._dir_widgets:
                fr, lb = self._dir_widgets[label]
                fr.config(bg=CLR_MOVE1); lb.config(bg=CLR_MOVE1, fg='black')
                # Stays lit until Move2 resets it

            self._pending_var.set("...")
            self._pending_lbl.config(fg=MUTED)
            self._status_var.set(
                f"Move1: '{label}' → quadrant '{self._gaze_state}' | "
                f"Now Move2 | {conf_str} | {key}")

        # ── MOVE 2: reset direction, light up resolved button ──
        elif step_type == "Move2":
            self._reset_all_highlights()
            self._gaze_state, self._pending_btn = op(
                move, self._gaze_state, self._pending_btn)
            self._move_step = 1

            if self._pending_btn in self._btn_widgets:
                self._btn_widgets[self._pending_btn].config(
                    bg=CLR_MOVE2, fg='white')
                # Stays lit until Blink resets it

            self._pending_var.set(self._pending_btn)
            self._pending_lbl.config(fg=CLR_PENDING)
            self._status_var.set(
                f"Move2: '{label}' → '{self._pending_btn}' pending | "
                f"Blink to confirm | {conf_str} | {key}")

        # ── BLINK: flash center + confirmed button, then reset ──
        else:
            btn = self._pending_btn

            self._gaze_state  = "center"
            self._pending_btn = "_"
            self._move_step   = 1
            self._pending_var.set("none")
            self._pending_lbl.config(fg=MUTED)

            # Reset all first, then flash blink colors
            self._reset_all_highlights()

            cf, cl = self._dir_widgets["Center"]
            cf.config(bg=CLR_BLINK); cl.config(bg=CLR_BLINK, fg='white')
            self.update_idletasks()
            self.after(2000, lambda cf=cf, cl=cl: (cf.config(bg=DIR_BG), cl.config(bg=DIR_BG, fg=DIR_FG)))

            if btn == 'C':
                self._do_clear()
                self._status_var.set(f"C confirmed — Cleared | model said: {label} | {conf_str}")
            elif btn == 'E':
                self._do_exit()
            elif btn != '_':
                if btn in self._btn_widgets:
                    b = self._btn_widgets[btn]
                    b.config(bg=CLR_CONFIRMED, fg='black')
                    self.after(2000, lambda b=b: b.config(bg=BTN_BG, fg=BTN_FG))
                self._do_press(btn)
                self._status_var.set(
                    f"✓ BLINK → '{btn}' confirmed | model said: {label} | {conf_str} | {key}")
            else:
                self._status_var.set(
                    f"Blink col — no pending button to confirm | model said: {label} | {key}")

    def _compute_result(self):
        """
        Check if the 3 confirmed values form a valid calculation:
          - Digit 1 must be a single digit (0-9)
          - Operator must be one of  + - * /
          - Digit 2 must be a single digit (0-9)
        If valid → compute and show result.
        If not   → show "Not a valid calculation" in Result box.
        """
        d1, op_val, d2 = self._digit1, self._oper, self._digit2
        is_valid = (
            d1.isdigit() and
            op_val in ('+', '-', '*', '/') and
            d2.isdigit()
        )
        if is_valid:
            try:
                result = str(eval(f"{d1}{op_val}{d2}"))
            except Exception:
                result = "Error"
            self._lbl_result.config(text=result, fg=CLR_RESULT)
            self._display.set(f"{d1} {op_val} {d2} = {result}")
            self._status_var.set(
                f"✓  {d1} {op_val} {d2} = {result}  |  Use C to clear")
        else:
            self._lbl_result.config(text="N/A", fg='#f87171')
            self._display.set(f"{d1}  {op_val}  {d2}  =  Not a valid calculation")
            self._status_var.set(
                f"⚠  '{d1}' '{op_val}' '{d2}' is not a valid calculation  "
                f"(need digit + operator + digit)  |  Use C to clear")

    def _update_display(self):
        parts = [p for p in [self._digit1, self._oper, self._digit2] if p]
        self._display.set("  ".join(parts))

    def _do_press(self, val):
        """Fill the next phase box in order: Digit1 → Operator → Digit2 → compute."""
        if self._phase == "d1":
            self._digit1 = val
            self._lbl_d1.config(text=val, fg=CLR_STEP_DONE)
            self._phase = "op"
            self._phase_var.set("Operator")
            self._update_display()

        elif self._phase == "op":
            self._oper = val
            self._lbl_op.config(text=val, fg=CLR_STEP_DONE)
            self._phase = "d2"
            self._phase_var.set("Digit 2")
            self._update_display()

        elif self._phase == "d2":
            self._digit2 = val
            self._lbl_d2.config(text=val, fg=CLR_STEP_DONE)
            self._phase = "result"
            self._phase_var.set("Done — use C to clear")
            self._compute_result()

        elif self._phase == "result":
            self._status_var.set(
                "Calculation done — use C (Left→Right→Blink) to clear")

    def _do_clear(self):
        self._digit1 = self._oper = self._digit2 = ""
        self._phase  = "d1"
        self._gaze_state  = "center"
        self._pending_btn = "_"
        self._move_step   = 1
        self._pending_var.set("none")
        self._pending_lbl.config(fg=MUTED)
        self._lbl_d1.config(    text='—', fg=CLR_STEP_WAIT)
        self._lbl_op.config(    text='—', fg=CLR_STEP_WAIT)
        self._lbl_d2.config(    text='—', fg=CLR_STEP_WAIT)
        self._lbl_result.config(text='—', fg=CLR_STEP_WAIT)
        self._display.set("")
        self._phase_var.set("Digit 1")
        self._reset_all_highlights()
        self._status_var.set("Cleared — Select Digit 1")

    def _do_exit(self):
        if btn := self._btn_widgets.get('E'):
            btn.config(bg=CLR_CONFIRMED, fg='black')
        if messagebox.askyesno("Exit", "Exit the EOG Calculator?"):
            self.after(300, self.destroy)

    # ══════════════════════════════════════════════════════════
    #  UI HELPERS
    # ══════════════════════════════════════════════════════════
    def _reset_all_highlights(self):
        for name, (frame, lbl) in self._dir_widgets.items():
            frame.config(bg=DIR_BG); lbl.config(bg=DIR_BG, fg=DIR_FG)
        for char, btn in self._btn_widgets.items():
            btn.config(bg=BTN_BG, fg=BTN_FG)


# ─────────────────────────────────────────────────────────────
#  WIDGET HELPERS
# ─────────────────────────────────────────────────────────────
def _lbl(parent, text, bg=PANEL, fg=TEXT, font=('Segoe UI', 9)):
    return tk.Label(parent, text=text, bg=bg, fg=fg, font=font)

def _btn(parent, text, cmd, bg='#334155', font=('Segoe UI', 8)):
    return tk.Button(parent, text=text, command=cmd,
                     bg=bg, fg='white', activebackground='#7c3aed',
                     activeforeground='white', font=font,
                     relief='flat', padx=8, pady=3)


# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    EOGApp().mainloop()