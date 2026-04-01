#!/usr/bin/env python3
"""
Fixed Grid Image Remapping - Tkinter GUI
=========================================
Interactive desktop interface for the Fixed Grid Remapping engine.

Layout
------
Left  : Control panel (transform, grid, parameters, interpolation)
Right : Notebook with two tabs
        * Images   - original vs remapped, side-by-side
        * Heatmaps - dX / dY / displacement magnitude colour maps
Bottom: Status bar (timing, grid stats, last action)

Dependencies
------------
  pip install opencv-python-headless numpy scipy Pillow

Run
---
  python gui_app.py                       # defaults: barrel on checkerboard
  python gui_app.py --image path/to.jpg   # load a real image on startup

Author: Balaji R
License: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Dict, Optional, Tuple, Any

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

try:
    from PIL import Image as PILImage
    from PIL import ImageTk
except ImportError:
    raise SystemExit(
        "Pillow is required for the GUI.\n"
        "Install it with:  pip install Pillow --break-system-packages"
    )

from grid_engine import DistortionPresets, GridRemapEngine
from sample_images import SAMPLE_IMAGES


# ---------------------------------------------------------------------------
# Constants / palette
# ---------------------------------------------------------------------------
PAD   = 6

# Dark theme (default)
THEME_DARK = {
    "bg": "#1e1e2e",         # dark base
    "bg2": "#2a2a3e",        # panel bg
    "bg3": "#353550",        # lighter panel for inputs
    "accent": "#7c83fd",     # highlight
    "accent_light": "#a0a8ff",  # lighter accent
    "fg": "#cdd6f4",         # main text
    "fg2": "#a6adc8",        # secondary text
    "ok": "#a6e3a1",
    "err": "#f38ba8",
    "warn": "#fab387",
    "select_bg": "#5b5fd0",  # Selection background
    "select_fg": "#ffffff",  # Selection foreground
    "info_bg": "#0d0d1a",    # Info text background
}

# Light theme
THEME_LIGHT = {
    "bg": "#f0f0f5",         # light base
    "bg2": "#e0e0e8",        # panel bg
    "bg3": "#d0d0db",        # inputs
    "accent": "#4a4fd9",     # highlight
    "accent_light": "#6a6ff0",  # lighter accent
    "fg": "#1a1a2e",         # main text (dark)
    "fg2": "#4a4a5e",        # secondary text
    "ok": "#2d7a2d",
    "err": "#c93535",
    "warn": "#c97800",
    "select_bg": "#4a4fd9",  # Selection background
    "select_fg": "#ffffff",  # Selection foreground
    "info_bg": "#f8f8fc",    # Info text background
}

# Current theme (will be set in __init__)
BG = BG2 = BG3 = ACCENT = ACCENT_LIGHT = FG = FG2 = OK = ERR = WARN = SELECT_BG = SELECT_FG = ""

FONT_BODY  = ("Segoe UI", 9)
FONT_BOLD  = ("Segoe UI", 9, "bold")
FONT_TITLE = ("Segoe UI", 10, "bold")
FONT_MONO  = ("Consolas", 9)

FONT_BODY  = ("Segoe UI", 9)
FONT_BOLD  = ("Segoe UI", 9, "bold")
FONT_TITLE = ("Segoe UI", 10, "bold")
FONT_MONO  = ("Consolas", 9)

DEFAULTS_FILE = "def_values_gui.json"

# Transform -> which parameter sliders are relevant
TRANSFORM_PARAMS: Dict[str, list[str]] = {
    "barrel":     ["k1"],
    "pincushion": ["k1"],
    "swirl":      ["strength"],
    "wave":       ["amplitude"],
    "fisheye":    [],
    "correction": ["k1"],
    "identity":   [],
}

ALL_PARAMS = ["k1", "strength", "amplitude"]

# Transform application order shared by the sequential pipeline, _apply_post_process,
# and _apply_geometric_fold.  Both methods MUST reference this constant so that a
# change here propagates everywhere and never silently diverges.
# Composition is NON-COMMUTATIVE: Flip→Rotate ≠ Rotate→Flip.
_FOLD_ORDER: Tuple[str, str] = ("flip", "rotate")


# ---------------------------------------------------------------------------
# Helper: numpy BGR array -> PIL PhotoImage (for Tkinter canvas)
# ---------------------------------------------------------------------------
def _bgr_to_photoimage(
    bgr: np.ndarray, max_w: int, max_h: int
) -> Tuple[ImageTk.PhotoImage, int, int]:
    """
    Convert a BGR uint8 ndarray to a Tkinter PhotoImage, scaled to fit
    within (max_w x max_h) while preserving aspect ratio.

    Returns
    -------
    photo, display_width, display_height
    """
    h, w = bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = PILImage.fromarray(rgb)
    return ImageTk.PhotoImage(pil), pil.width, pil.height


def _float_to_heatmap(grid: np.ndarray) -> np.ndarray:
    """
    Normalise a float32 2D array to uint8 and apply COLORMAP_RdBu-like colouring.
    Returns BGR uint8.
    """
    mn, mx = grid.min(), grid.max()
    if abs(mx - mn) < 1e-9:
        norm = np.full_like(grid, 128, dtype=np.uint8)
    else:
        norm = ((grid - mn) / (mx - mn) * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_COOL)


# ---------------------------------------------------------------------------
# Labelled scale (slider + numeric readout, in one row)
# ---------------------------------------------------------------------------
class LabelledScale(ttk.Frame):
    """A horizontal Scale widget with a value label on the right."""

    def __init__(
        self,
        parent,
        label: str,
        from_: float,
        to: float,
        resolution: float,
        initial: float,
        on_change=None,
        **kw,
    ):
        super().__init__(parent, **kw)
        self._cb = on_change

        ttk.Label(self, text=label, width=10, anchor="w",
                  font=FONT_BODY).pack(side=tk.LEFT)

        self._var = tk.DoubleVar(value=initial)
        self._scale = ttk.Scale(
            self, orient=tk.HORIZONTAL, from_=from_, to=to,
            variable=self._var, command=self._on_scale,
        )
        self._scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))

        self._lbl = ttk.Label(self, text=f"{initial:.2f}", width=6,
                              font=FONT_MONO, anchor="e")
        self._lbl.pack(side=tk.LEFT)

    def _on_scale(self, _=None):
        val = round(self._var.get(), 4)
        self._lbl.config(text=f"{val:.2f}")
        if self._cb:
            self._cb()

    @property
    def value(self) -> float:
        return self._var.get()

    def set(self, v: float):
        self._var.set(v)
        self._lbl.config(text=f"{v:.2f}")


# ---------------------------------------------------------------------------
# Image canvas with label overlay
# ---------------------------------------------------------------------------
class ImageCanvas(ttk.Frame):
    """A labelled canvas that displays a BGR ndarray."""

    def __init__(self, parent, title: str, **kw):
        super().__init__(parent, **kw)
        ttk.Label(self, text=title, font=FONT_BOLD, anchor="center").pack(
            fill=tk.X, padx=2, pady=(2, 0)
        )
        self._canvas = tk.Canvas(self, bg="#0d0d1a", highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._img_id: Optional[int] = None

        self._canvas.bind("<Configure>", self._on_resize)
        self._pending_bgr: Optional[np.ndarray] = None

    def show(self, bgr: np.ndarray):
        """Display a BGR ndarray, auto-scaling to canvas size."""
        self._pending_bgr = bgr
        self._render()

    def _on_resize(self, _=None):
        if self._pending_bgr is not None:
            self._render()

    def _render(self):
        if self._pending_bgr is None:
            return
        cw = max(self._canvas.winfo_width(), 100)
        ch = max(self._canvas.winfo_height(), 80)
        photo, pw, ph = _bgr_to_photoimage(self._pending_bgr, cw - 4, ch - 4)
        self._photo = photo           # keep reference - prevents GC
        if self._img_id is None:
            self._img_id = self._canvas.create_image(
                cw // 2, ch // 2, anchor=tk.CENTER, image=photo
            )
        else:
            self._canvas.coords(self._img_id, cw // 2, ch // 2)
            self._canvas.itemconfig(self._img_id, image=photo)


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------
class RemapGUI(tk.Tk):

    def __init__(self, startup_image_path: Optional[str] = None):
        super().__init__()

        # Initialize theme (load from defaults or use dark)
        self._theme_var = tk.StringVar(value="dark")
        self._apply_theme_colors(self._theme_var.get())

        self.title("Fixed Grid Image Remapping")
        self.geometry("1280x760")
        self.minsize(900, 600)
        self.configure(bg=BG)

        # Start in maximized (fullscreen) mode
        self.state("zoomed")

        # Apply a dark ttk theme
        self._apply_theme()

        # --- State ----------------------------------------------------------
        self._source_bgr: Optional[np.ndarray] = None    # current input image
        self._result_bgr: Optional[np.ndarray] = None    # last remapped result
        self._overlay_bgr: Optional[np.ndarray] = None   # result with grid lines
        self._remapped_bgr: Optional[np.ndarray] = None  # before flip/rotate
        self._transformed_bgr: Optional[np.ndarray] = None  # after flip/rotate
        self._dx_grid: Optional[np.ndarray] = None
        self._dy_grid: Optional[np.ndarray] = None
        self._processing = False

        # Comparison tab state
        self._cmp_result_a: Optional[np.ndarray] = None   # Method A final output
        self._cmp_result_b: Optional[np.ndarray] = None   # Method B grid-folded output
        self._cmp_diff: Optional[np.ndarray] = None       # amplified residual
        self._cmp_timer: Optional[str] = None             # comparison debounce timer
        self._live_timer: Optional[str] = None           # after() handle

        # Sparse vs Dense tab state
        self._spd_result_dense:  Optional[np.ndarray] = None
        self._spd_result_sparse: Optional[np.ndarray] = None
        self._spd_diff:          Optional[np.ndarray] = None
        self._spd_timer:         Optional[str] = None

        # --- Load defaults from JSON ----------------------------------------
        self._defaults = self._load_defaults_from_json()

        # --- Build UI -------------------------------------------------------
        self._build_layout()

        # Apply loaded defaults to UI controls
        self._apply_loaded_defaults()

        # Load startup image if provided
        if startup_image_path:
            self._load_image_file(startup_image_path)
        else:
            self._generate_sample()

        self._update_param_visibility()

        # Bind window close event to save defaults
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _apply_loaded_defaults(self):
        """Apply loaded defaults from JSON to UI variables."""
        if self._defaults is None:
            return
        try:
            # Input settings
            if "input" in self._defaults:
                inp = self._defaults["input"]
                if "sample_pattern" in inp:
                    self._sample_var.set(inp["sample_pattern"])
                if "width" in inp:
                    self._width_var.set(inp["width"])
                if "height" in inp:
                    self._height_var.set(inp["height"])

            # Transform settings
            if "transform" in self._defaults:
                tf = self._defaults["transform"]
                if "type" in tf:
                    self._transform_var.set(tf["type"])

            # Grid settings
            if "grid" in self._defaults:
                grid = self._defaults["grid"]
                if "rows" in grid:
                    self._grid_rows_var.set(grid["rows"])
                if "cols" in grid:
                    self._grid_cols_var.set(grid["cols"])

            # Parameters (sliders)
            if "parameters" in self._defaults:
                params = self._defaults["parameters"]
                for name in ["k1", "strength", "amplitude"]:
                    if name in params and name in self._sliders:
                        if "value" in params[name]:
                            self._sliders[name].set(params[name]["value"])

            # Interpolation settings
            if "interpolation" in self._defaults:
                interp = self._defaults["interpolation"]
                if "pixel_interp" in interp:
                    self._interp_var.set(interp["pixel_interp"])
                if "grid_interp" in interp:
                    self._grid_interp_var.set(interp["grid_interp"])
                if "border_mode" in interp:
                    self._border_var.set(interp["border_mode"])

            # Display options
            if "display" in self._defaults:
                disp = self._defaults["display"]
                if "show_grid_overlay" in disp:
                    self._overlay_var.set(disp["show_grid_overlay"])
                if "grid_viz_type" in disp:
                    self._grid_viz_var.set(disp["grid_viz_type"])
                if "live_preview" in disp:
                    self._live_var.set(disp["live_preview"])
                if "compute_heatmaps" in disp:
                    self._show_heatmap_var.set(disp["compute_heatmaps"])

            # ROI settings
            if "roi" in self._defaults:
                roi = self._defaults["roi"]
                if "draw" in roi:
                    self._roi_draw_var.set(roi["draw"])
                if "crop" in roi:
                    self._roi_crop_var.set(roi["crop"])
                if "x" in roi:
                    self._roi_x_var.set(roi["x"])
                if "y" in roi:
                    self._roi_y_var.set(roi["y"])
                if "w" in roi:
                    self._roi_w_var.set(roi["w"])
                if "h" in roi:
                    self._roi_h_var.set(roi["h"])

            # Flip & Rotate settings
            if "flip_rotate" in self._defaults:
                fr = self._defaults["flip_rotate"]
                if "flip_h" in fr:
                    self._flip_h_var.set(fr["flip_h"])
                if "flip_v" in fr:
                    self._flip_v_var.set(fr["flip_v"])
                if "rotate" in fr:
                    self._rotate_var.set(fr["rotate"])

            # Theme settings
            if "theme" in self._defaults:
                theme = self._defaults["theme"]
                if "name" in theme:
                    loaded_theme = theme["name"]
                    if loaded_theme != self._theme_var.get():
                        self._theme_var.set(loaded_theme)
                        self._apply_theme_colors(loaded_theme)
        except Exception as e:
            print(f"Warning: Error applying loaded defaults: {e}")

    def _on_closing(self):
        """Save defaults to JSON and close the application."""
        self._save_defaults_to_json()
        self.destroy()

    # ======================================================================
    # Theme
    # ======================================================================
    def _apply_theme(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        # Base configuration
        style.configure(".",          background=BG,  foreground=FG,
                        fieldbackground=BG2, font=FONT_BODY)
        style.configure("TFrame",     background=BG)
        style.configure("TLabel",     background=BG,  foreground=FG)
        style.configure("TLabelframe",background=BG,  foreground=ACCENT,
                        bordercolor=ACCENT)
        style.configure("TLabelframe.Label", background=BG, foreground=ACCENT,
                        font=FONT_TITLE)
        
        # Combobox - with proper selection colors
        style.configure("TCombobox",  fieldbackground=BG3, foreground=FG,
                        selectbackground=SELECT_BG, selectforeground=SELECT_FG)
        style.map("TCombobox", 
                  fieldbackground=[("readonly", BG3), ("active", BG3)],
                  selectbackground=[("readonly", SELECT_BG)],
                  selectforeground=[("readonly", SELECT_FG)])
        
        # Spinbox - with proper selection colors
        style.configure("TSpinbox",   fieldbackground=BG3, foreground=FG,
                        selectbackground=SELECT_BG, selectforeground=SELECT_FG)
        
        # Checkbutton
        style.configure("TCheckbutton", background=BG, foreground=FG)
        
        # Scale/Slider
        style.configure("TScale",     background=BG,  troughcolor=BG2)
        
        # Notebook tabs
        style.configure("TNotebook",  background=BG,  tabmargins=0)
        style.configure("TNotebook.Tab", background=BG2, foreground=FG2,
                        padding=[10, 4])
        style.map("TNotebook.Tab",    background=[("selected", BG)],
                  foreground=[("selected", ACCENT)])

        # Scrollbar styling
        style.configure("TScrollbar", background=BG2, troughcolor=BG,
                        bordercolor=BG, arrowcolor=FG)
        style.map("TScrollbar", background=[("active", BG3), ("pressed", ACCENT)])

        # Progress bar
        style.configure("Horizontal.TProgressbar", background=ACCENT, 
                        troughcolor=BG2, bordercolor=BG)

        # Custom button styles
        style.configure("Apply.TButton", background=ACCENT, foreground="#1e1e2e",
                        font=FONT_BOLD, padding=[8, 4])
        style.map("Apply.TButton", 
                  background=[("active", ACCENT_LIGHT), ("pressed", ACCENT)])
        
        style.configure("Save.TButton",  background="#45475a", foreground=FG,
                        font=FONT_BODY,  padding=[6, 4])
        style.map("Save.TButton",
                  background=[("active", "#5a5f7a"), ("pressed", "#353a55")])
        
        style.configure("Warn.TButton",  background=WARN, foreground="#1e1e2e",
                        font=FONT_BODY,  padding=[6, 4])
        style.map("Warn.TButton",
                  background=[("active", "#ffcfa6"), ("pressed", WARN)])

    # ======================================================================
    # Layout construction
    # ======================================================================
    def _build_layout(self):
        # Top-level paned window: controls | view
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # --- Left: control panel --------------------------------------------
        ctrl_outer = ttk.Frame(paned, width=300)
        ctrl_outer.pack_propagate(False)
        paned.add(ctrl_outer, weight=0)
        self._build_control_panel(ctrl_outer)

        # --- Right: view panel ----------------------------------------------
        view = ttk.Frame(paned)
        paned.add(view, weight=1)
        self._build_view_panel(view)

        # --- Bottom: status bar ---------------------------------------------
        self._build_status_bar()

    # ------------------------------------------------------------------
    # Control panel
    # ------------------------------------------------------------------
    def _build_control_panel(self, parent: ttk.Frame):
        canvas = tk.Canvas(parent, bg=BG2, highlightthickness=0)
        scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=inner, anchor=tk.NW)

        def _on_frame_configure(_):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(e):
            canvas.itemconfig(win_id, width=e.width)

        inner.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        # Mouse wheel scroll
        def _on_mousewheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        p = PAD
        row = 0

        # --- Section: Input -------------------------------------------------
        inp_frame = ttk.LabelFrame(inner, text="  Input Image  ")
        inp_frame.grid(row=row, column=0, sticky="ew", padx=p, pady=(p, 2))
        inner.columnconfigure(0, weight=1)
        row += 1

        # Sample pattern
        ttk.Label(inp_frame, text="Sample pattern").grid(
            row=0, column=0, sticky="w", padx=p, pady=2)
        self._sample_var = tk.StringVar(value="checkerboard")
        sample_cb = ttk.Combobox(
            inp_frame, textvariable=self._sample_var,
            values=list(SAMPLE_IMAGES.keys()), state="readonly", width=18,
        )
        sample_cb.grid(row=0, column=1, sticky="ew", padx=p, pady=2)
        sample_cb.bind("<<ComboboxSelected>>", lambda _: self._generate_sample())

        # Image size
        size_row = ttk.Frame(inp_frame)
        size_row.grid(row=1, column=0, columnspan=2, sticky="ew", padx=p, pady=2)
        ttk.Label(size_row, text="Size").pack(side=tk.LEFT)
        self._width_var = tk.IntVar(value=640)
        self._height_var = tk.IntVar(value=480)
        ttk.Spinbox(size_row, textvariable=self._width_var,
                    from_=64, to=1920, increment=64, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(size_row, text="x").pack(side=tk.LEFT)
        ttk.Spinbox(size_row, textvariable=self._height_var,
                    from_=64, to=1080, increment=64, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(size_row, text="Regenerate",
                   command=self._generate_sample,
                   style="Save.TButton").pack(side=tk.LEFT, padx=4)

        # Load file
        file_row = ttk.Frame(inp_frame)
        file_row.grid(row=2, column=0, columnspan=2, sticky="ew", padx=p, pady=(2, p))
        ttk.Button(file_row, text="Load Image File",
                   command=self._browse_image,
                   style="Save.TButton").pack(fill=tk.X, expand=True)

        inp_frame.columnconfigure(1, weight=1)

        # --- Section: Transform ---------------------------------------------
        tf_frame = ttk.LabelFrame(inner, text="  Transform  ")
        tf_frame.grid(row=row, column=0, sticky="ew", padx=p, pady=2)
        row += 1

        self._transform_var = tk.StringVar(value="barrel")
        tf_cb = ttk.Combobox(
            tf_frame, textvariable=self._transform_var,
            values=list(TRANSFORM_PARAMS.keys()), state="readonly",
        )
        tf_cb.pack(fill=tk.X, padx=p, pady=p)
        tf_cb.bind("<<ComboboxSelected>>", self._on_transform_changed)

        # --- Section: Grid --------------------------------------------------
        grid_frame = ttk.LabelFrame(inner, text="  Sparse Grid  ")
        grid_frame.grid(row=row, column=0, sticky="ew", padx=p, pady=2)
        row += 1

        gr = ttk.Frame(grid_frame)
        gr.pack(fill=tk.X, padx=p, pady=p)
        ttk.Label(gr, text="Rows").pack(side=tk.LEFT)
        self._grid_rows_var = tk.IntVar(value=33)
        ttk.Spinbox(gr, textvariable=self._grid_rows_var,
                    from_=3, to=33, increment=2, width=4,
                    command=self._schedule_live).pack(side=tk.LEFT, padx=4)
        ttk.Label(gr, text="Cols").pack(side=tk.LEFT)
        self._grid_cols_var = tk.IntVar(value=33)
        ttk.Spinbox(gr, textvariable=self._grid_cols_var,
                    from_=3, to=33, increment=2, width=4,
                    command=self._schedule_live).pack(side=tk.LEFT, padx=4)

        # --- Section: Parameters --------------------------------------------
        self._param_frame = ttk.LabelFrame(inner, text="  Parameters  ")
        self._param_frame.grid(row=row, column=0, sticky="ew", padx=p, pady=2)
        row += 1

        on_change = self._schedule_live

        self._sliders: Dict[str, LabelledScale] = {}

        self._sliders["k1"] = LabelledScale(
            self._param_frame, "k1 (radial)",
            from_=-1.0, to=1.0, resolution=0.01, initial=0.4,
            on_change=on_change,
        )
        self._sliders["k1"].pack(fill=tk.X, padx=p, pady=2)

        self._sliders["strength"] = LabelledScale(
            self._param_frame, "strength",
            from_=0.1, to=5.0, resolution=0.1, initial=2.5,
            on_change=on_change,
        )
        self._sliders["strength"].pack(fill=tk.X, padx=p, pady=2)

        self._sliders["amplitude"] = LabelledScale(
            self._param_frame, "amplitude",
            from_=1.0, to=60.0, resolution=1.0, initial=15.0,
            on_change=on_change,
        )
        self._sliders["amplitude"].pack(fill=tk.X, padx=p, pady=2)

        # --- Section: Interpolation -------------------------------------------
        interp_frame = ttk.LabelFrame(inner, text="  Interpolation & Border  ")
        interp_frame.grid(row=row, column=0, sticky="ew", padx=p, pady=2)
        row += 1

        rows_i = [
            ("Pixel interp",  "_interp_var",
             ["nearest", "linear", "cubic", "lanczos"], "cubic"),
            ("Grid interp",   "_grid_interp_var",
             ["bicubic", "linear"], "bicubic"),
            ("Border mode",   "_border_var",
             ["constant", "replicate", "reflect", "wrap"], "constant"),
        ]
        for lbl, attr, vals, default in rows_i:
            r = ttk.Frame(interp_frame)
            r.pack(fill=tk.X, padx=p, pady=2)
            ttk.Label(r, text=lbl, width=12, anchor="w").pack(side=tk.LEFT)
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            cb = ttk.Combobox(r, textvariable=var, values=vals,
                              state="readonly", width=12)
            cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
            cb.bind("<<ComboboxSelected>>", lambda _: self._schedule_live())

        # --- Section: Display options ---------------------------------------
        disp_frame = ttk.LabelFrame(inner, text="  Display Options  ")
        disp_frame.grid(row=row, column=0, sticky="ew", padx=p, pady=2)
        row += 1

        self._overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(disp_frame, text="Show grid overlay on output",
                        variable=self._overlay_var,
                        command=self._schedule_live).pack(
            anchor="w", padx=p, pady=2)

        # Grid visualization type
        viz_row = ttk.Frame(disp_frame)
        viz_row.pack(fill=tk.X, padx=p, pady=2)
        ttk.Label(viz_row, text="Grid viz type", width=12, anchor="w").pack(side=tk.LEFT)
        self._grid_viz_var = tk.StringVar(value="lines")
        viz_cb = ttk.Combobox(viz_row, textvariable=self._grid_viz_var,
                              values=["lines", "dots"], state="readonly", width=12)
        viz_cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
        viz_cb.bind("<<ComboboxSelected>>", lambda _: self._schedule_live())

        self._live_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(disp_frame, text="Live preview (auto-apply)",
                        variable=self._live_var).pack(
            anchor="w", padx=p, pady=2)

        self._show_heatmap_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(disp_frame, text="Compute heatmaps",
                        variable=self._show_heatmap_var,
                        command=self._schedule_live).pack(
            anchor="w", padx=p, pady=2)

        # --- Section: ROI (Crop) --------------------------------------------
        roi_frame = ttk.LabelFrame(inner, text="  Region of Interest (ROI)  ")
        roi_frame.grid(row=row, column=0, sticky="ew", padx=p, pady=2)
        row += 1

        # ROI Enable/Draw checkboxes
        roi_chk_row = ttk.Frame(roi_frame)
        roi_chk_row.pack(fill=tk.X, padx=p, pady=2)

        self._roi_draw_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(roi_chk_row, text="Draw ROI",
                        variable=self._roi_draw_var,
                        command=self._schedule_live).pack(side=tk.LEFT, padx=(0, 10))

        self._roi_crop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(roi_chk_row, text="Crop to ROI",
                        variable=self._roi_crop_var,
                        command=self._schedule_live).pack(side=tk.LEFT)

        # ROI entry fields (direct pixel values)
        roi_grid = ttk.Frame(roi_frame)
        roi_grid.pack(fill=tk.X, padx=p, pady=4)

        # X position
        ttk.Label(roi_grid, text="X:", width=3).grid(row=0, column=0, sticky="w")
        self._roi_x_var = tk.IntVar(value=100)
        self._roi_x_entry = ttk.Spinbox(roi_grid, textvariable=self._roi_x_var,
                                        from_=0, to=9999, increment=10, width=6,
                                        command=self._schedule_live)
        self._roi_x_entry.grid(row=0, column=1, padx=(0, 8))
        self._roi_x_entry.bind("<Return>", lambda _: self._schedule_live())

        # Y position
        ttk.Label(roi_grid, text="Y:", width=3).grid(row=0, column=2, sticky="w")
        self._roi_y_var = tk.IntVar(value=100)
        self._roi_y_entry = ttk.Spinbox(roi_grid, textvariable=self._roi_y_var,
                                        from_=0, to=9999, increment=10, width=6,
                                        command=self._schedule_live)
        self._roi_y_entry.grid(row=0, column=3)
        self._roi_y_entry.bind("<Return>", lambda _: self._schedule_live())

        # Width
        ttk.Label(roi_grid, text="W:", width=3).grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._roi_w_var = tk.IntVar(value=320)
        self._roi_w_entry = ttk.Spinbox(roi_grid, textvariable=self._roi_w_var,
                                        from_=10, to=9999, increment=10, width=6,
                                        command=self._schedule_live)
        self._roi_w_entry.grid(row=1, column=1, padx=(0, 8), pady=(4, 0))
        self._roi_w_entry.bind("<Return>", lambda _: self._schedule_live())

        # Height
        ttk.Label(roi_grid, text="H:", width=3).grid(row=1, column=2, sticky="w", pady=(4, 0))
        self._roi_h_var = tk.IntVar(value=240)
        self._roi_h_entry = ttk.Spinbox(roi_grid, textvariable=self._roi_h_var,
                                        from_=10, to=9999, increment=10, width=6,
                                        command=self._schedule_live)
        self._roi_h_entry.grid(row=1, column=3, pady=(4, 0))
        self._roi_h_entry.bind("<Return>", lambda _: self._schedule_live())

        ttk.Button(roi_frame, text="Reset ROI",
                   command=self._reset_roi,
                   style="Save.TButton").pack(fill=tk.X, padx=p, pady=4)

        # --- Section: Flip & Rotate -----------------------------------------
        flip_frame = ttk.LabelFrame(inner, text="  Flip & Rotate  ")
        flip_frame.grid(row=row, column=0, sticky="ew", padx=p, pady=2)
        row += 1

        # Flip options
        flip_row = ttk.Frame(flip_frame)
        flip_row.pack(fill=tk.X, padx=p, pady=2)

        self._flip_h_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(flip_row, text="Flip H",
                        variable=self._flip_h_var,
                        command=self._on_flip_rotate_changed).pack(side=tk.LEFT, padx=(0, 8))

        self._flip_v_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(flip_row, text="Flip V",
                        variable=self._flip_v_var,
                        command=self._on_flip_rotate_changed).pack(side=tk.LEFT)

        # Rotation
        rot_row = ttk.Frame(flip_frame)
        rot_row.pack(fill=tk.X, padx=p, pady=2)
        ttk.Label(rot_row, text="Rotate", width=8, anchor="w").pack(side=tk.LEFT)
        self._rotate_var = tk.IntVar(value=0)
        rot_cb = ttk.Combobox(rot_row, textvariable=self._rotate_var,
                              values=[0, 90, 180, 270], state="readonly", width=8)
        rot_cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
        rot_cb.bind("<<ComboboxSelected>>", lambda _: self._on_flip_rotate_changed())

        # --- Section: Actions -----------------------------------------------
        act_frame = ttk.Frame(inner)
        act_frame.grid(row=row, column=0, sticky="ew", padx=p, pady=(p, 2))
        row += 1

        ttk.Button(act_frame, text="Apply Remap",
                   command=self._apply,
                   style="Apply.TButton").pack(fill=tk.X, pady=2)

        ttk.Button(act_frame, text="Save Remapped Image",
                   command=self._save_image,
                   style="Save.TButton").pack(fill=tk.X, pady=2)

        ttk.Button(act_frame, text="Save Comparison Figure",
                   command=self._save_figure,
                   style="Save.TButton").pack(fill=tk.X, pady=2)

        ttk.Button(act_frame, text="Export All Presets",
                   command=self._export_all,
                   style="Warn.TButton").pack(fill=tk.X, pady=(4, 2))

    # ------------------------------------------------------------------
    # View panel (notebook)
    # ------------------------------------------------------------------
    def _build_view_panel(self, parent: ttk.Frame):
        self._notebook = ttk.Notebook(parent)
        self._notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Images
        tab_img = ttk.Frame(self._notebook)
        self._notebook.add(tab_img, text="  Images  ")
        self._build_images_tab(tab_img)

        # Tab 2: Transform (Remapped vs Transformed)
        tab_transform = ttk.Frame(self._notebook)
        self._notebook.add(tab_transform, text="  Transform  ")
        self._build_transform_tab(tab_transform)

        # Tab 3: Heatmaps
        tab_heat = ttk.Frame(self._notebook)
        self._notebook.add(tab_heat, text="  Heatmaps  ")
        self._build_heatmaps_tab(tab_heat)

        # Tab 4: Info
        tab_info = ttk.Frame(self._notebook)
        self._notebook.add(tab_info, text="  Info  ")
        self._build_info_tab(tab_info)

        # Tab 5: Comparison (Method A vs Method B grid-folded)
        tab_cmp = ttk.Frame(self._notebook)
        self._notebook.add(tab_cmp, text="  Comparison  ")
        self._build_comparison_tab(tab_cmp)

        # Tab 6: Sparse vs Dense — naive upsample vs bicubic spline reconstruction
        tab_spd = ttk.Frame(self._notebook)
        self._notebook.add(tab_spd, text="  Sparse vs Dense  ")
        self._build_sparse_vs_dense_tab(tab_spd)

        # Bind tab change to auto-run comparison when tab is selected
        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _build_images_tab(self, parent: ttk.Frame):
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        self._canvas_orig = ImageCanvas(parent, "Original")
        self._canvas_orig.grid(row=0, column=0, sticky="nsew", padx=(4, 2), pady=4)

        self._canvas_out = ImageCanvas(parent, "Remapped")
        self._canvas_out.grid(row=0, column=1, sticky="nsew", padx=(2, 4), pady=4)

    def _build_transform_tab(self, parent: ttk.Frame):
        """Tab showing Remapped (before flip/rotate) vs Transformed (after)."""
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        self._canvas_remapped = ImageCanvas(parent, "Remapped (Before Flip/Rotate)")
        self._canvas_remapped.grid(row=0, column=0, sticky="nsew", padx=(4, 2), pady=4)

        self._canvas_transformed = ImageCanvas(parent, "Transformed (After Flip/Rotate)")
        self._canvas_transformed.grid(row=0, column=1, sticky="nsew", padx=(2, 4), pady=4)

    def _build_heatmaps_tab(self, parent: ttk.Frame):
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(2, weight=1)
        parent.rowconfigure(0, weight=1)

        self._canvas_dx  = ImageCanvas(parent, "dX  (horizontal displacement)")
        self._canvas_dx.grid(row=0, column=0, sticky="nsew", padx=(4, 2), pady=4)

        self._canvas_dy  = ImageCanvas(parent, "dY  (vertical displacement)")
        self._canvas_dy.grid(row=0, column=1, sticky="nsew", padx=2, pady=4)

        self._canvas_mag = ImageCanvas(parent, "Displacement Magnitude")
        self._canvas_mag.grid(row=0, column=2, sticky="nsew", padx=(2, 4), pady=4)

    def _build_info_tab(self, parent: ttk.Frame):
        info_bg = THEME_DARK["info_bg"] if self._theme_var.get() == "dark" else THEME_LIGHT["info_bg"]
        self._info_text = tk.Text(
            parent, bg=info_bg, fg=FG, font=FONT_MONO,
            relief=tk.FLAT, state=tk.DISABLED, wrap=tk.WORD,
            selectbackground=SELECT_BG, selectforeground=SELECT_FG,
            inactiveselectbackground=SELECT_BG,
        )
        sb = ttk.Scrollbar(parent, command=self._info_text.yview)
        self._info_text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._info_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------
    def _build_status_bar(self):
        bar = ttk.Frame(self, style="TFrame")
        bar.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=(0, 4))

        self._status_var = tk.StringVar(value="Ready.")
        self._status_lbl = ttk.Label(
            bar, textvariable=self._status_var,
            font=FONT_MONO, foreground=FG2, anchor="w",
        )
        self._status_lbl.pack(side=tk.LEFT, padx=4)

        # Theme toggle button
        current_theme = self._theme_var.get()
        theme_btn = ttk.Button(
            bar, text="☀️ Light" if current_theme == "dark" else "🌙 Dark",
            width=10, command=self._toggle_theme
        )
        theme_btn.pack(side=tk.RIGHT, padx=(0, 4))
        self._theme_btn = theme_btn

        self._progress = ttk.Progressbar(
            bar, mode="indeterminate", length=120,
        )
        self._progress.pack(side=tk.RIGHT, padx=4)

    # ======================================================================
    # UI helpers
    # ======================================================================
    def _set_status(self, msg: str, colour: str = FG2):
        self._status_var.set(msg)
        self._status_lbl.config(foreground=colour)

    def _apply_theme_colors(self, theme: str):
        """Apply theme colors to global constants."""
        global BG, BG2, BG3, ACCENT, ACCENT_LIGHT, FG, FG2, OK, ERR, WARN, SELECT_BG, SELECT_FG
        theme_dict = THEME_LIGHT if theme == "light" else THEME_DARK
        BG = theme_dict["bg"]
        BG2 = theme_dict["bg2"]
        BG3 = theme_dict["bg3"]
        ACCENT = theme_dict["accent"]
        ACCENT_LIGHT = theme_dict["accent_light"]
        FG = theme_dict["fg"]
        FG2 = theme_dict["fg2"]
        OK = theme_dict["ok"]
        ERR = theme_dict["err"]
        WARN = theme_dict["warn"]
        SELECT_BG = theme_dict["select_bg"]
        SELECT_FG = theme_dict["select_fg"]

    def _toggle_theme(self):
        """Toggle between light and dark themes."""
        current = self._theme_var.get()
        new_theme = "light" if current == "dark" else "dark"
        self._theme_var.set(new_theme)

        # Apply new theme colors
        self._apply_theme_colors(new_theme)

        # Update theme button text
        self._theme_btn.config(text="☀️ Light" if new_theme == "dark" else "🌙 Dark")

        # Reapply styles
        self._apply_theme()

        # Update widget backgrounds
        self.configure(bg=BG)

        # Update info text widget
        info_bg = THEME_LIGHT["info_bg"] if new_theme == "light" else THEME_DARK["info_bg"]
        self._info_text.config(bg=info_bg, fg=FG)

        # Update canvases
        canvas_bg = "#0d0d1a" if new_theme == "dark" else "#f0f0f5"
        self._canvas_orig._canvas.config(bg=canvas_bg)
        self._canvas_out._canvas.config(bg=canvas_bg)
        self._canvas_remapped._canvas.config(bg=canvas_bg)
        self._canvas_transformed._canvas.config(bg=canvas_bg)
        self._canvas_dx._canvas.config(bg=canvas_bg)
        self._canvas_dy._canvas.config(bg=canvas_bg)
        self._canvas_mag._canvas.config(bg=canvas_bg)
        # Comparison tab canvases
        self._canvas_cmp_a._canvas.config(bg=canvas_bg)
        self._canvas_cmp_b._canvas.config(bg=canvas_bg)
        self._canvas_diff._canvas.config(bg=canvas_bg)
        # Sparse vs Dense tab canvases
        self._canvas_spd_dense._canvas.config(bg=canvas_bg)
        self._canvas_spd_sparse._canvas.config(bg=canvas_bg)
        self._canvas_spd_diff._canvas.config(bg=canvas_bg)

        # Update status label
        self._set_status(f"Theme switched to {new_theme}", OK)

    def _load_defaults_from_json(self) -> Optional[Dict[str, Any]]:
        """Load default GUI values from JSON file if it exists."""
        try:
            if os.path.exists(DEFAULTS_FILE):
                with open(DEFAULTS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load defaults from {DEFAULTS_FILE}: {e}")
        return None

    def _save_defaults_to_json(self):
        """Save all default GUI values to a JSON file."""
        defaults: Dict[str, Any] = {
            "window": {
                "title": "Fixed Grid Image Remapping",
                "geometry": "1280x760",
                "minsize": [900, 600]
            },
            "input": {
                "sample_pattern": self._sample_var.get(),
                "width": self._width_var.get(),
                "height": self._height_var.get()
            },
            "transform": {
                "type": self._transform_var.get(),
                "available_transforms": list(TRANSFORM_PARAMS.keys())
            },
            "grid": {
                "rows": self._grid_rows_var.get(),
                "cols": self._grid_cols_var.get()
            },
            "parameters": {
                "k1": {
                    "value": self._sliders["k1"].value,
                    "range": [-1.0, 1.0],
                    "resolution": 0.01
                },
                "strength": {
                    "value": self._sliders["strength"].value,
                    "range": [0.1, 5.0],
                    "resolution": 0.1
                },
                "amplitude": {
                    "value": self._sliders["amplitude"].value,
                    "range": [1.0, 60.0],
                    "resolution": 1.0
                }
            },
            "interpolation": {
                "pixel_interp": self._interp_var.get(),
                "pixel_interp_options": ["nearest", "linear", "cubic", "lanczos"],
                "grid_interp": self._grid_interp_var.get(),
                "grid_interp_options": ["bicubic", "linear"],
                "border_mode": self._border_var.get(),
                "border_mode_options": ["constant", "replicate", "reflect", "wrap"]
            },
            "display": {
                "show_grid_overlay": self._overlay_var.get(),
                "grid_viz_type": self._grid_viz_var.get(),
                "grid_viz_type_options": ["lines", "dots"],
                "live_preview": self._live_var.get(),
                "compute_heatmaps": self._show_heatmap_var.get()
            },
            "roi": {
                "draw": self._roi_draw_var.get(),
                "crop": self._roi_crop_var.get(),
                "x": self._roi_x_var.get(),
                "y": self._roi_y_var.get(),
                "w": self._roi_w_var.get(),
                "h": self._roi_h_var.get(),
            },
            "flip_rotate": {
                "flip_h": self._flip_h_var.get(),
                "flip_v": self._flip_v_var.get(),
                "rotate": self._rotate_var.get(),
                "rotate_options": [0, 90, 180, 270],
            },
            "theme": {
                "name": self._theme_var.get(),
                "background": BG,
                "background_secondary": BG2,
                "background_input": BG3,
                "accent": ACCENT,
                "accent_light": ACCENT_LIGHT,
                "foreground": FG,
                "foreground_secondary": FG2,
                "selection_background": SELECT_BG,
                "selection_foreground": SELECT_FG,
                "success_color": OK,
                "error_color": ERR,
                "warning_color": WARN
            },
            "transform_params_map": TRANSFORM_PARAMS
        }
        
        try:
            with open(DEFAULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(defaults, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save defaults to {DEFAULTS_FILE}: {e}")

    def _start_progress(self):
        self._processing = True
        self._progress.start(10)

    def _stop_progress(self):
        self._processing = False
        self._progress.stop()

    def _update_param_visibility(self):
        """Show only the sliders relevant to the current transform."""
        transform = self._transform_var.get()
        active = TRANSFORM_PARAMS.get(transform, [])
        for name, slider in self._sliders.items():
            if name in active:
                slider.pack(fill=tk.X, padx=PAD, pady=2)
            else:
                slider.pack_forget()

    def _on_transform_changed(self, _=None):
        self._update_param_visibility()
        self._schedule_live()

    def _on_flip_rotate_changed(self, _=None):
        """Called when flip or rotate parameters change."""
        self._schedule_live()
        # Also schedule the active diagnostic tab if it depends on flip/rotate.
        current_tab = self._notebook.select()
        tab_text = self._notebook.tab(current_tab, "text")
        if "Comparison" in tab_text:
            self._schedule_comparison()
        elif "Sparse vs Dense" in tab_text:
            self._schedule_spd_comparison()

    def _reset_roi(self):
        """Reset ROI to default values."""
        self._roi_x_var.set(100)
        self._roi_y_var.set(100)
        self._roi_w_var.set(320)
        self._roi_h_var.set(240)
        self._schedule_live()

    def _update_roi_max_values(self, width: int, height: int):
        """Update ROI entry max values based on image size."""
        self._roi_x_entry.configure(to=width)
        self._roi_y_entry.configure(to=height)
        self._roi_w_entry.configure(to=width)
        self._roi_h_entry.configure(to=height)

    def _schedule_live(self):
        """Debounce live preview: cancel pending timer, restart 300 ms."""
        if self._live_timer:
            self.after_cancel(self._live_timer)
        if self._live_var.get():
            self._live_timer = self.after(300, self._apply)
        # Re-run active diagnostic tab when any pipeline parameter changes.
        current_tab = self._notebook.select()
        tab_text = self._notebook.tab(current_tab, "text")
        if "Comparison" in tab_text:
            self._schedule_comparison()
        elif "Sparse vs Dense" in tab_text:
            self._schedule_spd_comparison()

    # ======================================================================
    # Image loading
    # ======================================================================
    def _generate_sample(self):
        name = self._sample_var.get()
        W    = self._width_var.get()
        H    = self._height_var.get()
        gen  = SAMPLE_IMAGES.get(name)
        if gen is None:
            self._set_status(f"Unknown sample: {name}", ERR)
            return
        self._source_bgr = gen(W, H)
        self._canvas_orig.show(self._source_bgr)
        self._update_roi_max_values(W, H)
        self._set_status(f"Generated '{name}'  {W}x{H}", OK)
        self._schedule_live()

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("All files", "*.*")],
        )
        if path:
            self._load_image_file(path)

    def _load_image_file(self, path: str):
        bgr = cv2.imread(path)
        if bgr is None:
            messagebox.showerror("Load Error", f"Cannot read:\n{path}")
            return
        W = self._width_var.get()
        H = self._height_var.get()
        self._source_bgr = cv2.resize(bgr, (W, H))
        self._canvas_orig.show(self._source_bgr)
        self._update_roi_max_values(W, H)
        self._set_status(f"Loaded '{Path(path).name}'  ->  {W}x{H}", OK)
        self._schedule_live()

    # ======================================================================
    # Core pipeline
    # ======================================================================
    def _apply(self):
        """Kick off remapping in a background thread."""
        if self._processing:
            return
        if self._source_bgr is None:
            self._set_status("No input image loaded.", WARN)
            return

        params = {
            "transform":      self._transform_var.get(),
            "grid_rows":      self._grid_rows_var.get(),
            "grid_cols":      self._grid_cols_var.get(),
            "k1":             self._sliders["k1"].value,
            "strength":       self._sliders["strength"].value,
            "amplitude":      self._sliders["amplitude"].value,
            "interp":         self._interp_var.get(),
            "grid_interp":    self._grid_interp_var.get(),
            "border":         self._border_var.get(),
            "overlay":        self._overlay_var.get(),
            "grid_viz_type":  self._grid_viz_var.get(),
            "heatmap":        self._show_heatmap_var.get(),
            "roi_draw":       self._roi_draw_var.get(),
            "roi_crop":       self._roi_crop_var.get(),
            "roi_x":          self._roi_x_var.get(),
            "roi_y":          self._roi_y_var.get(),
            "roi_w":          self._roi_w_var.get(),
            "roi_h":          self._roi_h_var.get(),
            "flip_h":         self._flip_h_var.get(),
            "flip_v":         self._flip_v_var.get(),
            "rotate":         self._rotate_var.get(),
        }

        self._start_progress()
        self._set_status("Processing ...", ACCENT)

        thread = threading.Thread(
            target=self._run_pipeline_thread,
            args=(self._source_bgr.copy(), params),
            daemon=True,
        )
        thread.start()

    def _run_pipeline_thread(self, image: np.ndarray, params: dict):
        """Background worker - results are sent back via after()."""
        try:
            H, W = image.shape[:2]
            transform   = params["transform"]
            grid_rows   = params["grid_rows"]
            grid_cols   = params["grid_cols"]
            interp      = params["interp"]
            grid_interp = params["grid_interp"]
            border      = params["border"]

            t0 = time.perf_counter()

            # 1. Build sparse grid
            dx_grid, dy_grid = self._build_grid(
                transform, H, W, grid_rows, grid_cols, params
            )

            # 2. Interpolate -> dense maps
            engine = GridRemapEngine()
            map_x, map_y = engine.build_remap_maps(
                (H, W), dx_grid, dy_grid, grid_interp=grid_interp
            )

            # 3. Remap
            result = engine.apply_remap(
                image, map_x, map_y,
                interpolation=interp, border_mode=border,
            )

            # 4. Optional overlay
            overlay = (
                engine.overlay_grid(result, grid_rows, grid_cols, map_x, map_y,
                                   viz_type=params.get("grid_viz_type", "lines"))
                if params["overlay"] else result
            )

            # Store remapped image before flip/rotate for Transform tab
            remapped_before = overlay.copy()

            # 5. Apply Flip and Rotation
            flip_h = params.get("flip_h", False)
            flip_v = params.get("flip_v", False)
            rotate = params.get("rotate", 0)

            if flip_h or flip_v:
                flip_code = None
                if flip_h and flip_v:
                    flip_code = -1  # Flip both
                elif flip_h:
                    flip_code = 1   # Flip horizontal
                else:
                    flip_code = 0   # Flip vertical
                result = cv2.flip(result, flip_code)
                overlay = cv2.flip(overlay, flip_code)

            if rotate != 0:
                if rotate == 90:
                    result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
                    overlay = cv2.rotate(overlay, cv2.ROTATE_90_CLOCKWISE)
                elif rotate == 180:
                    result = cv2.rotate(result, cv2.ROTATE_180)
                    overlay = cv2.rotate(overlay, cv2.ROTATE_180)
                elif rotate == 270:
                    result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    overlay = cv2.rotate(overlay, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Store transformed image after flip/rotate (before ROI)
            transformed_after = overlay.copy()

            # 6. Apply ROI operations
            roi_info = None
            if params.get("roi_draw") or params.get("roi_crop"):
                roi_x = int(params.get("roi_x", 100))
                roi_y = int(params.get("roi_y", 100))
                roi_w = int(params.get("roi_w", 320))
                roi_h = int(params.get("roi_h", 240))

                # Clamp to image bounds
                roi_x = max(0, min(roi_x, W - 1))
                roi_y = max(0, min(roi_y, H - 1))
                roi_w = max(1, min(roi_w, W - roi_x))
                roi_h = max(1, min(roi_h, H - roi_y))

                roi_info = (roi_x, roi_y, roi_w, roi_h)

                # Draw ROI rectangle on overlay
                if params.get("roi_draw") and roi_w > 0 and roi_h > 0:
                    cv2.rectangle(overlay, (roi_x, roi_y),
                                  (roi_x + roi_w, roi_y + roi_h),
                                  (0, 255, 255), 2)  # Cyan color

                # Crop to ROI
                if params.get("roi_crop") and roi_w > 0 and roi_h > 0:
                    overlay = overlay[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
                    result = result[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

            elapsed_ms = (time.perf_counter() - t0) * 1000

            # 5. Heatmaps - use full-resolution (dense) displacement maps
            dx_heat = dy_heat = mag_heat = None
            if params["heatmap"]:
                # Compute dense displacement from remap maps
                y_pixels, x_pixels = np.meshgrid(np.arange(H, dtype=np.float32),
                                                  np.arange(W, dtype=np.float32),
                                                  indexing='ij')
                dense_dx = map_x - x_pixels
                dense_dy = map_y - y_pixels

                dx_heat  = _float_to_heatmap(dense_dx)
                dy_heat  = _float_to_heatmap(dense_dy)
                mag      = np.sqrt(dense_dx**2 + dense_dy**2)
                mag_heat = _float_to_heatmap(mag)

            # Build flip description
            flip_desc = None
            if flip_h and flip_v:
                flip_desc = "Horizontal + Vertical"
            elif flip_h:
                flip_desc = "Horizontal"
            elif flip_v:
                flip_desc = "Vertical"

            info = {
                "transform":     transform,
                "grid":          f"{grid_rows}x{grid_cols}",
                "grid_interp":   grid_interp,
                "remap_interp":  interp,
                "border":        border,
                "size":          f"{W}x{H}",
                "elapsed_ms":    elapsed_ms,
                "dx_min":        float(dx_grid.min()),
                "dx_max":        float(dx_grid.max()),
                "dy_min":        float(dy_grid.min()),
                "dy_max":        float(dy_grid.max()),
                "mag_max":       float(np.sqrt(dx_grid**2 + dy_grid**2).max()),
                "flip":          flip_desc,
                "rotate":        rotate if rotate != 0 else None,
                "roi":           roi_info,
                "dx_grid":       dx_grid,
                "dy_grid":       dy_grid,
            }

            # Schedule UI update on main thread
            self.after(0, self._on_pipeline_done,
                       result, overlay, dx_grid, dy_grid,
                       dx_heat, dy_heat, mag_heat, info,
                       remapped_before, transformed_after)

        except Exception as exc:
            self.after(0, self._on_pipeline_error, str(exc))

    @staticmethod
    def _build_grid(
        transform: str,
        H: int, W: int,
        grid_rows: int, grid_cols: int,
        params: dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dispatch to DistortionPresets - no sys.exit, raises on bad name."""
        dispatch = {
            "barrel":     lambda: DistortionPresets.barrel(
                              (H, W), grid_rows, grid_cols, k1=params["k1"]),
            "pincushion": lambda: DistortionPresets.pincushion(
                              (H, W), grid_rows, grid_cols, k1=params["k1"]),
            "swirl":      lambda: DistortionPresets.swirl(
                              (H, W), grid_rows, grid_cols, strength=params["strength"]),
            "wave":       lambda: DistortionPresets.wave(
                              (H, W), grid_rows, grid_cols,
                              amplitude_x=params["amplitude"],
                              amplitude_y=params["amplitude"] * 0.7),
            "fisheye":    lambda: DistortionPresets.fisheye(
                              (H, W), grid_rows, grid_cols),
            "correction": lambda: DistortionPresets.pinhole_correction(
                              (H, W), grid_rows, grid_cols,
                              k1=params["k1"], k2=params["k1"] * 0.1),
            "identity":   lambda: DistortionPresets.identity(
                              (H, W), grid_rows, grid_cols),
        }
        if transform not in dispatch:
            raise ValueError(f"Unknown transform: '{transform}'")
        return dispatch[transform]()

    def _on_pipeline_done(
        self,
        result: np.ndarray,
        overlay: np.ndarray,
        dx_grid: np.ndarray,
        dy_grid: np.ndarray,
        dx_heat, dy_heat, mag_heat,
        info: dict,
        remapped_before: np.ndarray,
        transformed_after: np.ndarray,
    ):
        self._result_bgr  = result
        self._overlay_bgr = overlay
        self._dx_grid     = dx_grid
        self._dy_grid     = dy_grid

        # Update images tab
        self._canvas_out.show(overlay)

        # Update transform tab
        self._remapped_bgr = remapped_before
        self._transformed_bgr = transformed_after
        self._canvas_remapped.show(remapped_before)
        self._canvas_transformed.show(transformed_after)

        # Update heatmaps tab
        if dx_heat is not None:
            self._canvas_dx.show(dx_heat)
            self._canvas_dy.show(dy_heat)
            self._canvas_mag.show(mag_heat)

        # Update info tab
        self._update_info(info)

        # Status bar
        msg = (
            f"[OK] {info['transform']} | {info['grid']} grid | "
            f"dX [{info['dx_min']:.1f}, {info['dx_max']:.1f}] | "
            f"dY [{info['dy_min']:.1f}, {info['dy_max']:.1f}] | "
            f"{info['elapsed_ms']:.0f} ms"
        )
        self._set_status(msg, OK)
        self._stop_progress()

        # Auto-trigger comparison/sparse-vs-dense if those tabs are visible
        current_tab = self._notebook.select()
        tab_text = self._notebook.tab(current_tab, "text")
        if "Comparison" in tab_text:
            self._schedule_comparison()
        elif "Sparse vs Dense" in tab_text:
            self._schedule_spd_comparison()

    def _on_pipeline_error(self, msg: str):
        self._set_status(f"Error: {msg}", ERR)
        self._stop_progress()
        messagebox.showerror("Processing Error", msg)

    def _update_info(self, info: dict):
        lines = [
            "--- Remap Parameters ------------------------------------",
            f"  Transform       : {info['transform']}",
            f"  Sparse grid     : {info['grid']} nodes",
            f"  Grid interp     : {info['grid_interp']}",
            f"  Pixel interp    : {info['remap_interp']}",
            f"  Border mode     : {info['border']}",
            f"  Image size      : {info['size']}",
        ]

        # Add Flip/Rotate info
        if info.get("flip") or info.get("rotate"):
            lines.append("")
            lines.append("--- Post-Processing -------------------------------------")
            if info.get("flip"):
                lines.append(f"  Flip            : {info['flip']}")
            if info.get("rotate"):
                lines.append(f"  Rotation        : {info['rotate']}°")

        # Add ROI info if present
        if info.get("roi"):
            rx, ry, rw, rh = info["roi"]
            lines.append("")
            lines.append("--- Region of Interest (ROI) ----------------------------")
            lines.append(f"  Position        : ({rx}, {ry})")
            lines.append(f"  Size            : {rw} x {rh}")

        # Add Grid dX/dY values (show max 5x5 subset)
        if "dx_grid" in info and "dy_grid" in info:
            dx_grid = info["dx_grid"]
            dy_grid = info["dy_grid"]
            rows, cols = dx_grid.shape
            max_show = 5

            lines.append("")
            lines.append("--- Grid dX Values (Sparse) ------------------------------")
            lines.append(f"Shape: {rows}x{cols} (showing up to {min(rows, max_show)}x{min(cols, max_show)})")
            lines.append("")

            # Format dX grid with fixed width (limit to 5x5)
            for r in range(min(rows, max_show)):
                row_vals = [f"{dx_grid[r, c]:8.2f}" for c in range(min(cols, max_show))]
                lines.append("  " + " ".join(row_vals))
            if rows > max_show or cols > max_show:
                lines.append("  ... (truncated)")

            lines.append("")
            lines.append("--- Grid dY Values (Sparse) ------------------------------")
            lines.append(f"Shape: {rows}x{cols} (showing up to {min(rows, max_show)}x{min(cols, max_show)})")
            lines.append("")

            # Format dY grid with fixed width (limit to 5x5)
            for r in range(min(rows, max_show)):
                row_vals = [f"{dy_grid[r, c]:8.2f}" for c in range(min(cols, max_show))]
                lines.append("  " + " ".join(row_vals))
            if rows > max_show or cols > max_show:
                lines.append("  ... (truncated)")

        lines.extend([
            "",
            "--- Displacement Statistics ------------------------------",
            f"  dX range        : {info['dx_min']:.2f} .. {info['dx_max']:.2f} px",
            f"  dY range        : {info['dy_min']:.2f} .. {info['dy_max']:.2f} px",
            f"  Max magnitude   : {info['mag_max']:.2f} px",
            "",
            "--- Performance -----------------------------------------",
            f"  Elapsed         : {info['elapsed_ms']:.1f} ms",
            "",
            "--- Algorithm Notes -------------------------------------",
            "  Mapping type    : Inverse (src <- dst + displacement)",
            "  Core function   : cv2.remap(src, map_x, map_y, ...)",
            "  Grid upsampling : RectBivariateSpline (scipy)",
        ])

        text = "\n".join(lines)
        self._info_text.config(state=tk.NORMAL)
        self._info_text.delete("1.0", tk.END)
        self._info_text.insert(tk.END, text)
        self._info_text.config(state=tk.DISABLED)

    # ======================================================================
    # Save / Export actions
    # ======================================================================
    def _save_image(self):
        if self._result_bgr is None:
            messagebox.showwarning("Nothing to save", "Apply remapping first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All", "*.*")],
            initialfile=f"{self._transform_var.get()}_remapped.png",
        )
        if not path:
            return
        cv2.imwrite(path, self._overlay_bgr)
        self._set_status(f"Saved -> {path}", OK)

    def _save_figure(self):
        """Save a side-by-side comparison PNG using matplotlib (no display)."""
        if self._result_bgr is None or self._source_bgr is None:
            messagebox.showwarning("Nothing to save", "Apply remapping first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
            initialfile=f"{self._transform_var.get()}_comparison.png",
        )
        if not path:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.patch.set_facecolor("#111111")

            def bgr2rgb(img):
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            axes[0].imshow(bgr2rgb(self._source_bgr))
            axes[0].set_title("Original", color="white", fontsize=11)
            axes[0].axis("off")
            axes[1].imshow(bgr2rgb(self._overlay_bgr))
            axes[1].set_title(
                f"Remapped  [{self._transform_var.get()}]",
                color="white", fontsize=11,
            )
            axes[1].axis("off")

            if self._dx_grid is not None:
                mag = np.sqrt(self._dx_grid**2 + self._dy_grid**2)
                fig.text(
                    0.5, 0.01,
                    f"Grid {self._grid_rows_var.get()}x{self._grid_cols_var.get()}  |  "
                    f"Max displacement {mag.max():.1f} px  ·  "
                    f"Interp {self._interp_var.get()}",
                    ha="center", fontsize=9, color="#aaaaaa",
                )

            fig.tight_layout(rect=[0, 0.04, 1, 1])
            fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="#111111")
            plt.close(fig)
            self._set_status(f"Figure saved -> {path}", OK)
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _export_all(self):
        """Export all presets x all sample patterns to a chosen directory."""
        out_dir = filedialog.askdirectory(title="Choose export directory")
        if not out_dir:
            return
        out_path = Path(out_dir)

        transforms = list(TRANSFORM_PARAMS.keys())
        transforms.remove("identity")  # skip trivial pass-through
        samples = list(SAMPLE_IMAGES.keys())
        W = self._width_var.get()
        H = self._height_var.get()
        grid_rows = self._grid_rows_var.get()
        grid_cols = self._grid_cols_var.get()
        interp      = self._interp_var.get()
        grid_interp = self._grid_interp_var.get()
        border      = self._border_var.get()
        params = {
            "k1":       self._sliders["k1"].value,
            "strength": self._sliders["strength"].value,
            "amplitude": self._sliders["amplitude"].value,
        }

        total = len(transforms) * len(samples)
        self._set_status(f"Exporting {total} images ...", ACCENT)
        self._start_progress()

        def _worker():
            engine = GridRemapEngine()
            done = 0
            for tf in transforms:
                tf_dir = out_path / tf
                tf_dir.mkdir(parents=True, exist_ok=True)
                for sample_name in samples:
                    done += 1
                    try:
                        image = SAMPLE_IMAGES[sample_name](W, H)
                        dx_grid, dy_grid = self._build_grid(
                            tf, H, W, grid_rows, grid_cols,
                            {"k1": params["k1"], "strength": params["strength"],
                             "amplitude": params["amplitude"]},
                        )
                        map_x, map_y = engine.build_remap_maps(
                            (H, W), dx_grid, dy_grid, grid_interp=grid_interp
                        )
                        result = engine.apply_remap(
                            image, map_x, map_y,
                            interpolation=interp, border_mode=border,
                        )
                        overlay = engine.overlay_grid(
                            result, grid_rows, grid_cols, map_x, map_y
                        )
                        cv2.imwrite(str(tf_dir / f"{sample_name}.png"), overlay)
                    except Exception:
                        pass
                    self.after(
                        0, self._set_status,
                        f"Exporting ... {done}/{total} ({tf}/{sample_name})", ACCENT,
                    )

            self.after(0, self._set_status,
                       f"Exported {total} images -> {out_path}", OK)
            self.after(0, self._stop_progress)

        threading.Thread(target=_worker, daemon=True).start()

    # ======================================================================
    # Comparison Tab — Method A (Sequential) vs Method B (Grid-Folded)
    # ======================================================================

    def _build_comparison_tab(self, parent: ttk.Frame):
        """
        Tab layout
        ----------
        Row 0 : Controls strip  (Run button, diff-amp slider, colormap)
        Row 1 : Theory note banner
        Row 2 : Three ImageCanvases — Method A | Method B Grid | Residual A-B
        Row 3 : Statistics text
        """
        parent.rowconfigure(2, weight=1)
        parent.columnconfigure(0, weight=1)

        # ── Row 0: Controls ──────────────────────────────────────────────
        ctrl = ttk.Frame(parent)
        ctrl.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 2))

        # Auto-run indicator
        self._cmp_auto_lbl = ttk.Label(
            ctrl, text="● Auto-run", font=FONT_BODY, foreground=OK
        )
        self._cmp_auto_lbl.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(ctrl, text="Diff amp ×", font=FONT_BODY).pack(side=tk.LEFT)
        self._diff_amp_var = tk.DoubleVar(value=5.0)
        self._diff_amp_scale = ttk.Scale(
            ctrl, orient=tk.HORIZONTAL, from_=1.0, to=30.0,
            variable=self._diff_amp_var, length=120,
            command=lambda _: self._refresh_diff_display(),
        )
        self._diff_amp_scale.pack(side=tk.LEFT, padx=4)
        self._diff_amp_lbl = ttk.Label(ctrl, text="5.0", width=4, font=FONT_MONO)
        self._diff_amp_lbl.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(ctrl, text="Colormap", font=FONT_BODY).pack(side=tk.LEFT)
        self._diff_cmap_var = tk.StringVar(value="hot")
        cmap_cb = ttk.Combobox(
            ctrl, textvariable=self._diff_cmap_var,
            values=["hot", "jet", "plasma", "inferno", "gray"],
            state="readonly", width=9,
        )
        cmap_cb.pack(side=tk.LEFT, padx=4)
        cmap_cb.bind("<<ComboboxSelected>>", lambda _: self._refresh_diff_display())

        # ── Row 1: Theory note ────────────────────────────────────────────
        note_frame = ttk.Frame(parent, style="TFrame")
        note_frame.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 4))

        note_text = (
            "Theory test — Method A: cv2.remap → flip/rotate post-process.  "
            "Method B: fold flip/rotate into sparse dx/dy grid → re-interpolate → "
            "single cv2.remap.  "
            "Residual shows interpolation error introduced by grid re-sampling."
        )
        ttk.Label(
            note_frame, text=note_text, font=("Segoe UI", 8, "italic"),
            foreground=FG2, wraplength=900, anchor="w",
        ).pack(fill=tk.X)

        # ── Row 2: Three image canvases ───────────────────────────────────
        img_frame = ttk.Frame(parent)
        img_frame.grid(row=2, column=0, sticky="nsew", padx=6, pady=2)
        img_frame.columnconfigure(0, weight=1)
        img_frame.columnconfigure(1, weight=1)
        img_frame.columnconfigure(2, weight=1)
        img_frame.rowconfigure(0, weight=1)

        self._canvas_cmp_a = ImageCanvas(
            img_frame,
            "Method A — Sequential\n(Remap → flip/rotate)",
        )
        self._canvas_cmp_a.grid(row=0, column=0, sticky="nsew", padx=(0, 2), pady=2)

        self._canvas_cmp_b = ImageCanvas(
            img_frame,
            "Method B — Grid-Folded\n(fold into grid → re-interp → Remap)",
        )
        self._canvas_cmp_b.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)

        self._canvas_diff = ImageCanvas(
            img_frame,
            "Residual  |A − B|  (amplified)",
        )
        self._canvas_diff.grid(row=0, column=2, sticky="nsew", padx=(2, 0), pady=2)

        # ── Row 3: Statistics ─────────────────────────────────────────────
        stats_outer = ttk.LabelFrame(parent, text="  Comparison Statistics  ")
        stats_outer.grid(row=3, column=0, sticky="ew", padx=6, pady=(2, 6))

        self._cmp_stats_text = tk.Text(
            stats_outer,
            bg="#0d0d1a", fg=FG, font=FONT_MONO,
            relief=tk.FLAT, state=tk.DISABLED,
            height=6, wrap=tk.NONE,
            selectbackground=SELECT_BG, selectforeground=SELECT_FG,
        )
        sb = ttk.Scrollbar(stats_outer, command=self._cmp_stats_text.yview)
        self._cmp_stats_text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._cmp_stats_text.pack(fill=tk.X, padx=4, pady=4)

    # ------------------------------------------------------------------
    # Comparison — trigger
    # ------------------------------------------------------------------
    def _on_tab_changed(self, event=None):
        """Auto-run comparison when the comparison tab is selected."""
        current_tab = self._notebook.select()
        tab_text = self._notebook.tab(current_tab, "text")
        if "Comparison" in tab_text:
            self._schedule_comparison()
        elif "Sparse vs Dense" in tab_text:
            self._schedule_spd_comparison()

    def _schedule_comparison(self):
        """Schedule comparison to run (debounced)."""
        # Cancel any pending comparison timer
        if hasattr(self, '_cmp_timer') and self._cmp_timer:
            self.after_cancel(self._cmp_timer)
        # Schedule to run after a short delay
        self._cmp_timer = self.after(300, self._run_comparison)

    def _run_comparison(self):
        """Validate state, then run comparison in background thread."""
        if self._processing:
            self._set_status("Main pipeline is running — wait and retry.", WARN)
            return
        if self._source_bgr is None:
            self._set_status("Load an image first (Input Image section).", WARN)
            return
        if self._dx_grid is None or self._dy_grid is None:
            self._set_status("Run 'Apply Remap' first to generate the grid.", WARN)
            return

        flip_h  = self._flip_h_var.get()
        flip_v  = self._flip_v_var.get()
        rotate  = self._rotate_var.get()
        if not flip_h and not flip_v and rotate == 0:
            # No post-process active — show informational diff (should be zero)
            self._set_status(
                "No flip/rotate active — both methods are identical; diff will be zero.",
                WARN,
            )

        params = {
            # Grid / interpolation (same as main pipeline)
            "grid_rows":   self._grid_rows_var.get(),
            "grid_cols":   self._grid_cols_var.get(),
            "grid_interp": self._grid_interp_var.get(),
            "interp":      self._interp_var.get(),
            "border":      self._border_var.get(),
            # Post-process ops to fold
            "flip_h":  flip_h,
            "flip_v":  flip_v,
            "rotate":  rotate,
            # ROI (applied identically to both outputs for a fair comparison)
            "roi_draw": self._roi_draw_var.get(),
            "roi_crop": self._roi_crop_var.get(),
            "roi_x":    self._roi_x_var.get(),
            "roi_y":    self._roi_y_var.get(),
            "roi_w":    self._roi_w_var.get(),
            "roi_h":    self._roi_h_var.get(),
        }

        self._start_progress()
        self._set_status("Running comparison …", ACCENT)

        threading.Thread(
            target=self._run_comparison_thread,
            args=(
                self._source_bgr.copy(),
                self._dx_grid.copy(),
                self._dy_grid.copy(),
                params,
            ),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Comparison — background worker
    # ------------------------------------------------------------------
    def _run_comparison_thread(
        self,
        image: np.ndarray,
        dx_grid: np.ndarray,
        dy_grid: np.ndarray,
        params: dict,
    ):
        """
        Compute Method A and Method B, diff, and statistics.

        Method A — Sequential
        ─────────────────────
          1. Build dense maps from dx_grid / dy_grid (same as main pipeline).
          2. cv2.remap → remapped.
          3. Apply flip / rotate → result_a.
          4. Apply ROI crop (if enabled).

        Method B — Grid-Folded
        ──────────────────────
          1. Build same dense maps (step 1 of Method A).
          2. Mathematically fold flip/rotate into the DENSE maps
             → folded_map_x, folded_map_y  (shape may change for 90°/270°).
          3. SAMPLE folded dense maps at grid node positions
             → sparse dx_grid_B, dy_grid_B.
          4. Re-interpolate sparse B grid → dense maps B  (introduces error).
          5. cv2.remap with maps B → result_b (NO post-process).
          6. Apply same ROI crop (if enabled).

        The residual |A − B| reveals the error from grid re-sampling.
        """
        try:
            H, W     = image.shape[:2]
            gR       = params["grid_rows"]
            gC       = params["grid_cols"]
            g_interp = params["grid_interp"]
            px_interp = params["interp"]
            border   = params["border"]
            flip_h   = params["flip_h"]
            flip_v   = params["flip_v"]
            rotate   = params["rotate"]

            engine = GridRemapEngine()
            t0 = time.perf_counter()

            # ── Shared step: build dense maps from the given sparse grid ──
            map_x, map_y = engine.build_remap_maps(
                (H, W), dx_grid, dy_grid, grid_interp=g_interp
            )

            # ── Method A ──────────────────────────────────────────────────
            remapped_a = engine.apply_remap(
                image, map_x, map_y,
                interpolation=px_interp, border_mode=border,
            )
            result_a = self._apply_post_process(remapped_a, flip_h, flip_v, rotate)

            # ── Method B ──────────────────────────────────────────────────
            # Fold flip/rotate into the dense maps, then remap directly.
            # No sparse round-trip needed — folded maps are valid cv2.remap
            # inputs and produce zero approximation error.
            folded_mx, folded_my, out_H, out_W = self._apply_geometric_fold(
                map_x, map_y, H, W, flip_h, flip_v, rotate
            )
            result_b = engine.apply_remap(
                image, folded_mx, folded_my,
                interpolation=px_interp, border_mode=border,
            )

            t1 = time.perf_counter()

            # ── ROI — applied identically to both for a fair comparison ──
            result_a, result_b = self._apply_roi_to_pair(
                result_a, result_b, params
            )

            # ── Resize B to match A if rotation changed dimensions ────────
            if result_b.shape[:2] != result_a.shape[:2]:
                result_b = cv2.resize(
                    result_b,
                    (result_a.shape[1], result_a.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            # ── Compute residual and statistics ───────────────────────────
            diff_f  = result_a.astype(np.float32) - result_b.astype(np.float32)
            abs_diff = np.abs(diff_f)

            rmse   = float(np.sqrt(np.mean(diff_f ** 2)))
            max_err = float(abs_diff.max())
            mean_err = float(abs_diff.mean())
            psnr   = (
                float(20 * np.log10(255.0 / rmse)) if rmse > 1e-9 else float("inf")
            )
            identical_pct = float(
                np.sum(abs_diff.max(axis=2) == 0) / (result_a.shape[0] * result_a.shape[1]) * 100
            )

            ch_rmse: dict[str, float] = {}
            for idx, ch_name in enumerate(["B", "G", "R"]):
                ch_rmse[ch_name] = float(
                    np.sqrt(np.mean(diff_f[:, :, idx] ** 2))
                )

            folds = []
            if flip_h:
                folds.append("Flip-H")
            if flip_v:
                folds.append("Flip-V")
            if rotate:
                folds.append(f"Rotate {rotate}°")

            stats = {
                "rmse":          rmse,
                "max_err":       max_err,
                "mean_err":      mean_err,
                "psnr_db":       psnr,
                "identical_pct": identical_pct,
                "ch_rmse":       ch_rmse,
                "elapsed_ms":    (t1 - t0) * 1000,
                "folds":         folds if folds else ["(none — identity comparison)"],
                "grid":          f"{gR}×{gC}",
                "grid_interp":   g_interp,
                "out_size":      f"{result_a.shape[1]}×{result_a.shape[0]}",
            }

            self.after(
                0, self._on_comparison_done,
                result_a, result_b, diff_f, stats,
            )

        except Exception as exc:
            self.after(0, self._on_pipeline_error, f"Comparison: {exc}")

    # ------------------------------------------------------------------
    # Comparison — helpers (static)
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_post_process(
        image: np.ndarray,
        flip_h: bool,
        flip_v: bool,
        rotate: int,
    ) -> np.ndarray:
        """
        Apply flip then rotate to a remapped image.

        Order contract
        --------------
        Follows _FOLD_ORDER = ("flip", "rotate") — identical to
        _run_pipeline_thread and _apply_geometric_fold.
        DO NOT reorder: flip→rotate ≠ rotate→flip.

        Parameters
        ----------
        rotate : int
            Must be one of {0, 90, 180, 270}.  Any other value raises ValueError
            immediately rather than silently producing a no-op.
        """
        # Validate rotate before applying any post-processing.
        if rotate not in (0, 90, 180, 270):
            raise ValueError(
                f"Unsupported rotation {rotate!r}. Must be one of 0, 90, 180, 270."
            )

        result = image.copy()

        # Step 1 — flip  (order: flip first, per _FOLD_ORDER)
        if flip_h and flip_v:
            result = cv2.flip(result, -1)   # both axes
        elif flip_h:
            result = cv2.flip(result, 1)    # horizontal
        elif flip_v:
            result = cv2.flip(result, 0)    # vertical

        # Step 2 — rotate  (order: rotate second, per _FOLD_ORDER)
        if rotate == 90:
            result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == 180:
            result = cv2.rotate(result, cv2.ROTATE_180)
        elif rotate == 270:
            result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # rotate == 0: identity, no-op

        return result

    @staticmethod
    def _apply_geometric_fold(
        map_x: np.ndarray,
        map_y: np.ndarray,
        H: int,
        W: int,
        flip_h: bool,
        flip_v: bool,
        rotate: int,
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Fold flip/rotate post-processing into the dense remap maps so that a
        single cv2.remap call produces the same result as the sequential pipeline.

        Mathematical identity
        ---------------------
        For any operation T and its inverse T⁻¹:
            cv2.remap(src, folded_map_x, folded_map_y)
                == T( cv2.remap(src, map_x, map_y) )
        where  folded_map[r, c] = map[ T⁻¹(r, c) ]

        Fold derivation per operation
        ------------------------------
        Flip-H:    output[r, c] = in[r, W−1−c]      → m[:, ::-1]
        Flip-V:    output[r, c] = in[H−1−r, c]      → m[::-1, :]
        Rot-90CW:  output[r, c] = in[H−1−c, r]      → m[::-1, :].T   shape→(W, H)
        Rot-180:   output[r, c] = in[H−1−r, W−1−c]  → m[::-1, ::-1]
        Rot-270CW: output[r, c] = in[c, W−1−r]      → m[:, ::-1].T   shape→(W, H)

        Order contract
        ---------------------------
        Follows _FOLD_ORDER = ("flip", "rotate") — MUST be identical to
        _run_pipeline_thread and _apply_post_process.
        Flip→Rotate ≠ Rotate→Flip (non-commutative).

        Parameters
        ----------
        map_x, map_y : float32 ndarray, shape (H, W)
            Dense remap maps produced by build_remap_maps().  Must be float32.
        H, W : int
            Input map dimensions — kept as explicit parameters so dimension
            tracking is transparent and independent of array .shape.
        rotate : int
            Must be one of {0, 90, 180, 270}.  Validated before any work.

        Returns
        -------
        folded_map_x, folded_map_y : float32 C-contiguous ndarray
        cur_H, cur_W : int
            Output dimensions — equal to (W, H) for 90°/270°, (H, W) otherwise.
        """
        # Validate rotate before any folding work is done.
        if rotate not in (0, 90, 180, 270):
            raise ValueError(
                f"Unsupported rotation {rotate!r}. Must be one of 0, 90, 180, 270."
            )

        # Avoid eager copies. All flip ops below create numpy views
        # (reversed strides), not copies.  The first op that requires a
        # contiguous layout (rotation via np.ascontiguousarray) materialises
        # a new array only when actually needed.
        m_x = map_x
        m_y = map_y

        # Track current dimensions explicitly through each stage.
        # Flips are dimension-preserving; rotations swap H and W.  Using
        # cur_H / cur_W (rather than bare H / W) makes each fold operation
        # self-contained: if the order were ever extended, each stage would
        # still reference the correct shape.
        cur_H, cur_W = H, W

        # ── Step 1 — fold flips  (per _FOLD_ORDER: flip first) ───────────────
        # Axis-reversal views; dimension-preserving; cur_H / cur_W unchanged.
        if flip_h and flip_v:
            m_x = m_x[::-1, ::-1]
            m_y = m_y[::-1, ::-1]
        elif flip_h:
            m_x = m_x[:, ::-1]
            m_y = m_y[:, ::-1]
        elif flip_v:
            m_x = m_x[::-1, :]
            m_y = m_y[::-1, :]
        # cur_H, cur_W unchanged — flips are dimension-preserving.

        # ── Step 2 — fold rotation  (per _FOLD_ORDER: rotate second) ─────────
        # np.ascontiguousarray materialises a C-contiguous array.  It is a
        # no-op when the input is already C-contiguous, avoiding a redundant
        # copy in the rotate=0 path.
        if rotate == 90:
            # output[r, c] = after_flip[cur_H−1−c, r]  →  m[::-1, :].T
            m_x = np.ascontiguousarray(m_x[::-1, :].T)  # (cur_H, cur_W) → (cur_W, cur_H)
            m_y = np.ascontiguousarray(m_y[::-1, :].T)
            cur_H, cur_W = cur_W, cur_H                  # dimensions swap
        elif rotate == 180:
            # output[r, c] = after_flip[cur_H−1−r, cur_W−1−c]  →  m[::-1, ::-1]
            m_x = np.ascontiguousarray(m_x[::-1, ::-1])
            m_y = np.ascontiguousarray(m_y[::-1, ::-1])
            # cur_H, cur_W unchanged — 180° is dimension-preserving.
        elif rotate == 270:
            # output[r, c] = after_flip[c, cur_W−1−r]  →  m[:, ::-1].T
            m_x = np.ascontiguousarray(m_x[:, ::-1].T)  # (cur_H, cur_W) → (cur_W, cur_H)
            m_y = np.ascontiguousarray(m_y[:, ::-1].T)
            cur_H, cur_W = cur_W, cur_H                  # dimensions swap
        # rotate == 0: identity, no fold needed.

        # map_x / map_y are guaranteed float32 by build_remap_maps.
        # np.ascontiguousarray preserves dtype and is a no-op when the array is
        # already C-contiguous float32.  An explicit .astype(float32) would create
        # a redundant copy; we avoid it.
        return (
            np.ascontiguousarray(m_x),
            np.ascontiguousarray(m_y),
            cur_H,
            cur_W,
        )

    @staticmethod
    def _sample_grid_from_dense(
        folded_map_x: np.ndarray,
        folded_map_y: np.ndarray,
        out_H: int,
        out_W: int,
        grid_rows: int,
        grid_cols: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample the folded dense maps at uniformly-spaced grid node positions
        to produce a sparse (grid_rows × grid_cols) displacement grid B.

        Node positions MUST be the exact float linspace values that
        build_remap_maps() uses as spline knots.

        Why integer positions are wrong
        --------------------------------
        np.linspace(0, N-1, K) produces non-integer values for most K.
        Truncating to int shifts each knot by up to 0.94 px.
        For a Flip-H fold the displacement field has slope −2 px/px, so a
        0.94 px positional error → ~1.88 px displacement error per knot.
        The bicubic spline propagates this across the entire image, yielding
        ~19 px RMSE — a 200× amplification confirmed empirically.

        Fix: sub-pixel bilinear sampling via cv2.remap at exact float positions.
        BORDER_REPLICATE prevents out-of-range access at image edges.

        Returns
        -------
        dx_grid_b, dy_grid_b : float32 ndarray, shape (grid_rows, grid_cols)
        """
        # Exact float node positions — identical to those used in build_remap_maps.
        y_float = np.linspace(0.0, out_H - 1, grid_rows).astype(np.float32)
        x_float = np.linspace(0.0, out_W - 1, grid_cols).astype(np.float32)
        gx, gy  = np.meshgrid(x_float, y_float)   # both (grid_rows, grid_cols)

        # Sub-pixel bilinear sampling at exact float positions.
        sampled_mx = cv2.remap(
            folded_map_x, gx, gy,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        sampled_my = cv2.remap(
            folded_map_y, gx, gy,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Absolute source coordinates → displacements relative to node position.
        dx_grid_b = sampled_mx - gx
        dy_grid_b = sampled_my - gy
        return dx_grid_b, dy_grid_b

    @staticmethod
    def _apply_roi_to_pair(
        result_a: np.ndarray,
        result_b: np.ndarray,
        params: dict,
    ):
        """Apply the same ROI crop to both images (if enabled)."""
        if not params.get("roi_crop"):
            return result_a, result_b

        H_a, W_a = result_a.shape[:2]
        rx = int(max(0, min(params.get("roi_x", 0), W_a - 1)))
        ry = int(max(0, min(params.get("roi_y", 0), H_a - 1)))
        rw = int(max(1, min(params.get("roi_w", W_a), W_a - rx)))
        rh = int(max(1, min(params.get("roi_h", H_a), H_a - ry)))

        return (
            result_a[ry : ry + rh, rx : rx + rw],
            result_b[ry : ry + rh, rx : rx + rw],
        )

    # ------------------------------------------------------------------
    # Comparison — receive results on main thread
    # ------------------------------------------------------------------
    def _on_comparison_done(
        self,
        result_a: np.ndarray,
        result_b: np.ndarray,
        diff_f: np.ndarray,
        stats: dict,
    ):
        self._cmp_result_a = result_a
        self._cmp_result_b = result_b
        self._cmp_diff     = diff_f

        self._canvas_cmp_a.show(result_a)
        self._canvas_cmp_b.show(result_b)
        self._refresh_diff_display()
        self._update_comparison_stats(stats)

        self._set_status(
            f"Comparison done — RMSE: {stats['rmse']:.2f}  "
            f"Max err: {stats['max_err']:.1f}  "
            f"PSNR: {stats['psnr_db']:.1f} dB  "
            f"Identical px: {stats['identical_pct']:.1f}%  "
            f"({stats['elapsed_ms']:.0f} ms)",
            OK,
        )
        self._stop_progress()

    def _refresh_diff_display(self):
        """Re-render the diff canvas whenever amplification or colormap changes."""
        if self._cmp_diff is None:
            return

        amp  = self._diff_amp_var.get()
        cmap_name = self._diff_cmap_var.get()
        self._diff_amp_lbl.config(text=f"{amp:.1f}")

        _cmap_map = {
            "hot":     cv2.COLORMAP_HOT,
            "jet":     cv2.COLORMAP_JET,
            "plasma":  cv2.COLORMAP_PLASMA,
            "inferno": cv2.COLORMAP_INFERNO,
            "gray":    cv2.COLORMAP_BONE,
        }
        cmap_cv = _cmap_map.get(cmap_name, cv2.COLORMAP_HOT)

        # Amplify absolute diff, collapse to luminance, colourise
        abs_diff = np.abs(self._cmp_diff)
        # Per-pixel max across channels as a single 2-D magnitude
        magnitude = abs_diff.max(axis=2)
        scaled = np.clip(magnitude * amp, 0, 255).astype(np.uint8)
        diff_vis = cv2.applyColorMap(scaled, cmap_cv)
        self._canvas_diff.show(diff_vis)

    def _update_comparison_stats(self, stats: dict):
        """Write statistics into the comparison stats text widget."""
        folds_str = ", ".join(stats["folds"])
        psnr_str  = (
            f"{stats['psnr_db']:.2f} dB"
            if stats["psnr_db"] != float("inf")
            else "∞  (identical)"
        )
        ch = stats["ch_rmse"]

        lines = [
            "─── Overall Error ───────────────────────────────────────────",
            f"  RMSE            : {stats['rmse']:.4f} px",
            f"  Max error       : {stats['max_err']:.2f} px",
            f"  Mean error      : {stats['mean_err']:.4f} px",
            f"  PSNR            : {psnr_str}",
            f"  Identical px    : {stats['identical_pct']:.2f} %",
            "",
            "─── Per-Channel RMSE ────────────────────────────────────────",
            f"  Blue            : {ch['B']:.4f}",
            f"  Green           : {ch['G']:.4f}",
            f"  Red             : {ch['R']:.4f}",
            "",
            f"  Output size     : {stats['out_size']}",
            "",
            "─── Comparison Config ───────────────────────────────────────",
            f"  Transforms folded : {folds_str}",
            f"  Elapsed           : {stats['elapsed_ms']:.1f} ms",
            "",
            "─── Interpretation ──────────────────────────────────────────",
            "  RMSE = 0, PSNR = ∞  →  grid-folded is pixel-perfect",
            "  RMSE > 0            →  error from grid re-sampling / re-interp",
            "  Higher grid density  →  lower error (test with larger grid)",
        ]

        text = "\n".join(lines)
        self._cmp_stats_text.config(state=tk.NORMAL)
        self._cmp_stats_text.delete("1.0", tk.END)
        self._cmp_stats_text.insert(tk.END, text)
        self._cmp_stats_text.config(state=tk.DISABLED)


    # ======================================================================
    # Sparse vs Dense Tab — naive upsample vs bicubic spline reconstruction
    # ======================================================================

    def _build_sparse_vs_dense_tab(self, parent: ttk.Frame):
        """
        Tab layout — mirrors the Comparison tab structure.
        ──────────────────────────────────────────────────
        Row 0 : Controls strip  (diff-amp slider, colormap picker)
        Row 1 : Theory note banner
        Row 2 : Three ImageCanvases — Method B | Method C | Residual B-C
        Row 3 : Statistics text
        """
        parent.rowconfigure(2, weight=1)
        parent.columnconfigure(0, weight=1)

        # ── Row 0: Controls ──────────────────────────────────────────────
        ctrl = ttk.Frame(parent)
        ctrl.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 2))

        self._spd_auto_lbl = ttk.Label(
            ctrl, text="● Auto-run", font=FONT_BODY, foreground=OK
        )
        self._spd_auto_lbl.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(ctrl, text="Diff amp ×", font=FONT_BODY).pack(side=tk.LEFT)
        self._spd_amp_var = tk.DoubleVar(value=5.0)
        ttk.Scale(
            ctrl, orient=tk.HORIZONTAL, from_=1.0, to=30.0,
            variable=self._spd_amp_var, length=120,
            command=lambda _: self._refresh_spd_diff_display(),
        ).pack(side=tk.LEFT, padx=4)
        self._spd_amp_lbl = ttk.Label(ctrl, text="5.0", width=4, font=FONT_MONO)
        self._spd_amp_lbl.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(ctrl, text="Colormap", font=FONT_BODY).pack(side=tk.LEFT)
        self._spd_cmap_var = tk.StringVar(value="hot")
        cmap_cb = ttk.Combobox(
            ctrl, textvariable=self._spd_cmap_var,
            values=["hot", "jet", "plasma", "inferno", "gray"],
            state="readonly", width=9,
        )
        cmap_cb.pack(side=tk.LEFT, padx=4)
        cmap_cb.bind(
            "<<ComboboxSelected>>",
            lambda _: self._refresh_spd_diff_display(),
        )

        # ── Row 1: Theory note ────────────────────────────────────────────
        note_frame = ttk.Frame(parent)
        note_frame.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 4))

        note_text = (
            "Method B — Dense Folded (reference): dense maps → fold flip/rotate "
            "via array index ops → single cv2.remap.  Zero approximation error.  "
            "Method C — Sparse Direct: apply the identical array ops to the raw "
            "sparse (gR×gC) displacement grid → build_remap_maps → cv2.remap.  "
            "Residual exposes the interpolation error from bypassing the dense round-trip.  "
            "For Flip-H/V on a linear field the error is near-zero; "
            "for 90°/270° rotation or non-linear distortions the error grows."
        )
        ttk.Label(
            note_frame, text=note_text,
            font=("Segoe UI", 8, "italic"),
            foreground=FG2, wraplength=900, anchor="w",
        ).pack(fill=tk.X)

        # ── Row 2: Three image canvases ───────────────────────────────────
        img_frame = ttk.Frame(parent)
        img_frame.grid(row=2, column=0, sticky="nsew", padx=6, pady=2)
        img_frame.columnconfigure(0, weight=1)
        img_frame.columnconfigure(1, weight=1)
        img_frame.columnconfigure(2, weight=1)
        img_frame.rowconfigure(0, weight=1)

        self._canvas_spd_dense = ImageCanvas(
            img_frame,
            "Method B — Dense Folded\n(array ops on dense maps → cv2.remap)",
        )
        self._canvas_spd_dense.grid(
            row=0, column=0, sticky="nsew", padx=(0, 2), pady=2
        )

        self._canvas_spd_sparse = ImageCanvas(
            img_frame,
            "Method C — Sparse Direct\n(array ops on sparse grid → build_remap_maps → cv2.remap)",
        )
        self._canvas_spd_sparse.grid(
            row=0, column=1, sticky="nsew", padx=2, pady=2
        )

        self._canvas_spd_diff = ImageCanvas(
            img_frame,
            "Residual  |B − C|  (amplified)",
        )
        self._canvas_spd_diff.grid(
            row=0, column=2, sticky="nsew", padx=(2, 0), pady=2
        )

        # ── Row 3: Statistics ─────────────────────────────────────────────
        stats_outer = ttk.LabelFrame(parent, text="  Sparse vs Dense Statistics  ")
        stats_outer.grid(row=3, column=0, sticky="ew", padx=6, pady=(2, 6))

        self._spd_stats_text = tk.Text(
            stats_outer,
            bg="#0d0d1a", fg=FG, font=FONT_MONO,
            relief=tk.FLAT, state=tk.DISABLED,
            height=8, wrap=tk.NONE,
            selectbackground=SELECT_BG, selectforeground=SELECT_FG,
        )
        sb = ttk.Scrollbar(stats_outer, command=self._spd_stats_text.yview)
        self._spd_stats_text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._spd_stats_text.pack(fill=tk.X, padx=4, pady=4)

    # ------------------------------------------------------------------
    # Sparse vs Dense — trigger / scheduler
    # ------------------------------------------------------------------
    def _schedule_spd_comparison(self):
        """Debounced trigger — cancels any pending run and schedules a new one."""
        if self._spd_timer:
            self.after_cancel(self._spd_timer)
        self._spd_timer = self.after(300, self._run_spd_comparison)

    def _run_spd_comparison(self):
        """Validate state, then run Sparse vs Dense comparison in background."""
        if self._processing:
            self._set_status("Main pipeline is running — wait and retry.", WARN)
            return
        if self._source_bgr is None:
            self._set_status("Load an image first.", WARN)
            return
        if self._dx_grid is None or self._dy_grid is None:
            self._set_status("Run 'Apply Remap' first to generate the grid.", WARN)
            return

        params = {
            "grid_rows":      self._grid_rows_var.get(),
            "grid_cols":      self._grid_cols_var.get(),
            "grid_interp":    self._grid_interp_var.get(),
            "interp":         self._interp_var.get(),
            "border":         self._border_var.get(),
            "flip_h":         self._flip_h_var.get(),
            "flip_v":         self._flip_v_var.get(),
            "rotate":         self._rotate_var.get(),
        }

        self._start_progress()
        self._set_status("Running Sparse vs Dense comparison …", ACCENT)

        threading.Thread(
            target=self._run_spd_thread,
            args=(
                self._source_bgr.copy(),
                self._dx_grid.copy(),
                self._dy_grid.copy(),
                params,
            ),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Sparse vs Dense — background worker
    # ------------------------------------------------------------------
    def _run_spd_thread(
        self,
        image: np.ndarray,
        dx_grid: np.ndarray,
        dy_grid: np.ndarray,
        params: dict,
    ):
        """
        Method B — Dense Folded (reference)
        ─────────────────────────────────────
          sparse dx/dy → build_remap_maps (bicubic spline) → dense maps
          → _apply_geometric_fold (array index ops on H×W arrays)
          → single cv2.remap
          Zero approximation error by construction.

        Method C — Sparse Direct
        ─────────────────────────
          Apply the IDENTICAL array index ops as _apply_geometric_fold
          but on the raw (gR×gC) sparse displacement grid.
          → build_remap_maps on transformed sparse grid
          → cv2.remap

        Key insight: _apply_geometric_fold on the dense map reindexes
        every pixel correctly.  The same ops on the sparse grid reindex
        knot VALUES but the knot POSITIONS stay at linspace(0,N-1,K).
        For 90°/270° the grid shape also changes (gR×gC → gC×gR), so
        build_remap_maps places knots over a different pixel range,
        causing systematic error that grows with distortion curvature.
        """
        try:
            H, W      = image.shape[:2]
            g_interp  = params["grid_interp"]
            px_interp = params["interp"]
            border    = params["border"]
            flip_h    = params["flip_h"]
            flip_v    = params["flip_v"]
            rotate    = params["rotate"]
            gR, gC    = dx_grid.shape

            engine = GridRemapEngine()
            t0 = time.perf_counter()

            # ── Method B — Dense Folded (reference) ───────────────────────
            map_x, map_y = engine.build_remap_maps(
                (H, W), dx_grid, dy_grid, grid_interp=g_interp
            )
            folded_mx, folded_my, out_H_b, out_W_b = self._apply_geometric_fold(
                map_x, map_y, H, W, flip_h, flip_v, rotate
            )
            result_b = engine.apply_remap(
                image, folded_mx, folded_my,
                interpolation=px_interp, border_mode=border,
            )

            # ── Method C — Sparse Direct ───────────────────────────────────
            # Apply identical array ops as _apply_geometric_fold, but on
            # the (gR × gC) sparse grid instead of the (H × W) dense map.
            dx_c = dx_grid.copy()
            dy_c = dy_grid.copy()
            out_gR, out_gC = gR, gC
            out_H_c, out_W_c = H, W

            # Step 1: fold flips (dimension-preserving).
            if flip_h and flip_v:
                dx_c = dx_c[::-1, ::-1]
                dy_c = dy_c[::-1, ::-1]
            elif flip_h:
                dx_c = dx_c[:, ::-1]
                dy_c = dy_c[:, ::-1]
            elif flip_v:
                dx_c = dx_c[::-1, :]
                dy_c = dy_c[::-1, :]

            # Step 2: fold rotation (swaps gR↔gC for 90°/270°).
            if rotate == 90:
                dx_c = np.ascontiguousarray(dx_c[::-1, :].T)
                dy_c = np.ascontiguousarray(dy_c[::-1, :].T)
                out_gR, out_gC = gC, gR
                out_H_c, out_W_c = W, H
            elif rotate == 180:
                dx_c = np.ascontiguousarray(dx_c[::-1, ::-1])
                dy_c = np.ascontiguousarray(dy_c[::-1, ::-1])
            elif rotate == 270:
                dx_c = np.ascontiguousarray(dx_c[:, ::-1].T)
                dy_c = np.ascontiguousarray(dy_c[:, ::-1].T)
                out_gR, out_gC = gC, gR
                out_H_c, out_W_c = W, H

            map_x_c, map_y_c = engine.build_remap_maps(
                (out_H_c, out_W_c),
                np.ascontiguousarray(dx_c).astype(np.float32),
                np.ascontiguousarray(dy_c).astype(np.float32),
                grid_interp=g_interp,
            )
            result_c = engine.apply_remap(
                image, map_x_c, map_y_c,
                interpolation=px_interp, border_mode=border,
            )

            t1 = time.perf_counter()

            # ── Align shapes for diff (rotation swaps H↔W) ───────────────
            if result_c.shape[:2] != result_b.shape[:2]:
                result_c = cv2.resize(
                    result_c,
                    (result_b.shape[1], result_b.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            # ── Residual and statistics ───────────────────────────────────
            diff_f   = result_b.astype(np.float32) - result_c.astype(np.float32)
            abs_diff = np.abs(diff_f)

            rmse     = float(np.sqrt(np.mean(diff_f ** 2)))
            max_err  = float(abs_diff.max())
            mean_err = float(abs_diff.mean())
            psnr     = (
                float(20 * np.log10(255.0 / rmse)) if rmse > 1e-9 else float("inf")
            )
            identical_pct = float(
                np.sum(abs_diff.max(axis=2) == 0)
                / (result_b.shape[0] * result_b.shape[1]) * 100
            )

            ch_rmse: dict[str, float] = {}
            for idx, ch_name in enumerate(["B", "G", "R"]):
                ch_rmse[ch_name] = float(
                    np.sqrt(np.mean(diff_f[:, :, idx] ** 2))
                )

            folds = []
            if flip_h:
                folds.append("Flip-H")
            if flip_v:
                folds.append("Flip-V")
            if rotate:
                folds.append(f"Rotate {rotate}°")

            stats = {
                "rmse":          rmse,
                "max_err":       max_err,
                "mean_err":      mean_err,
                "psnr_db":       psnr,
                "identical_pct": identical_pct,
                "ch_rmse":       ch_rmse,
                "elapsed_ms":    (t1 - t0) * 1000,
                "folds":         folds if folds else ["(none — identity)"],
                "grid":          f"{gR}×{gC}",
                "grid_c":        f"{out_gR}×{out_gC}",
                "grid_interp":   g_interp,
                "out_size_b":    f"{result_b.shape[1]}×{result_b.shape[0]}",
                "out_size_c":    f"{out_W_c}×{out_H_c}",
            }

            self.after(
                0, self._on_spd_done,
                result_b, result_c, diff_f, stats,
            )

        except Exception as exc:
            self.after(0, self._on_pipeline_error, f"Sparse vs Dense: {exc}")

    # ------------------------------------------------------------------
    # Sparse vs Dense — receive results on main thread
    # ------------------------------------------------------------------
    def _on_spd_done(
        self,
        result_b: np.ndarray,
        result_c: np.ndarray,
        diff_f: np.ndarray,
        stats: dict,
    ):
        self._spd_result_dense  = result_b
        self._spd_result_sparse = result_c
        self._spd_diff          = diff_f

        self._canvas_spd_dense.show(result_b)
        self._canvas_spd_sparse.show(result_c)
        self._refresh_spd_diff_display()
        self._update_spd_stats(stats)

        self._set_status(
            f"Sparse vs Dense — RMSE: {stats['rmse']:.3f}  "
            f"Max err: {stats['max_err']:.1f}  "
            f"PSNR: {stats['psnr_db']:.1f} dB  "
            f"Identical px: {stats['identical_pct']:.1f}%  "
            f"({stats['elapsed_ms']:.0f} ms)",
            OK,
        )
        self._stop_progress()

    def _refresh_spd_diff_display(self):
        """Re-render the Sparse vs Dense diff canvas on amp/colormap change."""
        if self._spd_diff is None:
            return

        amp       = self._spd_amp_var.get()
        cmap_name = self._spd_cmap_var.get()
        self._spd_amp_lbl.config(text=f"{amp:.1f}")

        _cmap_map = {
            "hot":     cv2.COLORMAP_HOT,
            "jet":     cv2.COLORMAP_JET,
            "plasma":  cv2.COLORMAP_PLASMA,
            "inferno": cv2.COLORMAP_INFERNO,
            "gray":    cv2.COLORMAP_BONE,
        }
        cmap_cv = _cmap_map.get(cmap_name, cv2.COLORMAP_HOT)

        abs_diff  = np.abs(self._spd_diff)
        magnitude = abs_diff.max(axis=2)
        scaled    = np.clip(magnitude * amp, 0, 255).astype(np.uint8)
        self._canvas_spd_diff.show(cv2.applyColorMap(scaled, cmap_cv))

    def _update_spd_stats(self, stats: dict):
        """Write Sparse vs Dense statistics into the stats text widget."""
        psnr_str = (
            f"{stats['psnr_db']:.2f} dB"
            if stats["psnr_db"] != float("inf")
            else "∞  (identical — sparse direct matches dense fold)"
        )
        ch = stats["ch_rmse"]
        folds_str = ", ".join(stats["folds"])

        lines = [
            "─── Overall Error  |Method B − Method C| ────────────────────",
            f"  RMSE            : {stats['rmse']:.4f} px",
            f"  Max error       : {stats['max_err']:.2f} px",
            f"  Mean error      : {stats['mean_err']:.4f} px",
            f"  PSNR            : {psnr_str}",
            f"  Identical px    : {stats['identical_pct']:.2f} %",
            "",
            "─── Per-Channel RMSE ────────────────────────────────────────",
            f"  Blue            : {ch['B']:.4f}",
            f"  Green           : {ch['G']:.4f}",
            f"  Red             : {ch['R']:.4f}",
            "",
            "─── Config ──────────────────────────────────────────────────",
            f"  Transforms      : {folds_str}",
            f"  Input grid      : {stats['grid']} nodes",
            f"  Method C grid   : {stats['grid_c']} nodes  (shape after fold)",
            f"  Spline interp   : {stats['grid_interp']}",
            f"  Method B output : {stats['out_size_b']}",
            f"  Method C output : {stats['out_size_c']}",
            f"  Elapsed         : {stats['elapsed_ms']:.1f} ms",
            "",
            "─── Interpretation ──────────────────────────────────────────",
            "  RMSE ≈ 0  →  sparse direct matches dense fold",
            "             (linear fields, Flip-H/V with matching grid density)",
            "  RMSE > 0  →  knot values reindexed correctly but positions",
            "             remain at original linspace — spline reconstructs",
            "             wrong field, especially at 90°/270° rotation",
            "  Higher grid density  →  less inter-knot curvature, lower error",
        ]

        text = "\n".join(lines)
        self._spd_stats_text.config(state=tk.NORMAL)
        self._spd_stats_text.delete("1.0", tk.END)
        self._spd_stats_text.insert(tk.END, text)
        self._spd_stats_text.config(state=tk.DISABLED)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fixed Grid Image Remapping - Tkinter GUI"
    )
    p.add_argument(
        "--image", "-i", default=None,
        help="Path to an image file to load on startup (optional)",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    app = RemapGUI(startup_image_path=args.image)
    app.mainloop()


if __name__ == "__main__":
    main()
