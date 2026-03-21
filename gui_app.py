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
BG    = "#1e1e2e"    # dark base
BG2   = "#2a2a3e"    # panel bg
BG3   = "#353550"    # lighter panel for inputs
ACCENT = "#7c83fd"   # highlight
ACCENT_LIGHT = "#a0a8ff"  # lighter accent for selections
FG    = "#cdd6f4"    # main text
FG2   = "#a6adc8"    # secondary text
OK    = "#a6e3a1"
ERR   = "#f38ba8"
WARN  = "#fab387"
SELECT_BG = "#5b5fd0"  # Selection background - high contrast
SELECT_FG = "#ffffff"  # Selection foreground - white text

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
        self._dx_grid: Optional[np.ndarray] = None
        self._dy_grid: Optional[np.ndarray] = None
        self._processing = False
        self._live_timer: Optional[str] = None           # after() handle

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

        # Tab 2: Heatmaps
        tab_heat = ttk.Frame(self._notebook)
        self._notebook.add(tab_heat, text="  Heatmaps  ")
        self._build_heatmaps_tab(tab_heat)

        # Tab 3: Info
        tab_info = ttk.Frame(self._notebook)
        self._notebook.add(tab_info, text="  Info  ")
        self._build_info_tab(tab_info)

    def _build_images_tab(self, parent: ttk.Frame):
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        self._canvas_orig = ImageCanvas(parent, "Original")
        self._canvas_orig.grid(row=0, column=0, sticky="nsew", padx=(4, 2), pady=4)

        self._canvas_out = ImageCanvas(parent, "Remapped")
        self._canvas_out.grid(row=0, column=1, sticky="nsew", padx=(2, 4), pady=4)

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
        self._info_text = tk.Text(
            parent, bg="#0d0d1a", fg=FG, font=FONT_MONO,
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
            "theme": {
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

    def _schedule_live(self):
        """Debounce live preview: cancel pending timer, restart 300 ms."""
        if self._live_timer:
            self.after_cancel(self._live_timer)
        if self._live_var.get():
            self._live_timer = self.after(300, self._apply)

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
            }

            # Schedule UI update on main thread
            self.after(0, self._on_pipeline_done,
                       result, overlay, dx_grid, dy_grid,
                       dx_heat, dy_heat, mag_heat, info)

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
    ):
        self._result_bgr  = result
        self._overlay_bgr = overlay
        self._dx_grid     = dx_grid
        self._dy_grid     = dy_grid

        # Update images tab
        self._canvas_out.show(overlay)

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
        ]
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
