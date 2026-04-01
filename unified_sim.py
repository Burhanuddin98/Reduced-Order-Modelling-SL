"""
Unified Room Acoustics Simulator — 22 engines, proper acoustic GUI.
PySide6 + pyqtgraph + sounddevice
Launch:  python unified_sim.py
"""
import sys, json, time, wave, struct, os
import numpy as np
from pathlib import Path
from scipy.signal import spectrogram, fftconvolve, butter, sosfiltfilt, resample

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QPushButton, QGroupBox, QGridLayout, QCheckBox,
    QDoubleSpinBox, QComboBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QTextEdit, QProgressBar, QScrollArea, QFrame, QSizePolicy,
    QSpinBox, QFileDialog, QSlider, QToolBar, QStatusBar,
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer, QSize
from PySide6.QtGui import QFont, QColor, QAction, QIcon

import pyqtgraph as pg
import pyqtgraph.opengl as gl

try:
    import sounddevice as sd
    HAS_SD = True
except ImportError:
    HAS_SD = False

from engines import (
    Room, MATERIALS, SURFACE_NAMES, SURFACE_AREAS_BOX,
    ENGINE_REGISTRY, ENGINE_COLORS, get_engine_color,
    all_metrics, schroeder_decay, octave_bandpass, OCTAVE_BANDS,
    sabine_rt60, eyring_rt60, air_absorption_coeff,
)
from room_geometry import (
    TriMesh, load_geometry, SUPPORTED_FORMATS, SHOEBOX_ONLY,
    run_engine_on_mesh,
)

# ═══════════════════════════════════════════════════════════════════
# Theme
# ═══════════════════════════════════════════════════════════════════
BG      = "#0d1117"
PANEL   = "#161b22"
CARD    = "#1c2128"
BORDER  = "#30363d"
TXT     = "#e6edf3"
DIM     = "#8b949e"
CYAN    = "#00e5ff"
GREEN   = "#3fb950"
RED     = "#f85149"
ORANGE  = "#d29922"
PINK    = "#ff79c6"
PURPLE  = "#bc8cff"

BAND_COLORS = ["#ff6b6b", "#ff922b", "#ffd43b", "#51cf66", "#339af0", "#cc5de8"]

QSS = f"""
QMainWindow, QWidget {{ background:{BG}; color:{TXT}; font-family:"Segoe UI",Consolas; font-size:13px; }}
QSplitter::handle {{ background:{BORDER}; width:3px; }}
QGroupBox {{ background:{CARD}; border:1px solid {BORDER}; border-radius:6px;
    margin-top:14px; padding:14px 8px 8px 8px; font-weight:bold; color:{CYAN}; }}
QGroupBox::title {{ subcontrol-origin:margin; left:12px; padding:0 6px; }}
QTabWidget::pane {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:6px; padding:4px; }}
QTabBar::tab {{ background:{CARD}; color:{DIM}; padding:7px 14px; margin-right:2px;
    border-top-left-radius:6px; border-top-right-radius:6px; border:1px solid {BORDER}; border-bottom:none; }}
QTabBar::tab:selected {{ background:{PANEL}; color:{CYAN}; border-bottom:2px solid {CYAN}; }}
QTableWidget {{ background:{CARD}; border:1px solid {BORDER}; gridline-color:{BORDER}; }}
QHeaderView::section {{ background:{PANEL}; color:{CYAN}; border:1px solid {BORDER}; padding:4px 6px; font-weight:bold; }}
QPushButton {{ background:{CARD}; color:{CYAN}; border:1px solid {CYAN}44; border-radius:4px; padding:6px 14px; font-weight:bold; }}
QPushButton:hover {{ background:{CYAN}22; }}
QPushButton:disabled {{ color:{DIM}; border-color:{BORDER}; }}
QPushButton#run_btn {{ background:#00e5ff22; font-size:14px; padding:10px; }}
QPushButton#run_btn:hover {{ background:#00e5ff44; }}
QDoubleSpinBox, QSpinBox, QComboBox {{ background:{CARD}; color:{TXT}; border:1px solid {BORDER};
    border-radius:4px; padding:3px 6px; }}
QComboBox::drop-down {{ border:none; }}
QComboBox QAbstractItemView {{ background:{CARD}; color:{TXT}; border:1px solid {BORDER}; }}
QTextEdit {{ background:{CARD}; border:1px solid {BORDER}; border-radius:4px; color:{TXT};
    font-family:Consolas; font-size:11px; padding:4px; }}
QProgressBar {{ background:{CARD}; border:1px solid {BORDER}; border-radius:4px; text-align:center; color:{TXT}; height:18px; }}
QProgressBar::chunk {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {CYAN},stop:1 {GREEN}); border-radius:3px; }}
QCheckBox {{ spacing:5px; }} QCheckBox::indicator {{ width:15px; height:15px; }}
QScrollArea {{ background:transparent; border:none; }}
QLabel {{ color:{TXT}; }}
QToolBar {{ background:{PANEL}; border-bottom:1px solid {BORDER}; spacing:6px; padding:4px; }}
QStatusBar {{ background:{PANEL}; border-top:1px solid {BORDER}; color:{DIM}; }}
QSlider::groove:horizontal {{ background:{BORDER}; height:4px; border-radius:2px; }}
QSlider::handle:horizontal {{ background:{CYAN}; width:14px; margin:-5px 0; border-radius:7px; }}
"""

pg.setConfigOptions(background=PANEL, foreground=TXT)


# ═══════════════════════════════════════════════════════════════════
# Worker thread
# ═══════════════════════════════════════════════════════════════════
class EngineWorker(QThread):
    finished = Signal(str, object)   # name, Result
    error = Signal(str, str)

    def __init__(self, name, func, room, src, rec, sr, duration,
                 mesh=None, parent=None):
        super().__init__(parent)
        self.name = name; self.func = func
        self.room = room; self.src = src; self.rec = rec
        self.sr = sr; self.duration = duration; self.mesh = mesh

    def run(self):
        try:
            if self.mesh is not None:
                result = run_engine_on_mesh(self.name, self.mesh,
                                            self.src, self.rec,
                                            sr=self.sr, duration=self.duration)
            else:
                result = self.func(self.room, self.src, self.rec,
                                   sr=self.sr, duration=self.duration)
            self.finished.emit(self.name, result)
        except Exception as e:
            self.error.emit(self.name, str(e))


# ═══════════════════════════════════════════════════════════════════
# Room setup panel
# ═══════════════════════════════════════════════════════════════════
class RoomSetup(QGroupBox):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__("Room Geometry", parent)
        g = QGridLayout(self)
        g.setSpacing(5)
        row = 0
        self.dim_spins = {}
        for lbl, val in [("Length (m)", 8.4), ("Width (m)", 6.7), ("Height (m)", 3.0)]:
            g.addWidget(QLabel(lbl), row, 0)
            sp = QDoubleSpinBox(); sp.setRange(1, 50); sp.setValue(val)
            sp.setSingleStep(0.1); sp.setDecimals(1)
            sp.valueChanged.connect(self.changed.emit)
            self.dim_spins[lbl[0]] = sp
            g.addWidget(sp, row, 1)
            row += 1
        # volume label
        self.vol_label = QLabel()
        self.vol_label.setStyleSheet(f"color:{DIM}; font-size:11px;")
        g.addWidget(self.vol_label, row, 0, 1, 2); row += 1
        self._update_vol()
        for sp in self.dim_spins.values():
            sp.valueChanged.connect(self._update_vol)

    def _update_vol(self):
        L, W, H = self.dims()
        V = L * W * H
        S = 2 * (L*W + W*H + L*H)
        self.vol_label.setText(f"V = {V:.1f} m\u00b3  |  S = {S:.1f} m\u00b2")

    def dims(self):
        return (self.dim_spins["L"].value(), self.dim_spins["W"].value(),
                self.dim_spins["H"].value())


class MaterialSetup(QGroupBox):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__("Surface Materials", parent)
        g = QGridLayout(self)
        g.setSpacing(4)
        mat_names = sorted(MATERIALS.keys())
        self.combos = {}
        defaults = {"floor": "linoleum", "ceiling": "acoustic_tile", "left": "plaster",
                     "right": "plaster", "front": "plaster", "back": "glass"}
        for i, sn in enumerate(SURFACE_NAMES):
            g.addWidget(QLabel(sn.capitalize()), i, 0)
            cb = QComboBox(); cb.addItems(mat_names)
            cb.setCurrentText(defaults.get(sn, "plaster"))
            cb.currentTextChanged.connect(self.changed.emit)
            self.combos[sn] = cb
            g.addWidget(cb, i, 1)
        # absorption preview
        self.alpha_plot = pg.PlotWidget()
        self.alpha_plot.setFixedHeight(100)
        self.alpha_plot.showGrid(y=True, alpha=0.1)
        self.alpha_plot.setLabel("bottom", "Hz")
        self.alpha_plot.setLabel("left", "\u03b1")
        self.alpha_plot.setYRange(0, 1)
        g.addWidget(self.alpha_plot, len(SURFACE_NAMES), 0, 1, 2)
        for cb in self.combos.values():
            cb.currentTextChanged.connect(self._update_alpha_plot)
        self._update_alpha_plot()

    def _update_alpha_plot(self):
        self.alpha_plot.clear()
        for sn, cb in self.combos.items():
            mat = cb.currentText()
            alpha = MATERIALS.get(mat, MATERIALS["plaster"])
            col = {"floor": RED, "ceiling": ORANGE, "left": GREEN,
                   "right": CYAN, "front": PINK, "back": PURPLE}.get(sn, TXT)
            self.alpha_plot.plot(OCTAVE_BANDS, alpha, pen=pg.mkPen(col, width=1.5),
                                symbol='o', symbolSize=5, symbolBrush=col)

    def surfaces(self):
        return {sn: cb.currentText() for sn, cb in self.combos.items()}


class SourceRecSetup(QGroupBox):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__("Source / Receiver", parent)
        g = QGridLayout(self)
        g.setSpacing(4)
        g.addWidget(QLabel("Source"), 0, 0, 1, 4)
        self.src_spins = []
        for i, (lbl, val) in enumerate([("x", 2.0), ("y", 3.0), ("z", 1.5)]):
            g.addWidget(QLabel(lbl), 1, i * 2)
            sp = QDoubleSpinBox(); sp.setRange(0.1, 49); sp.setValue(val); sp.setDecimals(2)
            sp.valueChanged.connect(self.changed.emit)
            self.src_spins.append(sp)
            g.addWidget(sp, 1, i * 2 + 1)
        g.addWidget(QLabel("Receiver"), 2, 0, 1, 4)
        self.rec_spins = []
        for i, (lbl, val) in enumerate([("x", 6.0), ("y", 2.0), ("z", 1.2)]):
            g.addWidget(QLabel(lbl), 3, i * 2)
            sp = QDoubleSpinBox(); sp.setRange(0.1, 49); sp.setValue(val); sp.setDecimals(2)
            sp.valueChanged.connect(self.changed.emit)
            self.rec_spins.append(sp)
            g.addWidget(sp, 3, i * 2 + 1)
        # distance label
        self.dist_label = QLabel()
        self.dist_label.setStyleSheet(f"color:{DIM}; font-size:11px;")
        g.addWidget(self.dist_label, 4, 0, 1, 6)
        self.changed.connect(self._update_dist)
        self._update_dist()

    def _update_dist(self):
        s, r = np.array(self.src()), np.array(self.rec())
        d = np.linalg.norm(s - r)
        t_direct = d / 343.0
        self.dist_label.setText(f"d = {d:.2f} m  |  t_direct = {t_direct*1000:.1f} ms")

    def src(self): return tuple(sp.value() for sp in self.src_spins)
    def rec(self): return tuple(sp.value() for sp in self.rec_spins)


# ═══════════════════════════════════════════════════════════════════
# Geometry import panel
# ═══════════════════════════════════════════════════════════════════
class GeometryPanel(QGroupBox):
    mesh_loaded = Signal(object)   # emits TriMesh or None
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__("Geometry Import", parent)
        lay = QVBoxLayout(self)
        lay.setSpacing(4)

        # mode selector
        mode_row = QHBoxLayout()
        self.btn_shoebox = QPushButton("Shoebox")
        self.btn_shoebox.setCheckable(True)
        self.btn_shoebox.setChecked(True)
        self.btn_shoebox.clicked.connect(lambda: self._set_mode("shoebox"))
        self.btn_import = QPushButton("Import Model")
        self.btn_import.setCheckable(True)
        self.btn_import.clicked.connect(lambda: self._set_mode("import"))
        mode_row.addWidget(self.btn_shoebox)
        mode_row.addWidget(self.btn_import)
        lay.addLayout(mode_row)

        # file loader
        file_row = QHBoxLayout()
        self.btn_load = QPushButton("Load File...")
        self.btn_load.clicked.connect(self._load_file)
        self.btn_load.setEnabled(False)
        file_row.addWidget(self.btn_load)
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet(f"color:{DIM}; font-size:11px;")
        self.file_label.setWordWrap(True)
        file_row.addWidget(self.file_label, 1)
        lay.addLayout(file_row)

        # mesh info
        self.info_label = QLabel("")
        self.info_label.setStyleSheet(f"color:{DIM}; font-size:11px;")
        self.info_label.setWordWrap(True)
        lay.addWidget(self.info_label)

        # unit selector
        unit_row = QHBoxLayout()
        unit_row.addWidget(QLabel("Units:"))
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["m", "mm", "cm", "in", "ft"])
        self.unit_combo.setCurrentText("m")
        self.unit_combo.currentTextChanged.connect(self._rescale)
        self.unit_combo.setEnabled(False)
        unit_row.addWidget(self.unit_combo)
        lay.addLayout(unit_row)

        # surface group material assignment
        self.surface_scroll = QScrollArea()
        self.surface_scroll.setWidgetResizable(True)
        self.surface_scroll.setMaximumHeight(200)
        self.surface_container = QWidget()
        self.surface_layout = QVBoxLayout(self.surface_container)
        self.surface_layout.setSpacing(2)
        self.surface_layout.setContentsMargins(2, 2, 2, 2)
        self.surface_scroll.setWidget(self.surface_container)
        self.surface_scroll.setVisible(False)
        lay.addWidget(self.surface_scroll)

        self._mesh = None
        self._original_vertices = None
        self._mat_combos = {}

    @property
    def mesh(self):
        return self._mesh

    @property
    def mode(self):
        return "shoebox" if self.btn_shoebox.isChecked() else "import"

    def _set_mode(self, mode):
        self.btn_shoebox.setChecked(mode == "shoebox")
        self.btn_import.setChecked(mode == "import")
        self.btn_load.setEnabled(mode == "import")
        self.unit_combo.setEnabled(mode == "import" and self._mesh is not None)
        self.surface_scroll.setVisible(mode == "import" and self._mesh is not None)
        if mode == "shoebox":
            self._mesh = None
            self.mesh_loaded.emit(None)
        self.changed.emit()

    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Room Geometry", "", SUPPORTED_FORMATS)
        if not path:
            return
        try:
            mesh = load_geometry(path)
            mesh.move_floor_to_z0()
            self._original_vertices = mesh.vertices.copy()
            self._mesh = mesh
            self._update_info()
            self._build_surface_combos()
            self.unit_combo.setEnabled(True)
            self.surface_scroll.setVisible(True)
            self.file_label.setText(Path(path).name)
            self.mesh_loaded.emit(mesh)
            self.changed.emit()
        except Exception as e:
            self.file_label.setText(f"Error: {e}")
            self.info_label.setText("")

    def _update_info(self):
        if self._mesh is None: return
        m = self._mesh
        d = m.dims()
        try:
            vol = m.volume()
        except Exception:
            vol = d[0] * d[1] * d[2]
        sa = m.surface_area()
        self.info_label.setText(
            f"{len(m.faces)} faces  |  {len(m.face_groups)} surfaces\n"
            f"{d[0]:.2f} x {d[1]:.2f} x {d[2]:.2f} m\n"
            f"V = {vol:.1f} m\u00b3  |  S = {sa:.1f} m\u00b2")

    def _rescale(self, unit):
        if self._mesh is None or self._original_vertices is None: return
        factors = {"m": 1.0, "mm": 0.001, "cm": 0.01, "in": 0.0254, "ft": 0.3048}
        self._mesh.vertices = self._original_vertices * factors.get(unit, 1.0)
        self._mesh.move_floor_to_z0()
        self._update_info()
        self.mesh_loaded.emit(self._mesh)
        self.changed.emit()

    def _build_surface_combos(self):
        # clear old
        while self.surface_layout.count():
            item = self.surface_layout.takeAt(0)
            w = item.widget()
            if w: w.deleteLater()
        self._mat_combos.clear()
        if self._mesh is None: return
        mat_names = sorted(MATERIALS.keys())
        areas = self._mesh.face_areas()
        for gname in self._mesh.face_groups:
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(4)
            area = float(areas[self._mesh.face_groups[gname]].sum())
            lbl = QLabel(f"{gname} ({area:.1f}m\u00b2)")
            lbl.setStyleSheet(f"color:{TXT}; font-size:11px;")
            lbl.setFixedWidth(140)
            rl.addWidget(lbl)
            cb = QComboBox()
            cb.addItems(mat_names)
            cb.setCurrentText(self._mesh.materials.get(gname, "plaster"))
            cb.currentTextChanged.connect(lambda mat, g=gname: self._on_mat_changed(g, mat))
            rl.addWidget(cb)
            self._mat_combos[gname] = cb
            self.surface_layout.addWidget(row)

    def _on_mat_changed(self, group, material):
        if self._mesh:
            self._mesh.materials[group] = material
            self.changed.emit()


# ═══════════════════════════════════════════════════════════════════
# 3D Room view with ray paths
# ═══════════════════════════════════════════════════════════════════
class RoomView3D(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCameraPosition(distance=18, elevation=25, azimuth=45)
        self.setBackgroundColor(BG)
        grid = gl.GLGridItem()
        grid.setSize(20, 20); grid.setSpacing(1, 1)
        grid.setColor((60, 60, 60, 40))
        self.addItem(grid)
        self._items = []

    def _clear_items(self):
        for it in self._items:
            self.removeItem(it)
        self._items.clear()

    def update_room(self, Lx, Ly, Lz, src, rec):
        self._clear_items()
        # wireframe
        V = np.array([[0,0,0],[Lx,0,0],[Lx,Ly,0],[0,Ly,0],
                       [0,0,Lz],[Lx,0,Lz],[Lx,Ly,Lz],[0,Ly,Lz]], dtype=float)
        edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],
                 [0,4],[1,5],[2,6],[3,7]]
        pts = []
        for a, b in edges:
            pts.append(V[a]); pts.append(V[b])
        box = gl.GLLinePlotItem(pos=np.array(pts), color=(0,0.9,1,0.5), width=2, mode='lines')
        self.addItem(box); self._items.append(box)
        # floor quad
        floor_verts = np.array([[0,0,0],[Lx,0,0],[Lx,Ly,0],[0,Ly,0]], dtype=float)
        floor_faces = np.array([[0,1,2],[0,2,3]])
        floor_colors = np.array([[0.1,0.15,0.2,0.3]]*2)
        fm = gl.GLMeshItem(vertexes=floor_verts, faces=floor_faces, faceColors=floor_colors,
                           smooth=False, drawEdges=False)
        self.addItem(fm); self._items.append(fm)
        # source (red sphere)
        smd = gl.MeshData.sphere(rows=10, cols=10, radius=0.18)
        sm = gl.GLMeshItem(meshdata=smd, smooth=True, color=(1,0.25,0.15,1), shader='shaded')
        sm.translate(*src)
        self.addItem(sm); self._items.append(sm)
        # source label line
        sl = gl.GLLinePlotItem(pos=np.array([[*src],[src[0],src[1],src[2]+0.5]]),
                               color=(1,0.3,0.2,0.6), width=2)
        self.addItem(sl); self._items.append(sl)
        # receiver (blue sphere)
        rmd = gl.MeshData.sphere(rows=10, cols=10, radius=0.18)
        rm = gl.GLMeshItem(meshdata=rmd, smooth=True, color=(0.2,0.5,1,1), shader='shaded')
        rm.translate(*rec)
        self.addItem(rm); self._items.append(rm)
        # direct sound line
        direct = gl.GLLinePlotItem(pos=np.array([[*src],[*rec]]),
                                   color=(1,1,0,0.3), width=1.5)
        self.addItem(direct); self._items.append(direct)

    def show_rays(self, room_dims, src, n_rays=50, max_bounces=8):
        """Visualize a few ray paths in 3D."""
        from engines import box_exit, reflect_specular
        dims = np.array(room_dims)
        rng = np.random.default_rng(42)
        colors_list = [
            (1,0.3,0.3,0.25), (0.3,1,0.3,0.25), (0.3,0.3,1,0.25),
            (1,1,0.3,0.25), (1,0.3,1,0.25), (0.3,1,1,0.25),
        ]
        for i in range(n_rays):
            z = rng.uniform(-1, 1)
            phi = rng.uniform(0, 2*np.pi)
            r = np.sqrt(1 - z**2)
            d = np.array([r*np.cos(phi), r*np.sin(phi), z])
            pos = np.array(src, dtype=float)
            path = [pos.copy()]
            for b in range(max_bounces):
                origins = pos.reshape(1, 3)
                dirs = d.reshape(1, 3)
                t_hit, walls = box_exit(origins, dirs, dims)
                t = t_hit[0]
                if t > 1e6: break
                pos = pos + d * t
                path.append(pos.copy())
                w = walls[0]
                ax = w // 2
                d[ax] *= -1
            path = np.array(path)
            col = colors_list[i % len(colors_list)]
            line = gl.GLLinePlotItem(pos=path, color=col, width=1.2)
            self.addItem(line)
            self._items.append(line)

    def show_mesh(self, mesh, src, rec):
        """Render a TriMesh with per-group coloring + source/receiver."""
        self._clear_items()
        GROUP_COLORS = [
            (0.2, 0.6, 0.8, 0.35), (0.8, 0.3, 0.3, 0.35), (0.3, 0.8, 0.3, 0.35),
            (0.8, 0.8, 0.3, 0.35), (0.6, 0.3, 0.8, 0.35), (0.8, 0.5, 0.2, 0.35),
            (0.3, 0.8, 0.8, 0.35), (0.8, 0.3, 0.6, 0.35), (0.5, 0.8, 0.3, 0.35),
        ]
        # render each surface group with a different color
        verts = mesh.vertices.astype(np.float32)
        for gi, (gname, fidx) in enumerate(mesh.face_groups.items()):
            if len(fidx) == 0: continue
            group_faces = mesh.faces[fidx]
            col = GROUP_COLORS[gi % len(GROUP_COLORS)]
            face_colors = np.tile(np.array(col), (len(group_faces), 1)).astype(np.float32)
            mi = gl.GLMeshItem(vertexes=verts, faces=group_faces,
                               faceColors=face_colors, smooth=False,
                               drawEdges=True, edgeColor=(0.5, 0.5, 0.5, 0.15))
            self.addItem(mi)
            self._items.append(mi)
        # source
        smd = gl.MeshData.sphere(rows=10, cols=10, radius=0.18)
        sm = gl.GLMeshItem(meshdata=smd, smooth=True, color=(1,0.25,0.15,1), shader='shaded')
        sm.translate(*src)
        self.addItem(sm); self._items.append(sm)
        # receiver
        rmd = gl.MeshData.sphere(rows=10, cols=10, radius=0.18)
        rm = gl.GLMeshItem(meshdata=rmd, smooth=True, color=(0.2,0.5,1,1), shader='shaded')
        rm.translate(*rec)
        self.addItem(rm); self._items.append(rm)
        # direct line
        direct = gl.GLLinePlotItem(pos=np.array([[*src],[*rec]]),
                                   color=(1,1,0,0.3), width=1.5)
        self.addItem(direct); self._items.append(direct)
        # auto-fit camera
        bb_min, bb_max = mesh.bounding_box()
        center = (bb_min + bb_max) / 2
        extent = np.linalg.norm(bb_max - bb_min)
        self.setCameraPosition(distance=extent * 1.5, elevation=25, azimuth=45)
        self.pan(center[0], center[1], center[2])

    def show_rays_mesh(self, mesh, src, n_rays=30, max_bounces=6):
        """Visualize ray paths on a TriMesh."""
        from room_geometry import _reflect_off_surface
        rng = np.random.default_rng(42)
        colors_list = [
            (1,0.3,0.3,0.2), (0.3,1,0.3,0.2), (0.3,0.3,1,0.2),
            (1,1,0.3,0.2), (1,0.3,1,0.2), (0.3,1,1,0.2),
        ]
        for i in range(n_rays):
            z = rng.uniform(-1, 1)
            phi = rng.uniform(0, 2*np.pi)
            r_s = np.sqrt(1 - z**2)
            d = np.array([r_s*np.cos(phi), r_s*np.sin(phi), z])
            pos = np.array(src, dtype=float)
            path = [pos.copy()]
            for b in range(max_bounces):
                t_hit, fhit, normals = mesh.ray_intersect(
                    pos.reshape(1, 3), d.reshape(1, 3))
                if fhit[0] < 0 or t_hit[0] > 1e6: break
                pos = pos + d * t_hit[0]
                path.append(pos.copy())
                d = d - 2 * np.dot(d, normals[0]) * normals[0]
            if len(path) > 1:
                path = np.array(path)
                col = colors_list[i % len(colors_list)]
                line = gl.GLLinePlotItem(pos=path, color=col, width=1)
                self.addItem(line)
                self._items.append(line)


# ═══════════════════════════════════════════════════════════════════
# IR Waveform + Spectrogram display
# ═══════════════════════════════════════════════════════════════════
class IRDisplay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        # waveform
        self.pw_ir = pg.PlotWidget(title="Impulse Response")
        self.pw_ir.showGrid(x=True, y=True, alpha=0.12)
        self.pw_ir.setLabel("bottom", "Time", "s")
        self.pw_ir.setLabel("left", "Amplitude")
        lay.addWidget(self.pw_ir, 2)

        # spectrogram
        self.pw_spec = pg.PlotWidget(title="Spectrogram")
        self.pw_spec.setLabel("bottom", "Time", "s")
        self.pw_spec.setLabel("left", "Frequency", "Hz")
        self.spec_img = pg.ImageItem()
        self.pw_spec.addItem(self.spec_img)
        # colorbar
        self.colorbar = pg.ColorBarItem(
            values=(-80, 0), colorMap=pg.colormap.get('magma'),
            label="dB"
        )
        self.colorbar.setImageItem(self.spec_img)
        lay.addWidget(self.pw_spec, 2)

    def plot(self, ir, sr):
        if ir is None or len(ir) < 10: return
        t = np.arange(len(ir)) / sr
        step = max(1, len(ir) // 30000)
        self.pw_ir.clear()
        self.pw_ir.plot(t[::step], ir[::step], pen=pg.mkPen(CYAN, width=1))
        # envelope
        env = np.abs(ir)
        from scipy.ndimage import maximum_filter1d
        env = maximum_filter1d(env, max(int(0.002 * sr), 1))
        self.pw_ir.plot(t[::step], env[::step], pen=pg.mkPen(PINK, width=0.8))

        # spectrogram
        nperseg = min(1024, len(ir) // 4)
        if nperseg < 32: return
        f, t_spec, Sxx = spectrogram(ir, sr, nperseg=nperseg, noverlap=nperseg // 2)
        Sxx_db = 10 * np.log10(Sxx + 1e-30)
        Sxx_db = np.clip(Sxx_db, -80, 0)
        self.spec_img.setImage(Sxx_db.T, autoLevels=False)
        self.spec_img.setRect(0, 0, t_spec[-1], f[-1])
        self.pw_spec.setXRange(0, t_spec[-1])
        self.pw_spec.setYRange(0, min(f[-1], 8000))


# ═══════════════════════════════════════════════════════════════════
# Decay analysis — Schroeder + per-band RT60
# ═══════════════════════════════════════════════════════════════════
class DecayDisplay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        # Schroeder decay (broadband + per-band)
        self.pw_edc = pg.PlotWidget(title="Schroeder Decay Curves (per octave band)")
        self.pw_edc.showGrid(x=True, y=True, alpha=0.12)
        self.pw_edc.setLabel("bottom", "Time", "s")
        self.pw_edc.setLabel("left", "Level", "dB")
        self.pw_edc.setYRange(-80, 0)
        self.pw_edc.addLegend(offset=(60, 10))
        lay.addWidget(self.pw_edc, 3)

        # RT60 bar chart per band
        self.pw_rt = pg.PlotWidget(title="Reverberation Time per Octave Band")
        self.pw_rt.showGrid(y=True, alpha=0.12)
        self.pw_rt.setLabel("bottom", "Frequency", "Hz")
        self.pw_rt.setLabel("left", "T30", "s")
        lay.addWidget(self.pw_rt, 2)

    def plot(self, ir, sr):
        if ir is None or len(ir) < 100: return
        self.pw_edc.clear()
        self.pw_rt.clear()
        # broadband
        edc = schroeder_decay(ir, sr)
        t = np.arange(len(ir)) / sr
        step = max(1, len(ir) // 10000)
        self.pw_edc.plot(t[::step], edc[::step], pen=pg.mkPen(TXT, width=2), name="Broadband")
        # per-band
        t30_bands = []
        bands = [125, 250, 500, 1000, 2000, 4000]
        for i, fc in enumerate(bands):
            try:
                filtered = octave_bandpass(ir, sr, fc)
                edc_band = schroeder_decay(filtered, sr)
                col = BAND_COLORS[i % len(BAND_COLORS)]
                self.pw_edc.plot(t[::step], edc_band[::step],
                                pen=pg.mkPen(col, width=1.2), name=f"{fc} Hz")
                from engines import compute_t30
                t30, _ = compute_t30(filtered, sr)
                t30_bands.append(t30)
            except Exception:
                t30_bands.append(0)
        # bar chart
        x = np.arange(len(bands))
        colors = [pg.mkBrush(c + "BB") for c in BAND_COLORS]
        for i in range(len(bands)):
            bg = pg.BarGraphItem(x=[x[i]], height=[t30_bands[i]], width=0.6,
                                 brush=colors[i], pen=pg.mkPen(BAND_COLORS[i], width=1))
            self.pw_rt.addItem(bg)
        ticks = [(i, f"{fc}") for i, fc in enumerate(bands)]
        self.pw_rt.getAxis("bottom").setTicks([ticks])


# ═══════════════════════════════════════════════════════════════════
# Frequency analysis — spectrum + octave band energy
# ═══════════════════════════════════════════════════════════════════
class FrequencyDisplay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.pw_fft = pg.PlotWidget(title="Magnitude Spectrum")
        self.pw_fft.showGrid(x=True, y=True, alpha=0.12)
        self.pw_fft.setLabel("bottom", "Frequency", "Hz")
        self.pw_fft.setLabel("left", "Magnitude", "dB")
        self.pw_fft.setLogMode(x=True, y=False)
        lay.addWidget(self.pw_fft, 3)

        self.pw_energy = pg.PlotWidget(title="Octave Band Energy (dB)")
        self.pw_energy.showGrid(y=True, alpha=0.12)
        self.pw_energy.setLabel("bottom", "Hz")
        self.pw_energy.setLabel("left", "Energy", "dB")
        lay.addWidget(self.pw_energy, 2)

    def plot(self, ir, sr):
        if ir is None or len(ir) < 100: return
        self.pw_fft.clear()
        self.pw_energy.clear()
        # FFT
        N = len(ir)
        fft_mag = np.abs(np.fft.rfft(ir))
        freqs = np.fft.rfftfreq(N, 1.0 / sr)
        fft_db = 20 * np.log10(fft_mag + 1e-30)
        # smooth
        from scipy.ndimage import uniform_filter1d
        step = max(1, len(freqs) // 5000)
        fft_smooth = uniform_filter1d(fft_db, max(int(len(fft_db) / 500), 1))
        mask = freqs > 10
        self.pw_fft.plot(freqs[mask][::step], fft_smooth[mask][::step],
                         pen=pg.mkPen(CYAN, width=1.2))
        self.pw_fft.setXRange(np.log10(20), np.log10(min(sr / 2, 20000)))

        # octave band energy
        bands = [125, 250, 500, 1000, 2000, 4000]
        energies = []
        for fc in bands:
            try:
                filtered = octave_bandpass(ir, sr, fc)
                e = 10 * np.log10(np.sum(filtered ** 2) + 1e-30)
                energies.append(e)
            except Exception:
                energies.append(-80)
        x = np.arange(len(bands))
        for i in range(len(bands)):
            bg = pg.BarGraphItem(x=[x[i]], height=[energies[i]], width=0.6,
                                 brush=pg.mkBrush(BAND_COLORS[i] + "BB"),
                                 pen=pg.mkPen(BAND_COLORS[i], width=1))
            self.pw_energy.addItem(bg)
        ticks = [(i, f"{fc}") for i, fc in enumerate(bands)]
        self.pw_energy.getAxis("bottom").setTicks([ticks])


# ═══════════════════════════════════════════════════════════════════
# Auralization — convolve IR with audio, play back
# ═══════════════════════════════════════════════════════════════════
class AuralizationPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        # controls
        ctrl = QHBoxLayout()
        self.btn_load = QPushButton("Load Audio")
        self.btn_load.clicked.connect(self._load_audio)
        ctrl.addWidget(self.btn_load)
        self.audio_label = QLabel("No audio loaded")
        self.audio_label.setStyleSheet(f"color:{DIM};")
        ctrl.addWidget(self.audio_label, 1)
        lay.addLayout(ctrl)

        ctrl2 = QHBoxLayout()
        self.btn_convolve = QPushButton("Convolve")
        self.btn_convolve.clicked.connect(self._convolve)
        self.btn_convolve.setEnabled(False)
        ctrl2.addWidget(self.btn_convolve)
        self.btn_play_dry = QPushButton("Play Dry")
        self.btn_play_dry.clicked.connect(self._play_dry)
        self.btn_play_dry.setEnabled(False)
        ctrl2.addWidget(self.btn_play_dry)
        self.btn_play_wet = QPushButton("Play Wet (Auralized)")
        self.btn_play_wet.clicked.connect(self._play_wet)
        self.btn_play_wet.setEnabled(False)
        ctrl2.addWidget(self.btn_play_wet)
        self.btn_play_ir = QPushButton("Play IR")
        self.btn_play_ir.clicked.connect(self._play_ir)
        ctrl2.addWidget(self.btn_play_ir)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._stop)
        ctrl2.addWidget(self.btn_stop)
        self.btn_save = QPushButton("Save WAV")
        self.btn_save.clicked.connect(self._save)
        self.btn_save.setEnabled(False)
        ctrl2.addWidget(self.btn_save)
        lay.addLayout(ctrl2)

        # waveforms
        self.pw_dry = pg.PlotWidget(title="Dry Signal")
        self.pw_dry.showGrid(x=True, y=True, alpha=0.1)
        self.pw_dry.setLabel("bottom", "Time", "s")
        lay.addWidget(self.pw_dry, 1)

        self.pw_wet = pg.PlotWidget(title="Auralized Signal")
        self.pw_wet.showGrid(x=True, y=True, alpha=0.1)
        self.pw_wet.setLabel("bottom", "Time", "s")
        lay.addWidget(self.pw_wet, 1)

        self._dry = None; self._dry_sr = 44100
        self._wet = None
        self._ir = None; self._ir_sr = 44100

    def set_ir(self, ir, sr):
        self._ir = ir; self._ir_sr = sr
        self.btn_convolve.setEnabled(self._dry is not None and ir is not None)

    def _load_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Audio", "",
                                               "Audio (*.wav *.WAV)")
        if not path: return
        import wave as wavmod
        with wavmod.open(path, 'rb') as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            raw = wf.readframes(n)
        if sw == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768
        elif sw == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648
        else:
            data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64) / 128 - 1
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1)
        self._dry = data; self._dry_sr = sr
        self.audio_label.setText(f"{Path(path).name}  |  {sr} Hz  |  {len(data)/sr:.1f}s")
        self.btn_play_dry.setEnabled(HAS_SD)
        self.btn_convolve.setEnabled(self._ir is not None)
        # plot
        t = np.arange(len(data)) / sr
        step = max(1, len(data) // 20000)
        self.pw_dry.clear()
        self.pw_dry.plot(t[::step], data[::step], pen=pg.mkPen(DIM, width=0.8))

    def _convolve(self):
        if self._dry is None or self._ir is None: return
        # resample IR to match dry audio SR if needed
        ir = self._ir
        if self._ir_sr != self._dry_sr:
            target_len = int(len(ir) * self._dry_sr / self._ir_sr)
            ir = resample(ir, target_len)
        self._wet = fftconvolve(self._dry, ir)
        self._wet /= np.max(np.abs(self._wet)) + 1e-10
        t = np.arange(len(self._wet)) / self._dry_sr
        step = max(1, len(self._wet) // 20000)
        self.pw_wet.clear()
        self.pw_wet.plot(t[::step], self._wet[::step], pen=pg.mkPen(CYAN, width=0.8))
        self.btn_play_wet.setEnabled(HAS_SD)
        self.btn_save.setEnabled(True)

    def _play_dry(self):
        if HAS_SD and self._dry is not None:
            sd.play(self._dry.astype(np.float32), self._dry_sr)

    def _play_wet(self):
        if HAS_SD and self._wet is not None:
            sd.play(self._wet.astype(np.float32), self._dry_sr)

    def _play_ir(self):
        if HAS_SD and self._ir is not None:
            ir_play = self._ir / (np.max(np.abs(self._ir)) + 1e-10) * 0.8
            sd.play(ir_play.astype(np.float32), self._ir_sr)

    def _stop(self):
        if HAS_SD: sd.stop()

    def _save(self):
        if self._wet is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Auralized", "auralized.wav",
                                               "WAV (*.wav)")
        if not path: return
        import wave as wavmod
        data16 = (self._wet * 32767).astype(np.int16)
        with wavmod.open(path, 'w') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(self._dry_sr)
            wf.writeframes(data16.tobytes())


# ═══════════════════════════════════════════════════════════════════
# Metrics summary bar
# ═══════════════════════════════════════════════════════════════════
class MetricsBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet(f"background:{CARD}; border:1px solid {BORDER}; border-radius:6px; padding:6px;")
        self.lay = QHBoxLayout(self)
        self.lay.setSpacing(20)
        self.labels = {}
        for key, name, unit in [
            ("T30_s", "T30", "s"), ("EDT_s", "EDT", "s"), ("C80_dB", "C80", "dB"),
            ("D50", "D50", ""), ("TS_ms", "TS", "ms")]:
            box = QVBoxLayout()
            val_lbl = QLabel("—")
            val_lbl.setFont(QFont("Consolas", 20, QFont.Bold))
            val_lbl.setStyleSheet(f"color:{CYAN};")
            val_lbl.setAlignment(Qt.AlignCenter)
            name_lbl = QLabel(f"{name} ({unit})" if unit else name)
            name_lbl.setStyleSheet(f"color:{DIM}; font-size:11px;")
            name_lbl.setAlignment(Qt.AlignCenter)
            box.addWidget(val_lbl)
            box.addWidget(name_lbl)
            self.lay.addLayout(box)
            self.labels[key] = val_lbl

    def update_metrics(self, metrics):
        for key, lbl in self.labels.items():
            val = metrics.get(key, "—")
            if isinstance(val, float):
                lbl.setText(f"{val:.3f}" if val < 10 else f"{val:.1f}")
            else:
                lbl.setText(str(val))


# ═══════════════════════════════════════════════════════════════════
# Engine comparison tab
# ═══════════════════════════════════════════════════════════════════
class ComparisonPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        top = QHBoxLayout()
        self.pw_ir_cmp = pg.PlotWidget(title="IR Overlay")
        self.pw_ir_cmp.showGrid(x=True, y=True, alpha=0.1)
        self.pw_ir_cmp.addLegend(offset=(10, 10))
        self.pw_edc_cmp = pg.PlotWidget(title="Schroeder Decay Overlay")
        self.pw_edc_cmp.showGrid(x=True, y=True, alpha=0.1)
        self.pw_edc_cmp.setYRange(-80, 0)
        self.pw_edc_cmp.addLegend(offset=(10, 10))
        top.addWidget(self.pw_ir_cmp)
        top.addWidget(self.pw_edc_cmp)
        lay.addLayout(top, 2)

        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(
            ["Engine", "Category", "T30 (s)", "EDT (s)", "C80 (dB)", "D50", "TS (ms)", "Time (s)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(True)
        lay.addWidget(self.table, 1)

        self._results = {}

    def clear(self):
        self._results.clear()
        self.pw_ir_cmp.clear(); self.pw_edc_cmp.clear()
        self.table.setRowCount(0)

    def add_result(self, name, result):
        self._results[name] = result
        col = get_engine_color(name)
        if result.ir is not None and len(result.ir) > 10:
            t = np.arange(len(result.ir)) / result.sr
            step = max(1, len(result.ir) // 10000)
            self.pw_ir_cmp.plot(t[::step], result.ir[::step],
                                pen=pg.mkPen(col, width=1), name=name)
            edc = schroeder_decay(result.ir, result.sr)
            self.pw_edc_cmp.plot(t[::step], edc[::step],
                                 pen=pg.mkPen(col, width=1.2), name=name)
        row = self.table.rowCount()
        self.table.insertRow(row)
        m = result.metrics
        vals = [name, result.category,
                f"{m.get('T30_s','—')}", f"{m.get('EDT_s','—')}",
                f"{m.get('C80_dB','—')}", f"{m.get('D50','—')}",
                f"{m.get('TS_ms','—')}", f"{result.compute_time:.3f}"]
        for c, v in enumerate(vals):
            item = QTableWidgetItem(str(v))
            if c == 0:
                item.setForeground(QColor(col))
                item.setFont(QFont("Segoe UI", 11, QFont.Bold))
            self.table.setItem(row, c, item)


# ═══════════════════════════════════════════════════════════════════
# Main window
# ═══════════════════════════════════════════════════════════════════
class UnifiedSim(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Unified Room Acoustics Simulator — 22 Engines")
        self.resize(1600, 1000)
        self._worker = None
        self._current_result = None
        self._all_results = {}

        # toolbar
        tb = QToolBar("Main")
        tb.setIconSize(QSize(20, 20))
        tb.setMovable(False)
        self.addToolBar(tb)

        tb.addWidget(QLabel(" Engine: "))
        self.engine_combo = QComboBox()
        self.engine_combo.setMinimumWidth(180)
        for name, info in ENGINE_REGISTRY.items():
            self.engine_combo.addItem(f"{name}  [{info['cat']}]", name)
        tb.addWidget(self.engine_combo)

        tb.addSeparator()
        tb.addWidget(QLabel(" Duration: "))
        self.dur_spin = QDoubleSpinBox()
        self.dur_spin.setRange(0.1, 5.0); self.dur_spin.setValue(2.0); self.dur_spin.setSingleStep(0.1)
        tb.addWidget(self.dur_spin)

        tb.addSeparator()
        self.btn_run = QPushButton("  RUN  ")
        self.btn_run.setObjectName("run_btn")
        self.btn_run.clicked.connect(self._run_single)
        tb.addWidget(self.btn_run)

        self.btn_run_all = QPushButton(" Run All 22 ")
        self.btn_run_all.clicked.connect(self._run_all)
        tb.addWidget(self.btn_run_all)

        tb.addSeparator()
        self.progress = QProgressBar()
        self.progress.setFixedWidth(200)
        tb.addWidget(self.progress)

        # status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — select an engine and click RUN")

        # central
        central = QWidget()
        self.setCentralWidget(central)
        main_lay = QHBoxLayout(central)
        main_lay.setContentsMargins(4, 4, 4, 4)
        main_lay.setSpacing(4)

        # LEFT panel (scrollable setup)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFixedWidth(280)
        left_inner = QWidget()
        left_lay = QVBoxLayout(left_inner)
        left_lay.setContentsMargins(2, 2, 2, 2)
        left_lay.setSpacing(4)

        self.geo_panel = GeometryPanel()
        left_lay.addWidget(self.geo_panel)
        self.room_setup = RoomSetup()
        left_lay.addWidget(self.room_setup)
        self.mat_setup = MaterialSetup()
        left_lay.addWidget(self.mat_setup)
        self.srec_setup = SourceRecSetup()
        left_lay.addWidget(self.srec_setup)
        left_lay.addStretch()
        left_scroll.setWidget(left_inner)

        # RIGHT panel
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(4)

        # metrics bar
        self.metrics_bar = MetricsBar()
        right_lay.addWidget(self.metrics_bar)

        # tabs
        self.tabs = QTabWidget()
        self.ir_display = IRDisplay()
        self.tabs.addTab(self.ir_display, "IR + Spectrogram")
        self.decay_display = DecayDisplay()
        self.tabs.addTab(self.decay_display, "Decay Analysis")
        self.freq_display = FrequencyDisplay()
        self.tabs.addTab(self.freq_display, "Frequency Analysis")
        self.room_view = RoomView3D()
        self.tabs.addTab(self.room_view, "3D Room + Rays")
        self.aural_panel = AuralizationPanel()
        self.tabs.addTab(self.aural_panel, "Auralization")
        self.compare_panel = ComparisonPanel()
        self.tabs.addTab(self.compare_panel, "Compare All")

        right_lay.addWidget(self.tabs, 1)

        # assemble
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right)
        splitter.setSizes([280, 1300])
        main_lay.addWidget(splitter)

        # connect room changes to 3D view
        self.room_setup.changed.connect(self._update_3d)
        self.srec_setup.changed.connect(self._update_3d)
        self.geo_panel.mesh_loaded.connect(self._on_mesh_loaded)
        self.geo_panel.changed.connect(self._update_3d)
        self._update_3d()

    @property
    def _mesh(self):
        return self.geo_panel.mesh if self.geo_panel.mode == "import" else None

    def _get_room(self):
        if self._mesh is not None:
            return self._mesh.to_shoebox()
        Lx, Ly, Lz = self.room_setup.dims()
        return Room(Lx, Ly, Lz, self.mat_setup.surfaces())

    def _on_mesh_loaded(self, mesh):
        if mesh is not None:
            self.room_setup.setVisible(False)
            self.mat_setup.setVisible(False)
            self._update_3d()
            self.status.showMessage(
                f"Loaded: {len(mesh.faces)} faces, {len(mesh.face_groups)} surfaces, "
                f"V={mesh.volume():.1f} m\u00b3")
        else:
            self.room_setup.setVisible(True)
            self.mat_setup.setVisible(True)
            self._update_3d()

    def _update_3d(self):
        mesh = self._mesh
        if mesh is not None:
            self.room_view.show_mesh(mesh, self.srec_setup.src(), self.srec_setup.rec())
        else:
            Lx, Ly, Lz = self.room_setup.dims()
            self.room_view.update_room(Lx, Ly, Lz, self.srec_setup.src(), self.srec_setup.rec())

    def _run_single(self):
        name = self.engine_combo.currentData()
        if not name: return
        info = ENGINE_REGISTRY[name]
        self.btn_run.setEnabled(False)
        self.btn_run_all.setEnabled(False)
        self.progress.setMaximum(0)  # indeterminate
        self.status.showMessage(f"Running {name}...")
        room = self._get_room()
        src, rec = self.srec_setup.src(), self.srec_setup.rec()
        mesh = self._mesh
        self._worker = EngineWorker(name, info["func"], room, src, rec,
                                    44100, self.dur_spin.value(),
                                    mesh=mesh, parent=self)
        self._worker.finished.connect(self._on_single_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_single_done(self, name, result):
        self._current_result = result
        self._all_results[name] = result
        self.progress.setMaximum(1); self.progress.setValue(1)
        self.btn_run.setEnabled(True)
        self.btn_run_all.setEnabled(True)
        bl = f"  [band-limited < {result.max_freq:.0f} Hz]" if result.band_limited else ""
        self.status.showMessage(
            f"{name}  |  T30={result.metrics.get('T30_s','—')}s  "
            f"EDT={result.metrics.get('EDT_s','—')}s  "
            f"C80={result.metrics.get('C80_dB','—')}dB  "
            f"|  {result.compute_time:.2f}s{bl}")
        # update all displays
        self.metrics_bar.update_metrics(result.metrics)
        self.ir_display.plot(result.ir, result.sr)
        self.decay_display.plot(result.ir, result.sr)
        self.freq_display.plot(result.ir, result.sr)
        self.aural_panel.set_ir(result.ir, result.sr)
        # show rays in 3D for geometric engines
        mesh = self._mesh
        if mesh is not None:
            self.room_view.show_mesh(mesh, self.srec_setup.src(), self.srec_setup.rec())
            if result.category == "Geometric":
                self.room_view.show_rays_mesh(mesh, self.srec_setup.src(), n_rays=30, max_bounces=6)
        else:
            Lx, Ly, Lz = self.room_setup.dims()
            self.room_view.update_room(Lx, Ly, Lz, self.srec_setup.src(), self.srec_setup.rec())
            if result.category == "Geometric":
                self.room_view.show_rays((Lx, Ly, Lz), self.srec_setup.src(), n_rays=40, max_bounces=6)
        # also add to comparison
        self.compare_panel.add_result(name, result)

    def _on_error(self, name, err):
        self.progress.setMaximum(1); self.progress.setValue(0)
        self.btn_run.setEnabled(True)
        self.btn_run_all.setEnabled(True)
        self.status.showMessage(f"ERROR in {name}: {err}")

    def _run_all(self):
        self.btn_run.setEnabled(False)
        self.btn_run_all.setEnabled(False)
        self.compare_panel.clear()
        self._all_results.clear()
        self._engines_queue = list(ENGINE_REGISTRY.keys())
        self._total_engines = len(self._engines_queue)
        self.progress.setMaximum(self._total_engines)
        self.progress.setValue(0)
        self.status.showMessage(f"Running 0/{self._total_engines}...")
        self._run_next_in_queue()

    def _run_next_in_queue(self):
        if not self._engines_queue:
            self.btn_run.setEnabled(True)
            self.btn_run_all.setEnabled(True)
            total_time = sum(r.compute_time for r in self._all_results.values())
            self.status.showMessage(f"All {self._total_engines} engines done in {total_time:.1f}s")
            self.tabs.setCurrentWidget(self.compare_panel)
            return
        name = self._engines_queue.pop(0)
        info = ENGINE_REGISTRY[name]
        room = self._get_room()
        src, rec = self.srec_setup.src(), self.srec_setup.rec()
        self.status.showMessage(f"Running {name}... ({self._total_engines - len(self._engines_queue)}/{self._total_engines})")
        mesh = self._mesh
        self._worker = EngineWorker(name, info["func"], room, src, rec,
                                    44100, self.dur_spin.value(),
                                    mesh=mesh, parent=self)
        self._worker.finished.connect(self._on_batch_done)
        self._worker.error.connect(self._on_batch_error)
        self._worker.start()

    def _on_batch_done(self, name, result):
        self._all_results[name] = result
        self._current_result = result
        self.compare_panel.add_result(name, result)
        done = self._total_engines - len(self._engines_queue)
        self.progress.setValue(done)
        # show last result in main displays too
        self.metrics_bar.update_metrics(result.metrics)
        self.ir_display.plot(result.ir, result.sr)
        self._run_next_in_queue()

    def _on_batch_error(self, name, err):
        done = self._total_engines - len(self._engines_queue)
        self.progress.setValue(done)
        self.status.showMessage(f"ERROR: {name} — {err}")
        self._run_next_in_queue()


# ═══════════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    win = UnifiedSim()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
