#!/usr/bin/env python
"""Interactive builder for a 2D helical inimodel (RELION ``relion_helix_inimodel2d``).

Layout
------
* Top    : the *workspace* canvas, two cross-overs wide (``xrect`` x ``yrect``).
           2D classes are dropped here and positioned by hand; each placed class
           automatically gets a Y-flipped mate one cross-over (``xrect/2``) away.
* Middle : left  = the averaged ``_summed`` rectangle (computed on demand);
           right = the central cross-section slice of the ``_3d`` reconstruction.
* Bottom : a gallery of the selected 2D classes.

Interaction
-----------
* Double-click a gallery class            -> add it to the workspace centre (selected).
* Double-click a placed (white) class     -> remove it from the workspace.
* Single-click a placed class             -> select it (green); only selected
                                             classes can be moved / rotated.
* Drag a selected class                    -> reposition it.
* Arrow keys                               -> nudge the selected class by dX/dY.
* Hold R + Left/Right                      -> rotate the selected class by dAngle.
* Flip button                              -> rotate the selected class by 180 deg.
* Transparency slider                      -> class opacity in the workspace.
* Compute button                           -> build the summed rectangle and the
                                             reconstruction cross-section.

The placement/averaging/back-projection all reuse ``inimodel2d_rec3d`` and
``inimodel2d_from_classes`` so the result matches the command-line tools.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mrcfile
import numpy as np
from scipy.ndimage import rotate as nd_rotate

from PyQt5 import QtCore, QtGui, QtWidgets

import inimodel2d_rec3d as rec
from inimodel2d_from_classes import (
    build_canvas,
    compute_sizing,
    read_selected_classes,
)


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def normalise_u8(arr: np.ndarray, brightness: float = 0.0,
                 contrast: float = 1.0) -> np.ndarray:
    """Robustly scale a float image to uint8, then apply contrast/brightness."""
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(a, 2.0), np.percentile(a, 98.0)
    if hi <= lo:
        hi = lo + 1.0
    a = (a - lo) / (hi - lo)
    a = (a - 0.5) * contrast + 0.5 + brightness
    a = np.clip(a, 0.0, 1.0)
    return np.ascontiguousarray((a * 255.0).astype(np.uint8))


def array_to_pixmap(arr: np.ndarray, brightness: float = 0.0,
                    contrast: float = 1.0) -> QtGui.QPixmap:
    u8 = normalise_u8(arr, brightness, contrast)
    h, w = u8.shape
    qimg = QtGui.QImage(u8.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    return QtGui.QPixmap.fromImage(qimg.copy())


def transform_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """In-plane rotation of a class image (same convention used when building)."""
    if abs(angle_deg) < 1e-6:
        return img
    return nd_rotate(img, angle_deg, reshape=False, order=1,
                     mode="constant", cval=0.0)


def _place_centered(img: np.ndarray, yrect: int, xrect: int,
                    mask_radius: float | None = None):
    """Drop a class image into a (yrect, xrect) canvas, centred, with a Y-band mask.

    Returns the padded image and its (binary) support mask.
    """
    ny, nx = img.shape
    out = np.zeros((yrect, xrect), dtype=np.float64)
    mask = np.zeros((yrect, xrect), dtype=np.float64)
    r0 = (yrect - ny) // 2
    c0 = (xrect - nx) // 2
    out[r0:r0 + ny, c0:c0 + nx] = img
    mask[r0:r0 + ny, c0:c0 + nx] = 1.0
    if mask_radius is not None:
        cyc = yrect // 2
        outside = np.abs(np.arange(yrect) - cyc) > mask_radius
        out[outside, :] = 0.0
        mask[outside, :] = 0.0
    return out, mask


def wiggle_optimise(reference, base_image, cur_angle, cur_cx, cur_cy, params,
                    yrect, xrect, down_angpix, mask_diameter):
    """Cross-correlation refinement of one class against the reference rectangle.

    Reproduces the normalised cross-correlation of relion_helix_inimodel2d
    (primary + half-crossover Y-flipped mate placement), searched over flipped /
    rotated versions of the class and all i_offset/j_offset. Minimises:

        Cost = (1 - NCC)
               + k1 * dist(new, current) / crossover
               + k2 * |rot(new) - rot(current)| / max_rot

    The two regularisation terms (not present in RELION) anchor the result to the
    user's manual placement.
    """
    max_rot = float(params["max_rot"])
    delta_rot = float(params["delta_rot"])
    max_y = float(params["max_y"])
    delta_y = float(params["delta_y"])
    k1 = float(params["k1"])
    k2 = float(params["k2"])

    cyc, cxc = yrect // 2, xrect // 2
    crossover_pix = xrect / 2.0
    mask_radius = (mask_diameter / (2.0 * down_angpix)
                   if (mask_diameter and mask_diameter > 0) else yrect / 2.0 - 2)

    R = np.asarray(reference, dtype=np.float64)
    # dihedral-transformed reference: mate at (-row, col + xrect/2)
    Rt = np.roll(np.roll(np.flipud(R), 1, axis=0), -cxc, axis=1)
    FR = np.fft.fft2(R)
    FRt = np.fft.fft2(Rt)
    FM_src = _place_centered(np.ones_like(base_image), yrect, xrect, mask_radius)[1]
    FM = np.fft.fft2(FM_src)
    # denominator sum-of-squares of the reference under the class support
    a2 = (np.fft.ifft2(np.fft.fft2(R * R) * np.conj(FM)).real
          + np.fft.ifft2(np.fft.fft2(Rt * Rt) * np.conj(FM)).real)

    # rotation candidates (small search + the 180-deg flipped set)
    if delta_rot > 0:
        drots = list(np.arange(-max_rot, max_rot + 1e-6, delta_rot))
    else:
        drots = [0.0]
    candidates = [(d, flip) for flip in (False, True) for d in drots]

    # allowed Y positions (centred) and their wrapped row indices
    if delta_y > 0:
        dys = np.arange(-max_y, max_y + 1e-6, delta_y)
    else:
        dys = np.array([0.0])
    cy_news = np.array([int(round(cur_cy + d)) for d in dys])
    di_rows = cy_news % yrect

    cols = np.arange(xrect)
    cx_centered = np.where(cols < cxc, cols, cols - xrect).astype(np.float64)
    dx = cx_centered[None, :] - cur_cx
    dy = cy_news[:, None].astype(np.float64) - cur_cy
    dist = np.sqrt(dx * dx + dy * dy) / crossover_pix

    best = None
    for drot, flip in candidates:
        total_rot = drot + (180.0 if flip else 0.0)
        timg = transform_image(base_image, cur_angle + total_rot)
        tpad = _place_centered(timg, yrect, xrect, mask_radius)[0]
        ft = np.fft.fft2(tpad)
        num = (np.fft.ifft2(FR * np.conj(ft)).real
               + np.fft.ifft2(FRt * np.conj(ft)).real)
        x2 = 2.0 * float(np.sum(tpad * tpad))
        denom = np.sqrt(np.maximum(a2, 0.0) * x2)
        ncc = np.where(denom > 1e-12, num / denom, 0.0)

        sub = ncc[di_rows, :]
        rotpen = (abs(total_rot) / max_rot) if max_rot > 0 else 0.0
        cost = (1.0 - sub) + k1 * dist + k2 * rotpen
        r_min, c_min = np.unravel_index(int(np.argmin(cost)), cost.shape)
        value = float(cost[r_min, c_min])
        if best is None or value < best["cost"]:
            best = {
                "cost": value,
                "cx": float(cx_centered[c_min]),
                "cy": float(cy_news[r_min]),
                "angle": cur_angle + total_rot,
                "ncc": float(sub[r_min, c_min]),
                "drot": drot,
                "flip": flip,
            }

    best["msg"] = (
        f"Wiggle: NCC={best['ncc']:.3f}  "
        f"d(x,y)=({best['cx'] - cur_cx:+.1f},{best['cy'] - cur_cy:+.1f}) px  "
        f"drot={best['drot']:+.1f} deg{' +flip' if best['flip'] else ''}")
    return best


# --------------------------------------------------------------------------- #
# Placed-class model
# --------------------------------------------------------------------------- #
class Placement:
    """State of one 2D class placed in the workspace."""

    def __init__(self, class_idx: int, cx: float = 0.0, cy: float = 0.0,
                 angle: float = 0.0):
        self.class_idx = class_idx
        self.cx = cx          # centre X in centred canvas pixels
        self.cy = cy          # centre Y in centred canvas pixels
        self.angle = angle    # in-plane rotation (deg)


# --------------------------------------------------------------------------- #
# Workspace graphics item for a placed (interactive) class
# --------------------------------------------------------------------------- #
class ClassItem(QtWidgets.QGraphicsPixmapItem):
    def __init__(self, controller, class_idx: int):
        super().__init__()
        self.controller = controller
        self.class_idx = class_idx
        self.setTransformationMode(QtCore.Qt.SmoothTransformation)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(False)

    def center_offset(self) -> None:
        pm = self.pixmap()
        self.setOffset(-pm.width() / 2.0, -pm.height() / 2.0)

    # select on click
    def mousePressEvent(self, event):
        self.controller.select_class(self.class_idx)
        if event.button() == QtCore.Qt.LeftButton:
            self.controller.begin_move()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == QtCore.Qt.LeftButton:
            self.controller.end_move()

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionChange and self.scene():
            value = self.controller.constrain_position(value)
            self.controller.on_item_moved(self.class_idx, value)
        return super().itemChange(change, value)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        if self.controller.selected_idx == self.class_idx:
            pen = QtGui.QPen(QtGui.QColor(0, 220, 0), 0)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawRect(self.boundingRect())


# --------------------------------------------------------------------------- #
# Base canvas view: wheel zoom + pan (spacebar-left-drag or middle-mouse-drag)
# --------------------------------------------------------------------------- #
class CanvasView(QtWidgets.QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self._space_held = False
        self._panning = False
        self._pan_last = None

    def _pan_by(self, pos):
        delta = pos - self._pan_last
        self._pan_last = pos
        hbar, vbar = self.horizontalScrollBar(), self.verticalScrollBar()
        hbar.setValue(hbar.value() - delta.x())
        vbar.setValue(vbar.value() - delta.y())

    def mousePressEvent(self, event):
        if (event.button() == QtCore.Qt.MiddleButton
                or (event.button() == QtCore.Qt.LeftButton and self._space_held)):
            self._panning = True
            self._pan_last = event.pos()
            self.viewport().setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            self._pan_by(event.pos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning:
            self._panning = False
            self._pan_last = None
            self.viewport().setCursor(
                QtCore.Qt.OpenHandCursor if self._space_held
                else QtCore.Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            self._space_held = True
            self.viewport().setCursor(QtCore.Qt.OpenHandCursor)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            self._space_held = False
            if not self._panning:
                self.viewport().setCursor(QtCore.Qt.ArrowCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)


# --------------------------------------------------------------------------- #
# Workspace view (owns key handling for move/rotate)
# --------------------------------------------------------------------------- #
class WorkspaceView(CanvasView):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setRenderHint(QtGui.QPainter.Antialiasing, False)
        self._r_held = False
        self._user_zoom = False

    def fit(self):
        self._user_zoom = False
        if self.scene() is not None:
            self.fitInView(self.scene().sceneRect(), QtCore.Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        self._user_zoom = True
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if not isinstance(self.itemAt(event.pos()), ClassItem):
            self.fit()
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene() is not None and not self._user_zoom:
            self.fitInView(self.scene().sceneRect(), QtCore.Qt.KeepAspectRatio)

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_R:
            self._r_held = True
            event.accept()
            return
        if key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right,
                   QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
            if self._r_held and key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
                sign = -1.0 if key == QtCore.Qt.Key_Left else 1.0
                self.controller.rotate_selected(sign)
            else:
                dx = dy = 0.0
                step = self.controller.delta_move
                if key == QtCore.Qt.Key_Left:
                    dx = -step
                elif key == QtCore.Qt.Key_Right:
                    dx = step
                elif key == QtCore.Qt.Key_Up:
                    dy = -step
                elif key == QtCore.Qt.Key_Down:
                    dy = step
                self.controller.nudge_selected(dx, dy)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key_R:
            self._r_held = False
            event.accept()
            return
        super().keyReleaseEvent(event)


# --------------------------------------------------------------------------- #
# Zoomable image panel (summed rectangle / reconstruction cross-section)
# --------------------------------------------------------------------------- #
class ZoomableImageView(CanvasView):
    """Single-image panel with mouse-wheel zoom + pan; double-click resets to fit."""

    def __init__(self):
        super().__init__()
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = QtWidgets.QGraphicsPixmapItem()
        self._item.setTransformationMode(QtCore.Qt.SmoothTransformation)
        self._scene.addItem(self._item)
        self.setBackgroundBrush(QtGui.QColor("#111"))
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMinimumHeight(80)
        self._has_image = False
        self._user_zoom = False

    def set_image(self, pixmap):
        self._item.setPixmap(pixmap)
        self._scene.setSceneRect(QtCore.QRectF(pixmap.rect()))
        self._has_image = not pixmap.isNull()
        if self._has_image and not self._user_zoom:
            self.fitInView(self._item, QtCore.Qt.KeepAspectRatio)

    def clear(self):
        self._item.setPixmap(QtGui.QPixmap())
        self._has_image = False
        self._user_zoom = False

    def wheelEvent(self, event):
        if not self._has_image:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        self._user_zoom = True
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self._user_zoom = False
        if self._has_image:
            self.fitInView(self._item, QtCore.Qt.KeepAspectRatio)
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._has_image and not self._user_zoom:
            self.fitInView(self._item, QtCore.Qt.KeepAspectRatio)


# --------------------------------------------------------------------------- #
# Gallery of 2D classes
# --------------------------------------------------------------------------- #
class Gallery(QtWidgets.QListWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setViewMode(QtWidgets.QListWidget.IconMode)
        self.setIconSize(QtCore.QSize(96, 96))
        self.setResizeMode(QtWidgets.QListWidget.Adjust)
        self.setMovement(QtWidgets.QListWidget.Static)
        self.setSpacing(6)
        self.setSelectionMode(QtWidgets.QListWidget.NoSelection)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.itemClicked.connect(self._clicked)
        self.itemDoubleClicked.connect(self._double_clicked)

    def _clicked(self, item):
        self.controller.gallery_single_click(item.data(QtCore.Qt.UserRole))

    def _double_clicked(self, item):
        self.controller.gallery_double_click(item.data(QtCore.Qt.UserRole))


# --------------------------------------------------------------------------- #
# Main window / controller
# --------------------------------------------------------------------------- #
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, images, angpix, ori_size, crossover, maxres, mask_diameter,
                 class_refs=None, state_path=None):
        super().__init__()
        self.setWindowTitle("2D IniModel Builder")

        self.images_full = images
        self.class_refs = (
            list(class_refs) if class_refs is not None
            else [f"{i + 1:08d}@unknown.mrcs" for i in range(len(images))])
        self.angpix = angpix
        self.ori_size = ori_size
        self.crossover = crossover
        self.maxres = maxres
        self.mask_diameter = mask_diameter

        # interaction parameters
        self.delta_move = 1.0
        self.delta_rot = 2.0
        self.opacity = 0.6
        self.brightness = 0.0
        self.contrast = 1.0
        self.cross_brightness = 0.0
        self.cross_contrast = 1.0
        self.gallery_icon_size = 96
        self.wiggle = {
            "max_rot": 5.0, "delta_rot": 1.0,
            "max_y": 5.0, "delta_y": 1.0,
            "k1": 0.5, "k2": 0.5,
        }

        # cached arrays for re-rendering displays on brightness/contrast change
        self._last_aref = None
        self._last_central = None

        # state
        self.placements: dict[int, Placement] = {}
        self.items: dict[int, ClassItem] = {}
        self.mates: dict[int, QtWidgets.QGraphicsPixmapItem] = {}
        self.selected_idx: int | None = None
        self.gallery_items: dict[int, QtWidgets.QListWidgetItem] = {}
        self._undo_stack = []
        self._move_snapshot = None

        self._compute_geometry()
        self._build_ui()
        self._populate_gallery()
        self._reset_scene()
        if state_path is not None:
            self.load_state(Path(state_path))

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    def _compute_geometry(self):
        self.down_size, self.down_angpix, self.yrect, self.xrect = compute_sizing(
            self.angpix, self.ori_size, self.crossover, self.maxres)
        # class images downscaled to the model sampling (done once)
        self.images_ds = [rec.fourier_resize(img, self.down_size)
                          for img in self.images_full]

    def _pm(self, arr):
        """Render an array to a pixmap using the current brightness/contrast."""
        return array_to_pixmap(arr, self.brightness, self.contrast)

    def _pm_cross(self, arr):
        """Render the cross-section with its own brightness/contrast."""
        return array_to_pixmap(arr, self.cross_brightness, self.cross_contrast)

    def _update_status(self):
        self.statusBar().showMessage(
            f"xrect={self.xrect}  yrect={self.yrect}  "
            f"down_angpix={self.down_angpix:.3f}  crossover={self.crossover:g} A")

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        root.addLayout(self._build_controls())

        # workspace
        self.scene = QtWidgets.QGraphicsScene()
        self.view = WorkspaceView(self)
        self.view.setScene(self.scene)
        self.view.setBackgroundBrush(QtGui.QColor("#000"))
        self.view.setMinimumHeight(120)
        root.addWidget(self.view, stretch=3)

        # summed + cross-section row
        mid = QtWidgets.QHBoxLayout()
        summed_box, self.summed_view = self._make_image_panel("Summed rectangle")
        cross_box, self.cross_view = self._make_cross_panel(
            "Reconstruction cross-section")
        mid.addWidget(summed_box, stretch=2)
        mid.addWidget(cross_box, stretch=1)
        root.addLayout(mid, stretch=2)

        # gallery size control + gallery
        gal_row = QtWidgets.QHBoxLayout()
        gal_row.addWidget(QtWidgets.QLabel("2D class size"))
        self.gallery_size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gallery_size_slider.setRange(64, 320)
        self.gallery_size_slider.setValue(self.gallery_icon_size)
        self.gallery_size_slider.setFixedWidth(180)
        self.gallery_size_slider.valueChanged.connect(self.on_gallery_size)
        gal_row.addWidget(self.gallery_size_slider)
        gal_row.addStretch(1)
        root.addLayout(gal_row)

        self.gallery = Gallery(self)
        self.gallery.setIconSize(
            QtCore.QSize(self.gallery_icon_size, self.gallery_icon_size))
        self.gallery.setFixedHeight(self._gallery_height())
        root.addWidget(self.gallery)

        undo_sc = QtWidgets.QShortcut(QtGui.QKeySequence.Undo, self)
        undo_sc.activated.connect(self.undo)

        self._update_status()

    def _make_image_panel(self, title):
        box = QtWidgets.QGroupBox(title)
        lay = QtWidgets.QVBoxLayout(box)
        view = ZoomableImageView()
        lay.addWidget(view)
        return box, view

    def _make_cross_panel(self, title):
        box = QtWidgets.QGroupBox(title)
        lay = QtWidgets.QHBoxLayout(box)
        view = ZoomableImageView()
        lay.addWidget(view, stretch=1)

        self.cross_bright_slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.cross_bright_slider.setRange(-100, 100)
        self.cross_bright_slider.setValue(0)
        self.cross_bright_slider.valueChanged.connect(self.on_cross_brightness)

        self.cross_contrast_slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.cross_contrast_slider.setRange(10, 300)
        self.cross_contrast_slider.setValue(100)
        self.cross_contrast_slider.valueChanged.connect(self.on_cross_contrast)

        for caption, slider in (("B", self.cross_bright_slider),
                                ("C", self.cross_contrast_slider)):
            col = QtWidgets.QVBoxLayout()
            lbl = QtWidgets.QLabel(caption)
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            col.addWidget(lbl)
            col.addWidget(slider)
            lay.addLayout(col)
        return box, view

    def _gallery_height(self):
        return self.gallery_icon_size + 56

    def _build_controls(self):
        row = QtWidgets.QHBoxLayout()

        self.crossover_spin = QtWidgets.QDoubleSpinBox()
        self.crossover_spin.setRange(50.0, 5000.0)
        self.crossover_spin.setDecimals(1)
        self.crossover_spin.setSingleStep(10.0)
        self.crossover_spin.setValue(self.crossover)
        self.crossover_spin.setPrefix("crossover ")
        self.crossover_spin.setSuffix(" A")
        self.crossover_spin.valueChanged.connect(self.on_crossover_changed)
        row.addWidget(self.crossover_spin)

        self.move_spin = QtWidgets.QDoubleSpinBox()
        self.move_spin.setRange(0.1, 50.0)
        self.move_spin.setSingleStep(0.5)
        self.move_spin.setValue(self.delta_move)
        self.move_spin.setPrefix("dX/Y ")
        self.move_spin.setSuffix(" px")
        self.move_spin.valueChanged.connect(lambda v: setattr(self, "delta_move", v))
        row.addWidget(self.move_spin)

        self.rot_spin = QtWidgets.QDoubleSpinBox()
        self.rot_spin.setRange(0.1, 90.0)
        self.rot_spin.setSingleStep(0.5)
        self.rot_spin.setValue(self.delta_rot)
        self.rot_spin.setPrefix("dAngle ")
        self.rot_spin.setSuffix(" deg")
        self.rot_spin.valueChanged.connect(lambda v: setattr(self, "delta_rot", v))
        row.addWidget(self.rot_spin)

        row.addWidget(QtWidgets.QLabel("Transparency"))
        self.alpha_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(int(self.opacity * 100))
        self.alpha_slider.setFixedWidth(120)
        self.alpha_slider.valueChanged.connect(self._on_alpha)
        row.addWidget(self.alpha_slider)

        row.addWidget(QtWidgets.QLabel("Brightness"))
        self.bright_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bright_slider.setRange(-100, 100)
        self.bright_slider.setValue(int(self.brightness * 100))
        self.bright_slider.setFixedWidth(110)
        self.bright_slider.valueChanged.connect(self.on_brightness)
        row.addWidget(self.bright_slider)

        row.addWidget(QtWidgets.QLabel("Contrast"))
        self.contrast_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.contrast_slider.setRange(10, 300)
        self.contrast_slider.setValue(int(self.contrast * 100))
        self.contrast_slider.setFixedWidth(110)
        self.contrast_slider.valueChanged.connect(self.on_contrast)
        row.addWidget(self.contrast_slider)

        self.flip_btn = QtWidgets.QPushButton("Flip 180")
        self.flip_btn.clicked.connect(self.flip_selected)
        row.addWidget(self.flip_btn)

        self.wiggle_btn = QtWidgets.QPushButton("Wiggle")
        self.wiggle_btn.clicked.connect(self.wiggle_selected)
        row.addWidget(self.wiggle_btn)

        self.wiggle_settings_btn = QtWidgets.QPushButton("Wiggle settings")
        self.wiggle_settings_btn.clicked.connect(self.open_wiggle_settings)
        row.addWidget(self.wiggle_settings_btn)

        self.wiggle_all_btn = QtWidgets.QPushButton("Wiggle all")
        self.wiggle_all_btn.clicked.connect(self.wiggle_all)
        row.addWidget(self.wiggle_all_btn)

        self.compute_btn = QtWidgets.QPushButton("Compute summed + cross-section")
        self.compute_btn.clicked.connect(self.compute)
        row.addWidget(self.compute_btn)

        self.save_btn = QtWidgets.QPushButton("Save summed + 3D...")
        self.save_btn.clicked.connect(self.save_outputs)
        row.addWidget(self.save_btn)

        row.addStretch(1)
        return row

    # ------------------------------------------------------------------ #
    # Gallery
    # ------------------------------------------------------------------ #
    def _populate_gallery(self):
        for idx, img in enumerate(self.images_ds):
            item = QtWidgets.QListWidgetItem()
            item.setData(QtCore.Qt.UserRole, idx)
            item.setText(str(idx + 1))
            self.gallery.addItem(item)
            self.gallery_items[idx] = item
            self._refresh_gallery_icon(idx)

    def _refresh_gallery_icon(self, idx):
        size = self.gallery_icon_size
        pm = self._pm(self.images_ds[idx]).scaled(
            size, size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        canvas = QtGui.QPixmap(size, size)
        canvas.fill(QtGui.QColor("#000"))
        painter = QtGui.QPainter(canvas)
        x = (size - pm.width()) // 2
        y = (size - pm.height()) // 2
        painter.drawPixmap(x, y, pm)
        color = None
        if idx == self.selected_idx:
            color = QtGui.QColor(0, 220, 0)
        elif idx in self.placements:
            color = QtGui.QColor(255, 255, 0)
        if color is not None:
            bw = max(3, size // 24)
            pen = QtGui.QPen(color, bw)
            painter.setPen(pen)
            off = bw // 2 + 1
            painter.drawRect(off, off, size - 2 * off, size - 2 * off)
        painter.end()
        self.gallery_items[idx].setIcon(QtGui.QIcon(canvas))

    def gallery_single_click(self, idx):
        if idx in self.placements:
            self.select_class(idx)

    def gallery_double_click(self, idx):
        self.push_undo()
        if idx in self.placements:
            self.remove_class(idx)
        else:
            self.add_class(idx)
        self.view.setFocus()

    # ------------------------------------------------------------------ #
    # Scene management
    # ------------------------------------------------------------------ #
    def _reset_scene(self):
        self.scene.clear()
        self.items.clear()
        self.mates.clear()
        hw, hh = self.xrect / 2.0, self.yrect / 2.0
        self.scene.setSceneRect(-hw, -hh, self.xrect, self.yrect)

        # background rectangle + guide lines
        border = QtGui.QPen(QtGui.QColor(80, 120, 200), 0)
        border.setCosmetic(True)
        self.scene.addRect(-hw, -hh, self.xrect, self.yrect, border)
        guide = QtGui.QPen(QtGui.QColor(70, 70, 70), 0, QtCore.Qt.DashLine)
        guide.setCosmetic(True)
        self.scene.addLine(0, -hh, 0, hh, guide)        # crossover split
        self.scene.addLine(-hw, 0, hw, 0, guide)        # horizontal centre
        self.view.fit()

    def _wrap_x(self, x):
        hw = self.xrect / 2.0
        return (x + hw) % self.xrect - hw

    def constrain_position(self, pos: QtCore.QPointF) -> QtCore.QPointF:
        return QtCore.QPointF(self._wrap_x(pos.x()), pos.y())

    def _rendered_image(self, idx):
        return transform_image(self.images_ds[idx], self.placements[idx].angle)

    def _rebuild_item_pixmaps(self, idx):
        img = self._rendered_image(idx)
        self.items[idx].setPixmap(self._pm(img))
        self.items[idx].center_offset()
        mate_img = np.flipud(img)
        self.mates[idx].setPixmap(self._pm(mate_img))
        pm = self.mates[idx].pixmap()
        self.mates[idx].setOffset(-pm.width() / 2.0, -pm.height() / 2.0)
        self._update_positions(idx)

    def _update_positions(self, idx):
        p = self.placements[idx]
        self.items[idx].setPos(p.cx, p.cy)
        mate_x = self._wrap_x(p.cx + self.xrect / 2.0)
        self.mates[idx].setPos(mate_x, -p.cy)

    def add_class(self, idx):
        self.placements[idx] = Placement(idx, 0.0, 0.0, 0.0)
        self._create_graphics(idx)
        self.select_class(idx)
        self._refresh_gallery_icon(idx)

    def _create_graphics(self, idx):
        item = ClassItem(self, idx)
        item.setOpacity(self.opacity)
        item.setZValue(10)
        item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable,
                     idx == self.selected_idx)
        self.scene.addItem(item)
        self.items[idx] = item

        mate = QtWidgets.QGraphicsPixmapItem()
        mate.setTransformationMode(QtCore.Qt.SmoothTransformation)
        mate.setOpacity(self.opacity * 0.7)
        mate.setZValue(5)
        self.scene.addItem(mate)
        self.mates[idx] = mate

        self._rebuild_item_pixmaps(idx)

    def remove_class(self, idx):
        if idx in self.items:
            self.scene.removeItem(self.items.pop(idx))
        if idx in self.mates:
            self.scene.removeItem(self.mates.pop(idx))
        self.placements.pop(idx, None)
        if self.selected_idx == idx:
            self.selected_idx = None
        self._refresh_gallery_icon(idx)

    def select_class(self, idx):
        prev = self.selected_idx
        self.selected_idx = idx
        for i in (prev, idx):
            if i is None:
                continue
            if i in self.items:
                movable = i == idx
                self.items[i].setFlag(
                    QtWidgets.QGraphicsItem.ItemIsMovable, movable)
                # raise the selected class (and its mate) above the others
                self.items[i].setZValue(20 if movable else 10)
                self.mates[i].setZValue(15 if movable else 5)
                self.items[i].update()
            if i in self.gallery_items:
                self._refresh_gallery_icon(i)

    # ------------------------------------------------------------------ #
    # Undo history
    # ------------------------------------------------------------------ #
    def _snapshot(self):
        placed = {i: (p.cx, p.cy, p.angle) for i, p in self.placements.items()}
        return placed, self.selected_idx

    def push_undo(self):
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > 200:
            self._undo_stack.pop(0)

    def begin_move(self):
        self._move_snapshot = self._snapshot()

    def end_move(self):
        if self._move_snapshot is None:
            return
        before = self._move_snapshot[0]
        now = {i: (p.cx, p.cy, p.angle) for i, p in self.placements.items()}
        if before != now:
            self._undo_stack.append(self._move_snapshot)
        self._move_snapshot = None

    def undo(self):
        if not self._undo_stack:
            self.statusBar().showMessage("Nothing to undo.")
            return
        self._restore_state(self._undo_stack.pop())
        self.statusBar().showMessage("Undo.")

    def _restore_state(self, state):
        placed, sel = state
        affected = set(self.placements) | set(placed)
        for i in list(self.items):
            self.scene.removeItem(self.items[i])
            self.scene.removeItem(self.mates[i])
        self.items.clear()
        self.mates.clear()
        self.placements = {i: Placement(i, cx, cy, ang)
                           for i, (cx, cy, ang) in placed.items()}
        self.selected_idx = sel if sel in self.placements else None
        for i in sorted(self.placements):
            self._create_graphics(i)
        for i in affected:
            if i in self.gallery_items:
                self._refresh_gallery_icon(i)

    # ------------------------------------------------------------------ #
    # Movement / rotation callbacks
    # ------------------------------------------------------------------ #
    def on_item_moved(self, idx, pos: QtCore.QPointF):
        p = self.placements[idx]
        p.cx, p.cy = pos.x(), pos.y()
        mate_x = self._wrap_x(p.cx + self.xrect / 2.0)
        self.mates[idx].setPos(mate_x, -p.cy)

    def nudge_selected(self, dx, dy):
        if self.selected_idx is None:
            return
        self.push_undo()
        p = self.placements[self.selected_idx]
        p.cx = self._wrap_x(p.cx + dx)
        p.cy = p.cy + dy
        self._update_positions(self.selected_idx)

    def rotate_selected(self, sign):
        if self.selected_idx is None:
            return
        self.push_undo()
        self.placements[self.selected_idx].angle += sign * self.delta_rot
        self._rebuild_item_pixmaps(self.selected_idx)

    def flip_selected(self):
        if self.selected_idx is None:
            return
        self.push_undo()
        self.placements[self.selected_idx].angle += 180.0
        self._rebuild_item_pixmaps(self.selected_idx)

    def _wiggle_one(self, idx):
        others = [i for i in sorted(self.placements) if i != idx]
        if not others:
            return None
        ref_imgs = [transform_image(self.images_ds[i], self.placements[i].angle)
                    for i in others]
        ref_xs = [self.placements[i].cx for i in others]
        ref_ys = [self.placements[i].cy for i in others]
        reference = build_canvas(ref_imgs, ref_xs, ref_ys, self.yrect, self.xrect)
        p = self.placements[idx]
        result = wiggle_optimise(
            reference, self.images_ds[idx], p.angle, p.cx, p.cy,
            self.wiggle, self.yrect, self.xrect, self.down_angpix,
            self.mask_diameter)
        p.cx, p.cy, p.angle = result["cx"], result["cy"], result["angle"]
        self._rebuild_item_pixmaps(idx)
        self._refresh_gallery_icon(idx)
        return result

    def wiggle_selected(self):
        if self.selected_idx is None:
            self.statusBar().showMessage("No class selected to wiggle.")
            return
        idx = self.selected_idx
        if not [i for i in self.placements if i != idx]:
            self.statusBar().showMessage(
                "Wiggle needs at least one other placed class as reference.")
            return
        self.push_undo()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            result = self._wiggle_one(idx)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(result["msg"])

    def wiggle_all(self):
        placed = sorted(self.placements)
        if len(placed) < 2:
            self.statusBar().showMessage(
                "Wiggle all needs at least two placed classes.")
            return
        self.push_undo()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        moved = 0
        try:
            for idx in placed:
                if self._wiggle_one(idx) is not None:
                    moved += 1
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(
            f"Wiggle all: refined {moved}/{len(placed)} classes.")

    def open_wiggle_settings(self):
        w = self.wiggle
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Wiggle settings")
        form = QtWidgets.QFormLayout(dlg)

        def spin(value, lo, hi, step, suffix):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(1)
            s.setSingleStep(step)
            s.setValue(value)
            s.setSuffix(suffix)
            return s

        max_rot = spin(w["max_rot"], 0.0, 180.0, 1.0, " deg")
        delta_rot = spin(w["delta_rot"], 0.1, 90.0, 0.5, " deg")
        max_y = spin(w["max_y"], 0.0, 200.0, 1.0, " px")
        delta_y = spin(w["delta_y"], 0.1, 50.0, 0.5, " px")

        def rigidity(value):
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(int(round(value * 100)))
            slider.setFixedWidth(160)
            lab = QtWidgets.QLabel(f"{value:.2f}")
            slider.valueChanged.connect(lambda v: lab.setText(f"{v / 100:.2f}"))
            holder = QtWidgets.QWidget()
            box = QtWidgets.QHBoxLayout(holder)
            box.setContentsMargins(0, 0, 0, 0)
            box.addWidget(slider)
            box.addWidget(lab)
            return slider, holder

        k1_slider, k1_w = rigidity(w["k1"])
        k2_slider, k2_w = rigidity(w["k2"])

        form.addRow("Max rotation", max_rot)
        form.addRow("Delta rotation", delta_rot)
        form.addRow("Max Y displacement", max_y)
        form.addRow("Delta Y displacement", delta_y)
        form.addRow("Translation rigidity (k1)", k1_w)
        form.addRow("Rotation rigidity (k2)", k2_w)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            w["max_rot"] = max_rot.value()
            w["delta_rot"] = delta_rot.value()
            w["max_y"] = max_y.value()
            w["delta_y"] = delta_y.value()
            w["k1"] = k1_slider.value() / 100.0
            w["k2"] = k2_slider.value() / 100.0
            self.statusBar().showMessage("Wiggle settings updated.")

    def _on_alpha(self, value):
        self.opacity = value / 100.0
        for idx, item in self.items.items():
            item.setOpacity(self.opacity)
            self.mates[idx].setOpacity(self.opacity * 0.7)

    def on_crossover_changed(self, value):
        value = float(value)
        if abs(value - self.crossover) < 1e-9:
            return
        old_xrect, old_yrect = self.xrect, self.yrect
        self.crossover = value
        self._compute_geometry()
        # keep placements at the same relative position in the resized canvas
        sx = self.xrect / old_xrect
        sy = self.yrect / old_yrect
        for p in self.placements.values():
            p.cx = self._wrap_x(p.cx * sx)
            p.cy = p.cy * sy
        self._reset_scene()
        for idx in sorted(self.placements):
            self._create_graphics(idx)
        self._last_aref = None
        self._last_central = None
        self.summed_view.clear()
        self.cross_view.clear()
        self._update_status()

    def on_brightness(self, value):
        self.brightness = value / 100.0
        self._rerender_global()

    def on_contrast(self, value):
        self.contrast = value / 100.0
        self._rerender_global()

    def on_cross_brightness(self, value):
        self.cross_brightness = value / 100.0
        self._rerender_cross()

    def on_cross_contrast(self, value):
        self.cross_contrast = value / 100.0
        self._rerender_cross()

    def on_gallery_size(self, value):
        self.gallery_icon_size = int(value)
        self.gallery.setIconSize(
            QtCore.QSize(self.gallery_icon_size, self.gallery_icon_size))
        self.gallery.setFixedHeight(self._gallery_height())
        for idx in self.gallery_items:
            self._refresh_gallery_icon(idx)

    def _rerender_global(self):
        for idx in self.gallery_items:
            self._refresh_gallery_icon(idx)
        for idx in list(self.items):
            self._rebuild_item_pixmaps(idx)
        if self._last_aref is not None:
            self.summed_view.set_image(self._pm(self._last_aref))

    def _rerender_cross(self):
        if self._last_central is not None:
            self.cross_view.set_image(self._pm_cross(self._last_central))

    # ------------------------------------------------------------------ #
    # Compute
    # ------------------------------------------------------------------ #
    def _build_summed(self):
        if not self.placements:
            return None
        order = sorted(self.placements)
        imgs = [transform_image(self.images_ds[i], self.placements[i].angle)
                for i in order]
        xs = [self.placements[i].cx for i in order]
        ys = [self.placements[i].cy for i in order]
        return build_canvas(imgs, xs, ys, self.yrect, self.xrect)

    def compute(self):
        aref = self._build_summed()
        if aref is None:
            self.statusBar().showMessage("No classes placed.")
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.summed_view.set_image(self._pm(aref))

            cross = rec.reconstruct2d(aref)
            if self.mask_diameter and self.mask_diameter > 0:
                cross = rec.apply_circular_mask(
                    cross, self.mask_diameter / (2.0 * self.down_angpix))
            cross = rec.fourier_resize(cross, self.ori_size)
            deg_per_pixel = 180.0 * self.angpix / self.crossover
            central = nd_rotate(cross, deg_per_pixel * (self.ori_size // 2),
                                reshape=False, order=1, mode="constant", cval=0.0)
            self.cross_view.set_image(self._pm_cross(central))
            self._last_aref = aref
            self._last_central = central
            self.statusBar().showMessage(
                f"Computed from {len(self.placements)} classes.")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def save_outputs(self):
        aref = self._build_summed()
        if aref is None:
            self.statusBar().showMessage("No classes placed.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save 3D reconstruction", "inimodel.mrc", "MRC (*.mrc)")
        if not path:
            return
        out3d = Path(path)
        stem = str(out3d.with_suffix(""))
        summed = Path(stem + "_summed.mrc")
        reproj_path = Path(stem + "_reprojections.mrc")
        state_path = Path(stem + ".star")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            rec.write_mrc(summed, aref, self.down_angpix)

            cross = rec.reconstruct2d(aref)
            if self.mask_diameter and self.mask_diameter > 0:
                cross = rec.apply_circular_mask(
                    cross, self.mask_diameter / (2.0 * self.down_angpix))
            # reprojections: forward-project the 2D reconstruction (as RELION does)
            reproj = rec.project2d(cross, self.xrect)
            rec.write_mrc(reproj_path, reproj, self.down_angpix)

            cross_big = rec.fourier_resize(cross, self.ori_size)
            vol = rec.reconstruct3d(cross_big, self.angpix, self.crossover)
            rec.write_mrc(out3d, vol, self.angpix)

            self._write_state_star(state_path)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(
            f"Wrote {out3d.name}, {summed.name}, {reproj_path.name}, "
            f"{state_path.name}")

    def _write_state_star(self, path):
        lines = [
            "# 2D IniModel builder saved state",
            "",
            "data_inimodel2d",
            "",
            f"_inimodelCrossoverDistance   {self.crossover:.6f}",
            f"_inimodelPixelSize           {self.angpix:.6f}",
            f"_inimodelMaxResolution       {self.maxres:.6f}",
            f"_inimodelMaskDiameter        {self.mask_diameter:.6f}",
            "",
            "data_classes",
            "",
            "loop_",
            "_rlnReferenceImage #1",
            "_rlnAnglePsi #2",
            "_rlnOriginYAngst #3",
            "_inimodelXFraction #4",
        ]
        crossover_pix = self.xrect / 2.0
        for idx in sorted(self.placements):
            p = self.placements[idx]
            yangst = p.cy * self.down_angpix
            xfrac = p.cx / crossover_pix
            lines.append(
                f"{self.class_refs[idx]}   {p.angle:.6f}   "
                f"{yangst:.6f}   {xfrac:.6f}")
        Path(path).write_text("\n".join(lines) + "\n")

    def load_state(self, path):
        params, rows = parse_state_star(Path(path))
        new_crossover = params.get("crossover")
        if new_crossover is not None and abs(new_crossover - self.crossover) > 1e-9:
            self.crossover_spin.blockSignals(True)
            self.crossover_spin.setValue(new_crossover)
            self.crossover_spin.blockSignals(False)
            self.crossover = new_crossover
            self._compute_geometry()
            self._reset_scene()
            self._update_status()

        stack_to_idx = {}
        for i, ref in enumerate(self.class_refs):
            try:
                stack_to_idx[int(ref.split("@", 1)[0])] = i
            except ValueError:
                pass

        crossover_pix = self.xrect / 2.0
        placed = 0
        for row in rows:
            try:
                idx = stack_to_idx.get(int(row["ref"].split("@", 1)[0]))
            except ValueError:
                idx = None
            if idx is None or idx in self.placements:
                continue
            cx = row["xfrac"] * crossover_pix
            cy = row["yangst"] / self.down_angpix
            self.placements[idx] = Placement(idx, cx, cy, row["psi"])
            self._create_graphics(idx)
            self._refresh_gallery_icon(idx)
            placed += 1
        self.statusBar().showMessage(f"Loaded state: {placed} classes placed.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_state_star(path):
    """Parse a saved-state star: returns (params dict, list of class rows)."""
    key_map = {
        "_inimodelCrossoverDistance": "crossover",
        "_inimodelPixelSize": "angpix",
        "_inimodelMaxResolution": "maxres",
        "_inimodelMaskDiameter": "mask",
    }
    params = {}
    columns = []
    rows = []
    in_loop = False
    for raw in Path(path).read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s == "loop_":
            in_loop = True
            columns = []
            continue
        if s.startswith("data_"):
            in_loop = False
            continue
        head = s.split()[0]
        if head in key_map:
            params[key_map[head]] = float(s.split()[1])
            continue
        if in_loop and s.startswith("_"):
            columns.append(head)
            continue
        if in_loop:
            toks = s.split()
            if len(toks) >= len(columns):
                rows.append(dict(zip(columns, toks)))

    parsed = []
    for r in rows:
        if "_rlnReferenceImage" not in r:
            continue
        parsed.append({
            "ref": r["_rlnReferenceImage"],
            "psi": float(r.get("_rlnAnglePsi", 0.0)),
            "yangst": float(r.get("_rlnOriginYAngst", 0.0)),
            "xfrac": float(r.get("_inimodelXFraction", 0.0)),
        })
    return params, parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class-star", type=Path, required=True,
                        help="Star file listing the selected 2D classes")
    parser.add_argument("--class-mrcs", type=Path, default=None,
                        help="2D class stack (default: resolved from the star)")
    parser.add_argument("--project-dir", type=Path, default=None,
                        help="Project root for resolving the stack path in the star")
    parser.add_argument("--crossover", type=float, required=True,
                        help="Crossover distance (A)")
    parser.add_argument("--maxres", type=float, default=6.0,
                        help="Resolution limit (A) used for downscaling")
    parser.add_argument("--mask-diameter", type=float, default=570.0,
                        help="Circular mask diameter (A); <=0 to disable")
    parser.add_argument("--state", type=Path, default=None,
                        help="Optional saved-state star file to restore the workspace")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    images, angpix, ori_size, refs = read_selected_classes(
        args.class_star, args.class_mrcs, args.project_dir)

    app = QtWidgets.QApplication(sys.argv[:1])
    win = MainWindow(images, angpix, ori_size, args.crossover,
                     args.maxres, args.mask_diameter,
                     class_refs=refs, state_path=args.state)
    win.resize(1200, 820)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
