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
def normalise_u8(arr: np.ndarray) -> np.ndarray:
    """Robustly scale a float image to contiguous uint8 for display."""
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(a, 2.0), np.percentile(a, 98.0)
    if hi <= lo:
        hi = lo + 1.0
    a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return np.ascontiguousarray((a * 255.0).astype(np.uint8))


def array_to_pixmap(arr: np.ndarray) -> QtGui.QPixmap:
    u8 = normalise_u8(arr)
    h, w = u8.shape
    qimg = QtGui.QImage(u8.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    return QtGui.QPixmap.fromImage(qimg.copy())


def transform_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """In-plane rotation of a class image (same convention used when building)."""
    if abs(angle_deg) < 1e-6:
        return img
    return nd_rotate(img, angle_deg, reshape=False, order=1,
                     mode="constant", cval=0.0)


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
        super().mousePressEvent(event)

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
# Workspace view (owns key handling for move/rotate)
# --------------------------------------------------------------------------- #
class WorkspaceView(QtWidgets.QGraphicsView):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setRenderHint(QtGui.QPainter.Antialiasing, False)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self._r_held = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene() is not None:
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
    def __init__(self, images, angpix, ori_size, crossover, maxres, mask_diameter):
        super().__init__()
        self.setWindowTitle("2D IniModel Builder")

        self.images_full = images
        self.angpix = angpix
        self.ori_size = ori_size
        self.crossover = crossover
        self.maxres = maxres
        self.mask_diameter = mask_diameter

        # interaction parameters
        self.delta_move = 1.0
        self.delta_rot = 2.0
        self.opacity = 0.6

        # state
        self.placements: dict[int, Placement] = {}
        self.items: dict[int, ClassItem] = {}
        self.mates: dict[int, QtWidgets.QGraphicsPixmapItem] = {}
        self.selected_idx: int | None = None
        self.gallery_items: dict[int, QtWidgets.QListWidgetItem] = {}

        self._compute_geometry()
        self._build_ui()
        self._populate_gallery()
        self._reset_scene()

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    def _compute_geometry(self):
        self.down_size, self.down_angpix, self.yrect, self.xrect = compute_sizing(
            self.angpix, self.ori_size, self.crossover, self.maxres)
        # class images downscaled to the model sampling (done once)
        self.images_ds = [rec.fourier_resize(img, self.down_size)
                          for img in self.images_full]

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
        self.view.setMinimumHeight(200)
        root.addWidget(self.view, stretch=3)

        # summed + cross-section row
        mid = QtWidgets.QHBoxLayout()
        self.summed_label = self._make_image_panel("Summed rectangle")
        self.cross_label = self._make_image_panel("Reconstruction cross-section")
        mid.addWidget(self.summed_label[0], stretch=2)
        mid.addWidget(self.cross_label[0], stretch=1)
        root.addLayout(mid, stretch=2)

        # gallery
        self.gallery = Gallery(self)
        self.gallery.setFixedHeight(150)
        root.addWidget(self.gallery)

        self.statusBar().showMessage(
            f"xrect={self.xrect}  yrect={self.yrect}  "
            f"down_angpix={self.down_angpix:.3f}  crossover={self.crossover:g} A")

    def _make_image_panel(self, title):
        box = QtWidgets.QGroupBox(title)
        lay = QtWidgets.QVBoxLayout(box)
        label = QtWidgets.QLabel()
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setMinimumHeight(140)
        label.setStyleSheet("background:#111;")
        lay.addWidget(label)
        return box, label

    def _build_controls(self):
        row = QtWidgets.QHBoxLayout()

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

        self.flip_btn = QtWidgets.QPushButton("Flip 180")
        self.flip_btn.clicked.connect(self.flip_selected)
        row.addWidget(self.flip_btn)

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
        pm = array_to_pixmap(self.images_ds[idx]).scaled(
            96, 96, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        canvas = QtGui.QPixmap(96, 96)
        canvas.fill(QtGui.QColor("#000"))
        painter = QtGui.QPainter(canvas)
        x = (96 - pm.width()) // 2
        y = (96 - pm.height()) // 2
        painter.drawPixmap(x, y, pm)
        color = None
        if idx == self.selected_idx:
            color = QtGui.QColor(0, 220, 0)
        elif idx in self.placements:
            color = QtGui.QColor(255, 255, 255)
        if color is not None:
            pen = QtGui.QPen(color, 4)
            painter.setPen(pen)
            painter.drawRect(2, 2, 92, 92)
        painter.end()
        self.gallery_items[idx].setIcon(QtGui.QIcon(canvas))

    def gallery_single_click(self, idx):
        if idx in self.placements:
            self.select_class(idx)

    def gallery_double_click(self, idx):
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
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def _wrap_x(self, x):
        hw = self.xrect / 2.0
        return (x + hw) % self.xrect - hw

    def constrain_position(self, pos: QtCore.QPointF) -> QtCore.QPointF:
        return QtCore.QPointF(self._wrap_x(pos.x()), pos.y())

    def _rendered_image(self, idx):
        return transform_image(self.images_ds[idx], self.placements[idx].angle)

    def _rebuild_item_pixmaps(self, idx):
        img = self._rendered_image(idx)
        self.items[idx].setPixmap(array_to_pixmap(img))
        self.items[idx].center_offset()
        mate_img = np.flipud(img)
        self.mates[idx].setPixmap(array_to_pixmap(mate_img))
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

        item = ClassItem(self, idx)
        item.setOpacity(self.opacity)
        item.setZValue(10)
        self.scene.addItem(item)
        self.items[idx] = item

        mate = QtWidgets.QGraphicsPixmapItem()
        mate.setTransformationMode(QtCore.Qt.SmoothTransformation)
        mate.setOpacity(self.opacity * 0.7)
        mate.setZValue(5)
        self.scene.addItem(mate)
        self.mates[idx] = mate

        self._rebuild_item_pixmaps(idx)
        self.select_class(idx)
        self._refresh_gallery_icon(idx)

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
                self.items[i].update()
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
        p = self.placements[self.selected_idx]
        p.cx = self._wrap_x(p.cx + dx)
        p.cy = p.cy + dy
        self._update_positions(self.selected_idx)

    def rotate_selected(self, sign):
        if self.selected_idx is None:
            return
        self.placements[self.selected_idx].angle += sign * self.delta_rot
        self._rebuild_item_pixmaps(self.selected_idx)

    def flip_selected(self):
        if self.selected_idx is None:
            return
        self.placements[self.selected_idx].angle += 180.0
        self._rebuild_item_pixmaps(self.selected_idx)

    def _on_alpha(self, value):
        self.opacity = value / 100.0
        for idx, item in self.items.items():
            item.setOpacity(self.opacity)
            self.mates[idx].setOpacity(self.opacity * 0.7)

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
            self._show_array(self.summed_label[1], aref)

            cross = rec.reconstruct2d(aref)
            if self.mask_diameter and self.mask_diameter > 0:
                cross = rec.apply_circular_mask(
                    cross, self.mask_diameter / (2.0 * self.down_angpix))
            cross = rec.fourier_resize(cross, self.ori_size)
            deg_per_pixel = 180.0 * self.angpix / self.crossover
            central = nd_rotate(cross, deg_per_pixel * (self.ori_size // 2),
                                reshape=False, order=1, mode="constant", cval=0.0)
            self._show_array(self.cross_label[1], central)
            self._last_aref = aref
            self.statusBar().showMessage(
                f"Computed from {len(self.placements)} classes.")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _show_array(self, label, arr):
        pm = array_to_pixmap(arr).scaled(
            label.width(), label.height(),
            QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        label.setPixmap(pm)

    def save_outputs(self):
        aref = self._build_summed()
        if aref is None:
            self.statusBar().showMessage("No classes placed.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save 3D reconstruction", "test_3d.mrc", "MRC (*.mrc)")
        if not path:
            return
        out3d = Path(path)
        summed = out3d.with_name(out3d.stem + "_summed.mrc")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            rec.write_mrc(summed, aref, self.down_angpix)
            cross = rec.reconstruct2d(aref)
            if self.mask_diameter and self.mask_diameter > 0:
                cross = rec.apply_circular_mask(
                    cross, self.mask_diameter / (2.0 * self.down_angpix))
            cross = rec.fourier_resize(cross, self.ori_size)
            vol = rec.reconstruct3d(cross, self.angpix, self.crossover)
            rec.write_mrc(out3d, vol, self.angpix)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f"Wrote {summed} and {out3d}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
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
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    images, angpix, ori_size = read_selected_classes(
        args.class_star, args.class_mrcs, args.project_dir)

    app = QtWidgets.QApplication(sys.argv[:1])
    win = MainWindow(images, angpix, ori_size, args.crossover,
                     args.maxres, args.mask_diameter)
    win.resize(1200, 820)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
