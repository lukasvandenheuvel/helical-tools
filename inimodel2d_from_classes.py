#!/usr/bin/env python
"""Re-implementation of the canvas-building stage of
``relion_helix_inimodel2d``, driven by externally supplied class-centre
coordinates.

Given a set of selected 2D class averages and, for each one, the (X, Y)
coordinate of its centre along its crossover, this reproduces what the RELION
program holds after every iteration:

* downscale each class to the model's ``down_angpix`` sampling (``resizeMap``);
* paint each class onto the ``yrect x xrect`` rectangle at its (X, Y) centre,
  wrapping periodically in X, and additionally paint a Y-flipped copy half a
  crossover (``xrect/2``) away -- exactly RELION's ``Asum``/``Asumw`` loop;
* average the canvas (``Aref = Asum / Asumw``)  -> ``test_summed.mrc``.

The averaged rectangle is then fed through the same back-projection used by
``inimodel2d_rec3d`` (``reconstruct2D`` + resize + mask + ``reconstruct3D``)
-> ``test_3d.mrc``.

For now the (X, Y) centres are drawn randomly (X ~ U[0, 2*crossover),
Y ~ N(0, 2 px)); ``get_coordinates`` is isolated so it can later be replaced by
a reader for a star file of coordinates.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import mrcfile
import numpy as np

import inimodel2d_rec3d as rec


def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def compute_sizing(angpix: float, ori_size: int, crossover: float,
                   maxres: float, search_size: int = 5):
    """Replicate RELION's down_size/down_angpix/yrect/xrect selection."""
    if maxres < 0.0 or maxres < 2.0 * angpix:
        maxres = 2.0 * angpix
    down_size = ori_size * angpix * 2.0 / maxres

    best_fit, down_angpix, best_size = 1.0, 0.0, 0
    for delta in range(-search_size, search_size + 1, 2):
        mysize = int(down_size + delta)
        mysize -= mysize % 2
        if mysize <= ori_size:
            myangpix = angpix * ori_size / mysize
            mydiv = (2.0 * crossover) / myangpix
            myfit = math.fmod(mydiv, 2.0)
            if myfit > 1.0:
                myfit -= 2.0
            myfit = abs(myfit)
            if myfit < best_fit:
                best_fit, down_angpix, best_size = myfit, myangpix, mysize

    down_size = best_size
    yrect = round_half_up(ori_size * angpix / down_angpix)
    yrect -= yrect % 2
    xrect = round_half_up((2.0 * crossover) / down_angpix)
    return down_size, down_angpix, yrect, xrect


def read_selected_classes(star_path: Path, mrcs_path: Path | None,
                          project_dir: Path | None):
    """Return (images, angpix, ori_size, ref_strings) for the selected classes."""
    refs = []
    with open(star_path) as handle:
        for line in handle:
            tok = line.strip().split()
            if tok and "@" in tok[0] and ".mrcs" in tok[0]:
                idx, rel = tok[0].split("@", 1)
                refs.append((int(idx), rel, tok[0]))
    if not refs:
        raise ValueError(f"No _rlnReferenceImage entries found in {star_path}")

    if mrcs_path is None:
        root = project_dir if project_dir is not None else star_path.parents[2]
        mrcs_path = root / refs[0][1]

    with mrcfile.open(mrcs_path, permissive=True) as mrc:
        data = np.array(mrc.data, dtype=np.float32)
        angpix = float(mrc.voxel_size.x)
        ori_size = int(mrc.header.nx)

    images = [data[idx - 1] for idx, _, _ in refs]
    ref_strings = [raw for _, _, raw in refs]
    return images, angpix, ori_size, ref_strings


def get_coordinates(n: int, xrect: int, seed: int | None = None):
    """Draw class-centre coordinates (placeholder for a future star reader).

    X ~ Uniform[0, xrect)  (= 0 .. 2*crossover in pixels)
    Y ~ Normal(mean=0, std=2 px)
    """
    rng = np.random.default_rng(seed)
    coords_x = rng.uniform(0.0, xrect, size=n)
    coords_y = rng.normal(0.0, 2.0, size=n)
    return coords_x, coords_y


def build_canvas(images, coords_x, coords_y, yrect: int, xrect: int):
    """Paint classes onto the rectangle like RELION's Asum/Asumw accumulation."""
    asum = np.zeros((yrect, xrect), dtype=np.float64)
    asumw = np.zeros((yrect, xrect), dtype=np.float64)

    ny = images[0].shape[0]
    cy = ny // 2                 # image centre (== yrect//2 by construction)
    cxc = xrect // 2             # canvas X centre
    ii, jj = np.meshgrid(np.arange(ny), np.arange(ny), indexing="ij")
    i_cen = ii - cy
    j_cen = jj - cy

    for img, xo, yo in zip(images, coords_x, coords_y):
        xo = int(round_half_up(xo))
        yo = int(round_half_up(yo))

        i_shift = i_cen + yo                       # centred canvas Y
        j_shift = j_cen + xo                       # centred canvas X
        i_arr = (i_shift + cy) % yrect
        j_arr = (j_shift + cxc) % xrect
        np.add.at(asum, (i_arr, j_arr), img)
        np.add.at(asumw, (i_arr, j_arr), 1.0)

        # Y-flipped copy, half a crossover (xrect/2) further along X.
        i_flip = -i_shift
        valid = (i_flip >= -cy) & (i_flip <= cy - 1)
        i_flip_arr = (i_flip + cy) % yrect
        j_flip_arr = (j_shift + xrect // 2 + cxc) % xrect
        np.add.at(asum, (i_flip_arr[valid], j_flip_arr[valid]), img[valid])
        np.add.at(asumw, (i_flip_arr[valid], j_flip_arr[valid]), 1.0)

    aref = np.zeros_like(asum)
    nonzero = asumw > 0.0
    aref[nonzero] = asum[nonzero] / asumw[nonzero]
    return aref.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    parser.add_argument("--rot-sign", type=float, default=1.0,
                        help="Twist sign for the 3D sweep (use -1 to flip handedness)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for the placeholder coordinates")
    parser.add_argument("--out-summed", type=Path, default=Path("test_summed.mrc"))
    parser.add_argument("--out-3d", type=Path, default=Path("test_3d.mrc"))
    args = parser.parse_args()

    images, angpix, ori_size, _ = read_selected_classes(
        args.class_star, args.class_mrcs, args.project_dir)

    down_size, down_angpix, yrect, xrect = compute_sizing(
        angpix, ori_size, args.crossover, args.maxres)

    # Downscale each class to the model sampling (RELION resizeMap).
    images_ds = [rec.fourier_resize(img, down_size) for img in images]

    coords_x, coords_y = get_coordinates(len(images_ds), xrect, seed=args.seed)

    aref = build_canvas(images_ds, coords_x, coords_y, yrect, xrect)
    rec.write_mrc(args.out_summed, aref, down_angpix)

    cross = rec.reconstruct2d(aref)
    if args.mask_diameter and args.mask_diameter > 0:
        cross = rec.apply_circular_mask(cross, args.mask_diameter / (2.0 * down_angpix))
    cross = rec.fourier_resize(cross, ori_size)
    volume = rec.reconstruct3d(cross, angpix, args.crossover, rot_sign=args.rot_sign)
    rec.write_mrc(args.out_3d, volume, angpix)

    print(f"classes={len(images_ds)}  angpix={angpix:.4f}  ori_size={ori_size}")
    print(f"down_size={down_size}  down_angpix={down_angpix:.5f}  "
          f"yrect={yrect}  xrect={xrect}")
    print(f"Wrote {args.out_summed}  shape={aref.shape}")
    print(f"Wrote {args.out_3d}  shape={volume.shape}")


if __name__ == "__main__":
    main()
