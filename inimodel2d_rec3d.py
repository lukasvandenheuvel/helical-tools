#!/usr/bin/env python
"""Rebuild a RELION ``relion_helix_inimodel2d`` ``_rec3d.mrc`` from its
``_projections.spi`` rectangle.

The ``_projections.spi`` image is ``model.Aref`` from the program: a rectangle
whose Y axis (height ``yrect``) is the projection coordinate across the fibril
cross-section and whose X axis (width ``xrect``) spans two cross-overs = 360 deg
of azimuthal rotation about the helical axis. In other words it is a *sinogram*
of the fibril cross-section.

This script reproduces the two RELION steps:

* ``reconstruct2D`` : invert the sinogram (filtered back-projection) to get the
  2D cross-section, then resize it from ``yrect`` (``down_angpix``) up to
  ``ori_size`` (``angpix``) and apply the circular mask.
* ``reconstruct3D``: sweep the cross-section up the Z axis, rotating slice ``k``
  by ``deg_per_pixel * k`` with ``deg_per_pixel = 180 * angpix / crossover``,
  which bakes in the helical twist.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import mrcfile
import numpy as np
from PIL import Image
from scipy.ndimage import rotate as nd_rotate


def read_spi(path: Path) -> np.ndarray:
    """Read a single-image SPIDER file into a float32 2D array."""
    with Image.open(path) as im:
        return np.array(im, dtype=np.float32)


def parse_crossover_distance(name: str) -> float:
    """Extract the crossover distance (Angstrom) from a job*_<dist>_ filename."""
    match = re.search(r"job\d+_(\d+(?:\.\d+)?)_", name)
    if not match:
        raise ValueError(f"Could not parse crossover distance from {name!r}")
    return float(match.group(1))


def angpix_orisize_from_mrcs(mrcs_path: Path) -> tuple[float, int]:
    with mrcfile.open(mrcs_path, permissive=True, header_only=True) as mrc:
        angpix = float(mrc.voxel_size.x)
        ori_size = int(mrc.header.nx)
    if angpix <= 0:
        raise ValueError(f"Pixel size missing from header of {mrcs_path}")
    return angpix, ori_size


def angpix_orisize_from_star(star_path: Path, project_dir: Path | None) -> tuple[float, int]:
    """Resolve angpix/ori_size via the first _rlnReferenceImage stack in a star."""
    stack_ref = None
    with open(star_path) as handle:
        for line in handle:
            if "@" in line and ".mrcs" in line:
                stack_ref = line.split()[0]
                break
    if stack_ref is None:
        raise ValueError(f"No _rlnReferenceImage stack found in {star_path}")

    mrcs_rel = stack_ref.split("@", 1)[1]
    root = project_dir if project_dir is not None else star_path.parents[2]
    mrcs_path = Path(mrcs_rel)
    if not mrcs_path.is_absolute():
        mrcs_path = root / mrcs_rel
    return angpix_orisize_from_mrcs(mrcs_path)


def reconstruct2d(sinogram: np.ndarray) -> np.ndarray:
    """Filtered back-projection of the crossover sinogram -> 2D cross-section.

    ``sinogram`` has shape (yrect, xrect); column ``j`` is a 1D projection taken
    at azimuthal angle ``j * 360 / xrect`` degrees.
    """
    n_proj, n_ang = sinogram.shape
    angles = np.deg2rad(np.arange(n_ang) * 360.0 / n_ang)

    # Ramp-filter every projection (column) along the projection coordinate.
    ramp = 2.0 * np.abs(np.fft.rfftfreq(n_proj))
    filtered = np.fft.irfft(np.fft.rfft(sinogram, axis=0) * ramp[:, None],
                            n=n_proj, axis=0).astype(np.float32)

    center = n_proj // 2
    coords = np.arange(n_proj) - center
    grid_x, grid_y = np.meshgrid(coords, coords)
    grid_y = -grid_y  # match RELION's CenterFFT/Fourier-insertion Y convention

    rec = np.zeros((n_proj, n_proj), dtype=np.float32)
    for j in range(n_ang):
        t = grid_x * np.cos(angles[j]) + grid_y * np.sin(angles[j]) + center
        i0 = np.floor(t).astype(np.intp)
        frac = (t - i0).astype(np.float32)
        in_range = (t >= 0.0) & (t <= n_proj - 1)
        i0c = np.clip(i0, 0, n_proj - 2)
        col = filtered[:, j]
        val = col[i0c] * (1.0 - frac) + col[i0c + 1] * frac
        rec += np.where(in_range, val, 0.0)

    rec *= np.pi / n_ang
    return rec


def fourier_resize(img: np.ndarray, new_size: int) -> np.ndarray:
    """Resize a square image by cropping/padding its (centered) Fourier transform."""
    old = img.shape[0]
    if new_size == old:
        return img.astype(np.float32)

    ft = np.fft.fftshift(np.fft.fft2(img))
    if new_size > old:
        pad = (new_size - old) // 2
        buf = np.zeros((new_size, new_size), dtype=complex)
        buf[pad:pad + old, pad:pad + old] = ft
    else:
        crop = (old - new_size) // 2
        buf = ft[crop:crop + new_size, crop:crop + new_size]

    out = np.fft.ifft2(np.fft.ifftshift(buf)).real * (new_size / old) ** 2
    return out.astype(np.float32)


def apply_circular_mask(img: np.ndarray, radius_pix: float) -> np.ndarray:
    """Hard circular mask: pixels outside the radius set to their mean (cosine_width=0)."""
    n = img.shape[0]
    c = n // 2
    yy, xx = np.ogrid[:n, :n]
    outside = (xx - c) ** 2 + (yy - c) ** 2 > radius_pix ** 2
    out = img.copy()
    if outside.any():
        out[outside] = out[outside].mean()
    return out


def reconstruct3d(cross_section: np.ndarray, angpix: float, crossover: float,
                  rot_sign: float = 1.0) -> np.ndarray:
    """Spin the 2D cross-section up Z, applying the helical twist per slice."""
    ori_size = cross_section.shape[0]
    deg_per_pixel = 180.0 * angpix / crossover
    vol = np.empty((ori_size, ori_size, ori_size), dtype=np.float32)
    for k in range(ori_size):
        ang = rot_sign * deg_per_pixel * k
        vol[k] = nd_rotate(cross_section, ang, reshape=False, order=1,
                           mode="constant", cval=0.0)
    return vol


def projections_to_rec3d(spi_path: Path, angpix: float, ori_size: int,
                         crossover: float, mask_diameter: float | None = None,
                         rot_sign: float = 1.0) -> np.ndarray:
    sinogram = read_spi(spi_path)
    yrect, xrect = sinogram.shape
    down_angpix = 2.0 * crossover / xrect

    cross = reconstruct2d(sinogram)
    if mask_diameter is not None and mask_diameter > 0:
        cross = apply_circular_mask(cross, mask_diameter / (2.0 * down_angpix))
    cross = fourier_resize(cross, ori_size)
    return reconstruct3d(cross, angpix, crossover, rot_sign=rot_sign)


def write_mrc(path: Path, volume: np.ndarray, angpix: float) -> None:
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(volume.astype(np.float32))
        mrc.voxel_size = angpix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("projections", type=Path,
                        help="Input *_projections.spi rectangle")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output *_rec3d.mrc (default: alongside input)")
    parser.add_argument("--angpix", type=float, default=None,
                        help="Pixel size (A). Overrides --class-mrcs/--class-star")
    parser.add_argument("--ori-size", type=int, default=None,
                        help="Original box size (px). Overrides --class-mrcs/--class-star")
    parser.add_argument("--class-mrcs", type=Path, default=None,
                        help="2D class stack to read angpix/ori_size from its header")
    parser.add_argument("--class-star", type=Path, default=None,
                        help="2D class star; angpix/ori_size read from its referenced stack")
    parser.add_argument("--project-dir", type=Path, default=None,
                        help="Project root for resolving paths in --class-star")
    parser.add_argument("--crossover", type=float, default=None,
                        help="Crossover distance (A); default parsed from filename")
    parser.add_argument("--mask-diameter", type=float, default=None,
                        help="Circular mask diameter (A) as used by --mask_diameter")
    parser.add_argument("--rot-sign", type=float, default=1.0,
                        help="Sign of the twist (use -1 to flip handedness)")
    args = parser.parse_args()

    if args.angpix is not None and args.ori_size is not None:
        angpix, ori_size = args.angpix, args.ori_size
    elif args.class_mrcs is not None:
        angpix, ori_size = angpix_orisize_from_mrcs(args.class_mrcs)
    elif args.class_star is not None:
        angpix, ori_size = angpix_orisize_from_star(args.class_star, args.project_dir)
    else:
        parser.error("Provide --angpix and --ori-size, or --class-mrcs, or --class-star")

    crossover = args.crossover or parse_crossover_distance(args.projections.name)

    volume = projections_to_rec3d(args.projections, angpix, ori_size, crossover,
                                  mask_diameter=args.mask_diameter, rot_sign=args.rot_sign)

    out = args.output
    if out is None:
        out = args.projections.with_name(
            args.projections.name.replace("_projections.spi", "_rec3d.mrc"))
    write_mrc(out, volume, angpix)
    print(f"Wrote {out}  shape={volume.shape}  angpix={angpix}  crossover={crossover}")


if __name__ == "__main__":
    main()
