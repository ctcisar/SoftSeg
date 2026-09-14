"""
Convert a SoftSeg-format dataset into a :class:`spatialdata.SpatialData` object.

The SoftSeg on-disk format (see :class:`SoftSeg.SoftAssigner.SoftAssigner`) is a
set of per-FOV files, addressed by format strings that are ``.format(fov)``-ed:

* ``csv_loc``   — per-FOV transcript CSV. Columns: ``x``, ``y`` (local, within
  the FOV), optional ``global_z``, a gene column, and a leading index column.
* ``mask_loc``  — per-FOV integer-labelled segmentation mask TIFF, 2D ``(y, x)``
  or 3D ``(z, y, x)`` (z is axis 0). Mask value ``m`` denotes cell id ``m - 1``;
  value 0 is background.
* ``image_loc`` — (optional) per-FOV raw microscopy image TIFF.

Per-FOV global placement is given by ``fov_locs``, the same structure SoftSeg's
``convert_to_adata`` consumes: ``{fov: {"x": [x0, ...], "y": [y0, ...]}}`` — the
FOV's global origin is ``(x0, y0)`` and a transcript's global position is its
local position plus that offset.

Design (mirrors :func:`spatialdata_io.cosmx`)
---------------------------------------------
* Every element is **split by FOV**: elements are named ``f"{fov}_image"``,
  ``f"{fov}_labels"``, ``f"{fov}_points"``.
* There is **one coordinate system per FOV** (named ``f"fov_{fov}"``) plus a
  shared ``"global"`` coordinate system, and two type-restricted global systems:
  ``"global_only_image"`` (images only) and ``"global_only_labels"`` (labels
  only), each carrying the same transform as ``"global"`` — mirroring
  :func:`spatialdata_io.cosmx`, so images or labels can be rendered/operated on
  in isolation. Points stay on ``fov`` + ``"global"`` only.
* Each element carries an ``Identity`` transform into its own FOV system and a
  ``Translation`` into ``"global"`` (the FOV offset). So the transcripts have
  **two sets of coordinates**: their raw ``x, y`` are their location *within the
  FOV* (the FOV coordinate system), and mapping to ``"global"`` gives their
  location *in the whole experiment*. For convenience the global position is also
  written explicitly into the points frame as ``global_x``/``global_y``.
* All raster elements are validated to xarray ``DataArray``\\s with **correctly
  labelled axes**, including ``z`` for 3D stacks (``("z", "y", "x")`` for masks,
  ``("c", "z", "y", "x")`` for images).

Pixel convention
----------------
The transcript CSVs' ``x``/``y`` are **1-based** with respect to the mask array:
SoftSeg's :meth:`~SoftSeg.SoftAssigner.SoftAssigner.blur_fov` tests a transcript
against a cell contour at ``(x - 1, y - 1)``, so transcript ``(x, y)`` belongs to
mask pixel ``[y - 1, x - 1]``. Points are stored here **verbatim** from the CSV
(so they still match the files on disk, and ``global_x``/``global_y`` stay
comparable across the dataset); the correction is carried by the *polygons*
instead — :func:`masks_to_shapes` shifts every contour vertex by ``+1``
(``polygon_offset``). Aggregating points into those shapes therefore reproduces
SoftSeg's own transcript-to-cell assignment exactly.

Z alignment
-----------
Transcript ``global_z`` values must index the mask's z-slices. Because the two
come from different files they can disagree (e.g. a 7-plane transcript table
against a 6-slice mask), which would silently misassign every transcript on the
extra plane. :func:`softseg_to_spatialdata` therefore validates the points' z
against the labels' z axis and **warns** when they do not line up — but keeps
every transcript, so the points element stays a faithful copy of the transcript
table. What to do about the strays is recorded as a policy in
``sdata.attrs[SOFTSEG_ATTRS]`` (``z_offset``, ``valid_z``, ``snap_z``) and acted
on by :func:`aggregate_zslice_shapes`, which is where they are dropped or
snapped. Images and labels are expected to have the same number of z-slices, and
a mismatch is warned about.
"""

from __future__ import annotations

import glob
import multiprocessing as mp
import os
import re
import threading
import warnings
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import anndata as ad
import cv2
import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import parse as _parse
import tifffile
import logging
from tqdm.auto import tqdm
from shapely.geometry import MultiPolygon, Polygon
from spatialdata import SpatialData, read_zarr
from spatialdata import aggregate as _sd_aggregate
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    ShapesModel,
)
from spatialdata.transformations import (
    Affine,
    Identity,
    Translation,
    get_transformation,
)


def _fov_coordinate_system(fov) -> str:
    """Name of the per-FOV coordinate system."""
    return f"fov_{fov}"


def _normalize_pixel_size(pixel_size):
    """Normalise a pixel-size spec to ``(sx, sy, sz)`` in physical units/pixel.

    Accepts ``None`` (no scaling), a scalar (isotropic), a 2-/3-sequence
    ``(sx, sy[, sz])`` (x, y, z order), or a dict with ``x``/``y``/``z`` keys.
    A missing z scale defaults to 1.0.
    """
    if pixel_size is None:
        return None
    if isinstance(pixel_size, Mapping):
        return (
            float(pixel_size.get("x", 1.0)),
            float(pixel_size.get("y", 1.0)),
            float(pixel_size.get("z", 1.0)),
        )
    if np.isscalar(pixel_size):
        s = float(pixel_size)
        return (s, s, s)
    seq = [float(v) for v in pixel_size]
    if len(seq) == 2:
        return (seq[0], seq[1], 1.0)
    if len(seq) == 3:
        return (seq[0], seq[1], seq[2])
    raise ValueError(f"pixel_size must be scalar, (sx, sy[, sz]), or dict; got {pixel_size!r}")


def _global_transform(offset, pixel_size, is_3d: bool):
    """Build the element's transform into the ``"global"`` coordinate system.

    Without ``pixel_size`` this is just the FOV ``Translation`` (pixel units, the
    original behaviour). With ``pixel_size`` the global system becomes *physical*:
    an ``Affine`` that applies the FOV pixel offset and then the per-axis pixel
    size, so ``global`` coords are ``(local_pixel + offset) * pixel_size`` (z is
    scaled by ``sz``). The real-world pixel size is therefore encoded in the
    coordinate system as the linear part of this transform.
    """
    x0, y0 = offset
    if pixel_size is None:
        return Translation([x0, y0], axes=("x", "y"))
    sx, sy, sz = pixel_size
    if is_3d:
        matrix = np.array(
            [
                [sx, 0.0, 0.0, sx * x0],
                [0.0, sy, 0.0, sy * y0],
                [0.0, 0.0, sz, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return Affine(matrix, input_axes=("x", "y", "z"), output_axes=("x", "y", "z"))
    matrix = np.array(
        [
            [sx, 0.0, sx * x0],
            [0.0, sy, sy * y0],
            [0.0, 0.0, 1.0],
        ]
    )
    return Affine(matrix, input_axes=("x", "y"), output_axes=("x", "y"))


def _fov_offset(fov_locs, fov) -> tuple[float, float]:
    """Extract the (x0, y0) global origin of a FOV from a fov_locs entry.

    Accepts the SoftSeg dict form ``{"x": [x0, ...], "y": [y0, ...]}`` as well as
    a plain ``(x0, y0)`` / ``[x0, y0]`` pair.
    """
    if fov_locs is None:
        return 0.0, 0.0
    # FOVs discovered from filenames are strings; fov_locs may be keyed by int.
    if fov in fov_locs:
        entry = fov_locs[fov]
    elif str(fov) in fov_locs:
        entry = fov_locs[str(fov)]
    else:
        try:
            entry = fov_locs[int(fov)]
        except (KeyError, ValueError, TypeError):
            raise KeyError(f"FOV {fov!r} not found in fov_locs keys {list(fov_locs)}")
    if isinstance(entry, Mapping):
        x0 = entry["x"][0] if hasattr(entry["x"], "__len__") else entry["x"]
        y0 = entry["y"][0] if hasattr(entry["y"], "__len__") else entry["y"]
    else:  # (x0, y0) pair
        x0, y0 = entry[0], entry[1]
    return float(x0), float(y0)


def _resolve_image_dims(image_dims, fov):
    """Resolve the axis labels to use for a given FOV's image.

    ``image_dims`` may be a single sequence applied to every image, or a mapping
    ``{fov: dims}`` giving per-image labels (FOVs from filenames are strings, so
    str/int keys are both accepted). Returns ``None`` when no labels are given
    for this image (triggering the automatic axis inference in :func:`_prep_image`).
    """
    if image_dims is None:
        return None
    if isinstance(image_dims, Mapping):
        if fov in image_dims:
            return image_dims[fov]
        if str(fov) in image_dims:
            return image_dims[str(fov)]
        try:
            if int(fov) in image_dims:
                return image_dims[int(fov)]
        except (ValueError, TypeError):
            pass
        return None
    return image_dims


def _prep_image(arr, image_dims: Optional[Sequence[str]] = None):
    """Return ``(array, dims)`` for an image with a correctly labelled, present
    channel axis.

    Handles the shapes SoftSeg images come in and guarantees a ``c`` axis
    (spatialdata's image models require one):

    * 2D ``(y, x)``       -> promote to ``(1, y, x)``, dims ``("c", "y", "x")``
    * 3D ``(c, y, x)``    -> dims ``("c", "y", "x")``             (multichannel 2D)
    * 4D                  -> the axis of length 3 is taken to be the colour axis
      (``"c"``) and the remaining axes are labelled ``z, y, x`` in order; if no
      axis has length 3, ``("c", "z", "y", "x")`` is assumed.

    Pass ``image_dims`` explicitly to override the inference (e.g. to disambiguate
    a 3D array as a single-channel ``("z", "y", "x")`` stack, or to name a 4D
    image whose colour axis is not length 3). ``image_dims`` describes the axis
    order of ``arr``.

    A 4D image is physically transposed into canonical ``("c", "z", "y", "x")``
    order (required by spatialdata) before being returned.
    """
    arr = da.asarray(arr)
    ndim = arr.ndim

    if image_dims is not None:
        dims = tuple(image_dims)
        if "c" not in dims:  # single-channel stack: add the channel axis
            arr = arr[None, ...]
            dims = ("c",) + dims
    elif ndim == 2:  # (y, x) grayscale
        arr = arr[None, ...]
        dims = ("c", "y", "x")
    elif ndim == 3:  # assume multichannel 2D (c, y, x)
        dims = ("c", "y", "x")
    elif ndim == 4:
        # A 4D image is (c, z, y, x) in some order: treat the length-3 axis as
        # the colour channel (RGB) rather than the z axis, labelling the rest
        # z, y, x in their existing order.
        c_axes = [i for i, length in enumerate(arr.shape) if length == 3]
        if c_axes:
            c_axis = c_axes[0]
            remaining = iter(("z", "y", "x"))
            dims = tuple("c" if i == c_axis else next(remaining) for i in range(4))
        else:
            dims = ("c", "z", "y", "x")
    else:
        raise ValueError(f"Unsupported image ndim={ndim}; pass image_dims explicitly.")

    # spatialdata requires 4D images in (c, z, y, x) order: physically rearrange
    # the axes so the stored array matches, rather than relying on parse to do it.
    canonical = ("c", "z", "y", "x")
    if len(dims) == 4 and tuple(dims) != canonical:
        order = [dims.index(ax) for ax in canonical]
        arr = da.transpose(arr, axes=order)
        dims = canonical

    return arr, dims


def _parse_image(arr, fov, offset, image_dims=None, pixel_size=None, **image_models_kwargs):
    """Parse a raw image array into a spatialdata image element.

    Placed in the FOV system (``Identity``), the shared ``"global"`` system, and
    an image-only ``"global_only_image"`` system (same transform as ``"global"``)
    so images can be operated on/rendered in isolation.
    """
    fov_cs = _fov_coordinate_system(fov)
    arr, dims = _prep_image(arr, image_dims)
    is_3d = "z" in dims
    transformations = {
        fov_cs: Identity(),
        "global": _global_transform(offset, pixel_size, is_3d),
        "global_only_image": _global_transform(offset, pixel_size, is_3d),
    }
    model = Image3DModel if is_3d else Image2DModel
    return model.parse(arr, dims=dims, transformations=transformations, **image_models_kwargs)


def _parse_mask(arr, fov, offset, pixel_size=None, **image_models_kwargs):
    """Parse an integer mask array into a spatialdata labels element with
    correctly labelled axes.

    Placed in the FOV system (``Identity``), the shared ``"global"`` system, and
    a labels-only ``"global_only_labels"`` system (same transform as ``"global"``)
    so labels can be operated on/rendered in isolation.
    """
    fov_cs = _fov_coordinate_system(fov)
    arr = da.asarray(arr)
    if arr.ndim == 2:  # (y, x)
        transformations = {
            fov_cs: Identity(),
            "global": _global_transform(offset, pixel_size, False),
            "global_only_labels": _global_transform(offset, pixel_size, False),
        }
        return Labels2DModel.parse(
            arr, dims=("y", "x"), transformations=transformations, **image_models_kwargs
        )
    if arr.ndim == 3:  # (z, y, x)
        transformations = {
            fov_cs: Identity(),
            "global": _global_transform(offset, pixel_size, True),
            "global_only_labels": _global_transform(offset, pixel_size, True),
        }
        return Labels3DModel.parse(
            arr, dims=("z", "y", "x"), transformations=transformations, **image_models_kwargs
        )
    raise ValueError(f"Unsupported mask ndim={arr.ndim}; expected 2 (y,x) or 3 (z,y,x).")


# Key under which this module's own bookkeeping lives in ``SpatialData.attrs``.
SOFTSEG_ATTRS = "softseg"


def _softseg_attrs(sdata: SpatialData) -> dict:
    """This module's bookkeeping from ``sdata.attrs``, or ``{}`` if absent.

    A SpatialData assembled by hand (or round-tripped through something that
    drops ``attrs``) simply has no recorded policy, and callers fall back to
    their own defaults.
    """
    attrs = getattr(sdata, "attrs", None) or {}
    recorded = attrs.get(SOFTSEG_ATTRS) or {}
    return recorded if isinstance(recorded, Mapping) else {}


def _resolve_snap_z(sdata: SpatialData, snap_z: Optional[bool]) -> bool:
    """The z-correction policy in force: explicit argument, else what was
    recorded at conversion, else drop."""
    if snap_z is not None:
        return bool(snap_z)
    return bool(_softseg_attrs(sdata).get("snap_z", False))


def _recorded_valid_z(sdata: SpatialData, fov) -> Optional[list]:
    """The valid z-slices recorded for a FOV, or ``None`` if none were recorded.

    ``fov`` is either a FOV token or a list of ``"{fov}_z{z}_shapes"`` names; in
    the latter case the FOV is taken from the first name.
    """
    valid = _softseg_attrs(sdata).get("valid_z") or {}
    if not valid:
        return None
    if isinstance(fov, (list, tuple)):
        if not len(fov):
            return None
        m = re.match(r"^(.*)_z\d+_shapes$", str(fov[0]))
        fov = m.group(1) if m else str(fov[0])
    fov = str(fov)
    entry = valid.get(fov)
    if entry is None:
        try:
            entry = valid.get(str(int(fov)))
        except (TypeError, ValueError):
            entry = None
    return list(entry) if entry is not None else None


def _assign_points_to_slices(zvals, allowed, snap_z: bool):
    """Map each point's z to the valid slice index it belongs to.

    ``allowed`` is the sorted array of valid slice indices. A point belongs to
    the slice its z rounds to (within half a slice; ties go to the lower slice).
    Points that round to no valid slice are either snapped to the nearest one
    (``snap_z``) or marked ``NaN``, i.e. belonging to no slice at all.

    Returns ``(slice_of_point, off)`` where ``off`` flags the points that did not
    line up with any valid slice.
    """
    nearest = allowed[np.argmin(np.abs(zvals[:, None] - allowed[None, :]), axis=1)]
    off = np.abs(zvals - nearest) > 0.5
    if off.any() and not snap_z:
        nearest = nearest.astype(float, copy=True)
        nearest[off] = np.nan
    return nearest, off


def _align_points_z(df, fov, valid_z=None, z_offset=0, snap_z=False):
    """Give a transcript table a ``z`` column and warn if it misses the segmentation.

    Adds a ``z`` column holding ``global_z + z_offset`` — the slice index the
    transcript is taken to sit on — while leaving ``global_z`` untouched, so the
    raw value from the CSV stays available for provenance.

    A transcript "lines up" with the segmentation if its ``z`` rounds to one of
    ``valid_z`` (i.e. is within half a slice of it). Ones that do not are
    **kept**: this function only warns, so the points element stays a faithful
    copy of the transcript table. Which of them end up in a cell is decided later
    by :func:`aggregate_zslice_shapes`, per the ``snap_z`` preference recorded in
    :data:`SOFTSEG_ATTRS`. ``snap_z`` is passed in here only so the warning can
    say what will happen.

    A misalignment *more than a whole slice* wide is called out separately,
    because it usually means the two z axes are genuinely offset rather than
    merely differing in extent.

    Returns the frame with its ``z`` column; a table with no ``global_z`` is
    returned unchanged.
    """
    if "global_z" not in df.columns:
        return df

    z_raw = df["global_z"].to_numpy(dtype=float)
    df = df.copy()
    df["z"] = z_raw + z_offset

    if valid_z is None or len(valid_z) == 0:
        return df  # nothing to validate against

    slices = np.asarray(sorted({int(v) for v in valid_z}), dtype=float)
    z = df["z"].to_numpy(dtype=float)
    gap = np.abs(z - slices[np.argmin(np.abs(z[:, None] - slices[None, :]), axis=1)])
    off = gap > 0.5  # does not round to any valid slice

    if off.any():
        bad = sorted({float(v) for v in z_raw[off]})
        shown = ", ".join(f"{v:g}" for v in bad[:10]) + ("..." if len(bad) > 10 else "")
        msg = (
            f"FOV {fov!r}: {int(off.sum())} of {len(df)} transcripts have a z that "
            f"does not line up with a segmentation z-slice (valid slices "
            f"{slices.min():g}-{slices.max():g}; offending global_z values: {shown})"
        )
        if z_offset:
            msg += f", with z_offset={z_offset} applied"
        msg += "."
        far = int((gap > 1).sum())
        if far:
            msg += (
                f" {far} of them are more than a whole slice away from the nearest "
                "valid slice, which usually means the transcript and image z axes "
                "are genuinely offset rather than merely different in extent; "
                "consider passing z_offset= or valid_z=."
            )
        msg += (
            " They are kept in the points element; aggregation will "
            + ("snap them to the nearest slice" if snap_z else "drop them")
            + f" (recorded as snap_z={snap_z} in sdata.attrs[{SOFTSEG_ATTRS!r}])."
        )
        warnings.warn(msg, stacklevel=3)

    return df


def _parse_points(df, fov, offset, gene_col="gene", pixel_size=None):
    """Parse a per-FOV transcript table into a spatialdata points element.

    Local ``x, y`` (and the slice index ``z``, see :func:`_align_points_z`) are
    stored as the point coordinates; ``Identity`` maps them into the FOV system
    and a ``Translation`` into ``"global"``. ``global_x``/``global_y`` are also
    added as explicit columns so both coordinate sets are directly available.

    ``x``/``y`` are kept exactly as they appear in the CSV — the 1-pixel offset
    between the transcript coordinates and the mask array is carried by the
    polygons instead (see :func:`masks_to_shapes`).
    """
    fov_cs = _fov_coordinate_system(fov)
    x0, y0 = offset
    df = df.copy()
    df["global_x"] = df["x"] + x0
    df["global_y"] = df["y"] + y0

    coordinates = {"x": "x", "y": "y"}
    z_col = "z" if "z" in df.columns else ("global_z" if "global_z" in df.columns else None)
    is_3d = z_col is not None
    if is_3d:
        coordinates["z"] = z_col

    kwargs: dict[str, Any] = {}
    if gene_col in df.columns:
        kwargs["feature_key"] = gene_col

    return PointsModel.parse(
        df,
        coordinates=coordinates,
        transformations={
            fov_cs: Identity(),
            "global": _global_transform(offset, pixel_size, is_3d),
        },
        **kwargs,
    )


def _discover_fovs(loc: str) -> list:
    """Infer FOV identifiers from a ``.format(fov)`` path pattern on disk."""
    hits = glob.glob(loc.format("****"))
    fovs = []
    for h in hits:
        parsed = _parse.parse(loc, h)
        if parsed is not None:
            fovs.append(parsed[0])
    return fovs


def softseg_to_spatialdata(
    csv_loc: str,
    mask_loc: str,
    fov_locs: Optional[Mapping] = None,
    *,
    image_loc: Optional[str] = None,
    fovs: Optional[Sequence] = None,
    gene_col: str = "gene",
    image_dims: Optional[Union[Sequence[str], Mapping]] = None,
    pixel_size=None,
    valid_z: Optional[Sequence[int]] = None,
    z_offset: int = 0,
    snap_z: bool = False,
    imread_kwargs: Mapping[str, Any] = {},
    image_models_kwargs: Mapping[str, Any] = {},
    save_loc: str = None,
) -> SpatialData:
    """Convert a SoftSeg-format dataset to a :class:`spatialdata.SpatialData`.

    Parameters
    ----------
    csv_loc, mask_loc, image_loc
        ``.format(fov)`` path patterns for the per-FOV transcript CSV, the
        integer segmentation mask TIFF, and (optionally) the raw image TIFF.
    fov_locs
        Per-FOV global origins, ``{fov: {"x": [x0, ...], "y": [y0, ...]}}`` (the
        SoftSeg form) or ``{fov: (x0, y0)}``. If ``None``, every FOV is placed at
        the origin (all-Identity to ``"global"``).
    fovs
        Which FOVs to convert. If ``None``, inferred from the files matching
        both ``csv_loc`` and ``mask_loc``.
    gene_col
        Name of the gene/feature column in the transcript CSV.
    image_dims
        Axis labels for raw images. Either a single sequence applied to **all**
        images, or a mapping ``{fov: dims}`` giving labels **per image object**
        (str/int FOV keys both accepted); FOVs absent from the mapping fall back
        to automatic inference. Use it when the array is ambiguous (e.g. a 3D
        image that is a ``("z", "y", "x")`` single-channel stack rather than
        ``("c", "y", "x")``). When not given, axes are inferred; in particular a
        4D image's length-3 axis is taken as the colour channel, not z. See
        :func:`_prep_image`.
    pixel_size
        Real-world size of a pixel/voxel, encoded into the ``"global"`` coordinate
        system as a ``Scale`` (via an ``Affine`` that also carries the FOV offset),
        following the spatialdata convention for physical units (as used by the
        Xenium/MACSima readers). Accepts a scalar (isotropic), ``(sx, sy[, sz])``
        in x, y, z order, or a dict with ``x``/``y``/``z`` keys. The ``sz``
        component sets the physical spacing between z-slices, which
        :func:`masks_to_shapes` reads back to place the per-slice polygons. If
        ``None`` (default) ``"global"`` stays in pixel units (a pure translation).
        The per-FOV ``f"fov_{fov}"`` system always remains in raw pixels.
    valid_z
        The z-slice indices a transcript's ``global_z`` may refer to. If ``None``
        (default) they are inferred per FOV from the label mask (``range(n_z)``).
        Pass an explicit list when only some planes of the mask are meaningful.
        The resolved per-FOV sets are recorded in ``attrs``.
    z_offset
        Added to every ``global_z`` before it is matched against the valid
        slices, for datasets whose transcript z axis is shifted relative to the
        imaging planes (e.g. ``z_offset=-1`` for a 1-based transcript z). The
        shifted value is stored as the points' ``z`` coordinate; the original
        ``global_z`` column is preserved alongside it.
    snap_z
        The **preferred correction** for transcripts whose z does not round to a
        valid slice. ``False`` (default) means they should be dropped, matching
        :meth:`~SoftSeg.SoftAssigner.SoftAssigner.blur_fov`, which only considers
        transcripts sitting on an actual mask plane; ``True`` means they should
        be snapped to the nearest valid slice. This is a *policy*, not an action:
        nothing is dropped here — the points element stays a faithful copy of the
        transcript table, a warning reports the misalignment, and the policy is
        recorded in ``attrs`` for :func:`aggregate_zslice_shapes` to apply.
    imread_kwargs
        Passed to :func:`tifffile.imread` (preserves TIFF page order as z,y,x;
        unlike ``skimage.io.imread`` it never reinterprets a leading 3/4-length
        axis as channels).
    image_models_kwargs
        Passed to the image/labels model ``parse`` (e.g. ``chunks``, ``scale_factors``).
    save_loc
        If provided, SpatialData will be saved to this location. Required when the
        contents of the whole SpatialData object is too large to be stored in memory.
    Returns
    -------
    spatialdata.SpatialData
        With ``images``/``labels``/``points`` split per FOV, a coordinate system
        per FOV (``f"fov_{fov}"``), a shared ``"global"`` system, and the
        type-restricted ``"global_only_image"`` / ``"global_only_labels"``
        systems (images / labels respectively).

        ``attrs[SOFTSEG_ATTRS]`` carries the z-correction policy:

        * ``"z_offset"`` — the offset already folded into the points' ``z``.
        * ``"snap_z"``   — whether transcripts that miss a slice should be
          snapped (``True``) or dropped (``False``) when aggregating.
        * ``"valid_z"``  — ``{fov: [slice indices]}``, the slices each FOV's
          segmentation actually covers.
    """
    if fovs is None:
        csv_fovs = set(_discover_fovs(csv_loc))
        mask_fovs = set(_discover_fovs(mask_loc))
        fovs = sorted(csv_fovs & mask_fovs, key=lambda v: (str(type(v)), v))
        if not fovs:
            raise FileNotFoundError(
                f"No FOVs found matching both {csv_loc!r} and {mask_loc!r}."
            )

    pbar = tqdm(total=len(fovs))

    pixel_size = _normalize_pixel_size(pixel_size)

    valid_z_by_fov: dict = {}
    our_sd = SpatialData()
    if save_loc is not None:
        our_sd.write(Path(save_loc), overwrite=True)

        # ignore uneeded warning when converting dataset
        logging.getLogger("ome_zarr.reader").addFilter(
            lambda r: not r.getMessage().startswith("no parent found for")
        )

    for fov in fovs:
        offset = _fov_offset(fov_locs, fov)
        if save_loc is not None:
            del our_sd
            our_sd = read_zarr(Path(save_loc))

        # --- segmentation mask -> labels (per FOV) ---
        n_label_z = None
        mask_path = mask_loc.format(fov)
        if Path(mask_path).is_file():
            mask = tifffile.imread(mask_path, **imread_kwargs)
            element = _parse_mask(
                mask, fov, offset, pixel_size=pixel_size, **image_models_kwargs
            )
            our_sd.labels[f"{fov}_labels"] = element
            if save_loc is not None:
                our_sd.write_element(f"{fov}_labels")
            n_label_z = int(element.sizes["z"]) if "z" in element.dims else 1

        # --- raw image -> image (per FOV) ---
        if image_loc is not None:
            image_path = image_loc.format(fov)
            if Path(image_path).is_file():
                im = tifffile.imread(image_path, **imread_kwargs)
                element = _parse_image(
                    im,
                    fov,
                    offset,
                    image_dims=_resolve_image_dims(image_dims, fov),
                    pixel_size=pixel_size,
                    **image_models_kwargs,
                )
                our_sd.images[f"{fov}_image"] = element
                if save_loc is not None:
                    our_sd.write_element(f"{fov}_image")
                # images and labels describe the same stack, so they must agree
                # on how many planes it has
                n_image_z = int(element.sizes["z"]) if "z" in element.dims else 1
                if n_label_z is not None and n_image_z != n_label_z:
                    warnings.warn(
                        f"FOV {fov!r}: the raw image has {n_image_z} z-slice(s) but "
                        f"the segmentation mask has {n_label_z}; images and labels "
                        "are expected to cover the same planes. Transcript z values "
                        "are validated against the mask.",
                        stacklevel=2,
                    )
                del im

        # The slices a transcript may sit on: whatever the caller declared, else
        # the planes the FOV's mask actually has. Recorded per FOV so the
        # aggregation step knows the valid set without re-reading the masks.
        if valid_z is not None:
            fov_valid_z = sorted({int(v) for v in valid_z})
        elif n_label_z is not None:
            fov_valid_z = list(range(n_label_z))
        else:  # no mask for this FOV: nothing to validate against
            fov_valid_z = None
        if fov_valid_z is not None:
            valid_z_by_fov[str(fov)] = fov_valid_z

        # --- transcripts -> points (per FOV, local + global coords) ---
        csv_path = csv_loc.format(fov)
        if Path(csv_path).is_file():
            df = pd.read_csv(csv_path, index_col=0).reset_index()
            df = _align_points_z(
                df, fov, valid_z=fov_valid_z, z_offset=z_offset, snap_z=snap_z
            )
            if len(df) > 0:
                our_sd.points[f"{fov}_points"] = _parse_points(
                    df.reset_index(drop=True),
                    fov,
                    offset,
                    gene_col=gene_col,
                    pixel_size=pixel_size,
                )

                if save_loc is not None:
                    our_sd.write_element(f"{fov}_points")
            del df
        pbar.update(1)
    pbar.close()

    # Record how z should be corrected so downstream steps (aggregation) apply
    # the same policy without being told again. z_offset is provenance: it has
    # already been folded into the points' "z" column.
    our_sd.attrs = {
        SOFTSEG_ATTRS: {
            "z_offset": z_offset,
            "snap_z": snap_z,
            "valid_z": valid_z_by_fov,
        }
    }
    if save_loc is not None:
        our_sd.write_attrs()
    return our_sd
    # return SpatialData(images=images, labels=labels, points=points, attrs=attrs)


# --------------------------------------------------------------------------- #
# Segmentation masks -> Shapes                                                 #
# --------------------------------------------------------------------------- #

# Name of the shared 3D coordinate system into which every per-slice, per-FOV
# shapes element is placed at its correct (x, y, z) location. This is the
# coordinate system that "accounts for" splitting a 3D stack into separate 2D
# shapes elements (geopandas has no 3D polygons).
STACK_3D_CS = "global_3d"


def _polygon_parts(geom) -> list:
    """Flatten a geometry into its constituent :class:`Polygon`\\s.

    ``Polygon.buffer(0)``, used to repair a self-intersecting contour, does not
    always return a ``Polygon``: repairing a ring that touches itself splits it
    into separate pieces and yields a ``MultiPolygon``, and a contour with
    degenerate slivers can yield a ``GeometryCollection`` holding lines or points
    alongside the polygons. Those results have to be flattened before they can go
    into a ``MultiPolygon``, which rejects a sequence that itself contains
    multi-polygons (*"Sequences of multi-polygons are not valid arguments"*).

    Non-areal parts (lines, points) are discarded — they enclose no pixels, so
    they cannot hold a transcript.
    """
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    parts: list = []
    for sub in getattr(geom, "geoms", ()):  # MultiPolygon / GeometryCollection
        parts.extend(_polygon_parts(sub))
    return parts


def _slice_to_polygons(slice_labels, min_points: int = 3, offset: float = 1.0):
    """Extract one polygon per cell label in a single 2D labelled slice.

    Uses ``cv2.findContours`` on each label's binary mask. cv2 returns contours
    as ``(N, 1, 2)`` in ``(x, y)`` order, which is already the polygon coordinate
    order. A label with several disjoint external contours — or whose contour
    repair splits it, see :func:`_polygon_parts` — becomes a ``MultiPolygon``.
    Returns ``{cell_id: geometry}`` with the SoftSeg convention
    ``cell_id = mask_value - 1``.

    ``offset`` is added to every vertex to move the contour out of mask-array
    index space and into the transcripts' coordinate space; see
    :func:`masks_to_shapes`.
    """
    slice_labels = np.asarray(slice_labels)
    geoms: dict = {}
    for m in np.unique(slice_labels):
        if m == 0:  # background
            continue
        binimg = (slice_labels == m).astype(np.uint8)
        contours = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        polys = []
        for cnt in contours:
            pts = np.asarray(cnt).reshape(-1, 2)
            if len(pts) < min_points:
                continue
            poly = Polygon([(float(x) + offset, float(y) + offset) for x, y in pts])
            if not poly.is_valid:
                poly = poly.buffer(0)
            # buffer(0) may have split the ring into several polygons, so collect
            # the parts flat rather than nesting them -- see _polygon_parts
            polys.extend(_polygon_parts(poly))
        if not polys:
            continue

        geom = polys[0] if len(polys) == 1 else MultiPolygon(polys)
        geoms[int(m) - 1] = geom  # mask value m -> cell id m-1
    return geoms


def _labels_global_affine(labels_element, fov) -> tuple[str, np.ndarray]:
    """Recover a labels element's FOV coordinate-system name and its pixel->global
    mapping as a homogeneous ``(x, y, z) -> (x, y, z)`` 4x4 affine matrix.

    This is the transform :func:`softseg_to_spatialdata` attached to ``"global"``:
    it carries both the FOV offset and, when a ``pixel_size`` was given, the
    physical scale. Its ``z`` scale (``M[2, 2]``) is the real-world spacing
    between z-slices. A 2D element's matrix is promoted to 4x4 with a unit,
    zero-offset z axis. Falls back to identity if transforms can't be read.
    """
    fov_cs = _fov_coordinate_system(fov)
    matrix = np.eye(4)
    try:
        transforms = get_transformation(labels_element, get_all=True)
        # Identify the element's FOV coordinate system: prefer the canonical
        # ``fov_{fov}`` name, else the first system that is neither the shared
        # ``"global"`` / ``"global_only_*"`` systems nor the 3D stack system.
        if fov_cs not in transforms:
            for name in transforms:
                if name == "global" or name.startswith("global_only") or name == STACK_3D_CS:
                    continue
                fov_cs = name
                break
        t = transforms["global"]
        if "z" in getattr(labels_element, "dims", ()):
            matrix = np.asarray(
                t.to_affine_matrix(input_axes=("x", "y", "z"), output_axes=("x", "y", "z"))
            )
        else:  # 2D: embed into 4x4 with an identity z axis
            m2 = np.asarray(t.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")))
            matrix = np.eye(4)
            matrix[np.ix_([0, 1], [0, 1])] = m2[:2, :2]
            matrix[[0, 1], 3] = m2[:2, 2]
    except Exception:
        pass
    return fov_cs, matrix


def _stack_affine_from_matrix(M: np.ndarray, z_index: int, z_size: float) -> Affine:
    """Build the per-slice ``(x, y) -> (x, y, z)`` transform into the 3D stack CS.

    ``M`` is the labels' pixel->global 4x4 affine (:func:`_labels_global_affine`),
    which supplies the x/y placement (offset + physical scale). The slice is
    pinned to ``z = z_index * z_size``, where ``z_size`` is the physical z-spacing
    (the labels' encoded z pixel size, or an explicit override).
    """
    matrix = np.array(
        [
            [M[0, 0], M[0, 1], M[0, 2] * z_index + M[0, 3]],
            [M[1, 0], M[1, 1], M[1, 2] * z_index + M[1, 3]],
            [0.0, 0.0, z_size * z_index + M[2, 3]],
            [0.0, 0.0, 1.0],
        ]
    )
    return Affine(matrix, input_axes=("x", "y"), output_axes=("x", "y", "z"))


def _global_2d_from_matrix(M: np.ndarray) -> Affine:
    """The 2D ``(x, y) -> (x, y)`` projection of the labels' pixel->global affine,
    used as the shapes' ``"global"`` transform (physical x/y when pixel_size set)."""
    matrix = np.array(
        [
            [M[0, 0], M[0, 1], M[0, 3]],
            [M[1, 0], M[1, 1], M[1, 3]],
            [0.0, 0.0, 1.0],
        ]
    )
    return Affine(matrix, input_axes=("x", "y"), output_axes=("x", "y"))


def _masks_to_shapes_worker(payload):
    """Build the per-z-slice shapes for a single FOV. Module-level so it can be
    dispatched to a multiprocessing ``Pool``.

    ``payload`` = ``(fov, fov_cs, M, z_size, stack, min_points, polygon_offset)``
    where ``stack`` is the FOV mask as a ``(z, y, x)`` numpy array. Returns
    ``{shape_element_name: parsed_shapes_gdf}``.
    """
    fov, fov_cs, M, z_size, stack, min_points, polygon_offset = payload
    global_2d = _global_2d_from_matrix(M)
    out: dict = {}
    for z in range(stack.shape[0]):
        geoms = _slice_to_polygons(stack[z], min_points=min_points, offset=polygon_offset)
        if not geoms:
            continue
        gdf = gpd.GeoDataFrame(
            {"cell_id": list(geoms.keys())},
            geometry=list(geoms.values()),
        )
        out[f"{fov}_z{z}_shapes"] = ShapesModel.parse(
            gdf,
            transformations={
                fov_cs: Identity(),
                "global": global_2d,
                STACK_3D_CS: _stack_affine_from_matrix(M, z, z_size),
            },
        )
    return out


def masks_to_shapes(
    sdata: SpatialData,
    *,
    labels_keys: Optional[Sequence[str]] = None,
    dist_between_slices: Optional[float] = None,
    min_points: int = 3,
    polygon_offset: float = 1.0,
    pool_size: Optional[int] = None,
    inplace: bool = True,
) -> SpatialData:
    """Convert the segmentation-mask labels of an SdSpatialData object to Shapes.

    For every labelled mask, each z-slice is contoured with ``cv2.findContours``
    and turned into its **own** shapes element (geopandas has no 3D polygons), so
    a 3D FOV mask yields one shapes key per slice: ``f"{fov}_z{z}_shapes"`` (2D
    masks give ``f"{fov}_z0_shapes"``).

    Each shapes element is placed into three coordinate systems, all derived from
    the labels element's own transforms so the shapes inherit any physical
    ``pixel_size`` that :func:`softseg_to_spatialdata` encoded:

    * ``f"fov_{fov}"`` — ``Identity``; the polygons' raw ``(x, y)`` within the FOV.
    * ``"global"``     — the labels' pixel->global x/y mapping (FOV offset, plus
      physical scale when a ``pixel_size`` was set).
    * :data:`STACK_3D_CS` (``"global_3d"``) — an ``Affine`` mapping ``(x, y)`` to
      ``(x, y, z)`` that applies the same x/y placement **and** pins the slice to
      its z-plane. This single 3D system reunites all the per-slice/per-FOV
      elements into one coherent 3D space — it is the coordinate system that
      accounts for the z-slice splitting.

    The **distance between slices is taken from the encoded pixel size**: the
    z-spacing is the ``z`` scale of the labels element's ``"global"`` transform
    (i.e. the physical voxel depth passed as ``pixel_size``'s ``sz``), so slice
    ``i`` sits at ``z = i * sz``. Pass ``dist_between_slices`` to override it.

    Cell ids follow the SoftSeg convention (``cell_id = mask_value - 1``) and are
    stored in a ``cell_id`` column.

    Every contour vertex is shifted by ``polygon_offset`` (``+1`` by default).
    ``cv2.findContours`` returns vertices as mask-array indices, whereas the
    transcript CSVs' ``x``/``y`` are 1-based with respect to that array — SoftSeg's
    :meth:`~SoftSeg.SoftAssigner.SoftAssigner.blur_fov` tests a transcript against
    a contour at ``(x - 1, y - 1)``. Shifting the polygons rather than the points
    puts the two in the same space while leaving the transcript coordinates as
    they appear on disk, so aggregating points into these shapes reproduces
    SoftSeg's own assignment. Pass ``polygon_offset=0`` for polygons in raw
    mask-index coordinates.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        Object containing labels elements (e.g. from :func:`softseg_to_spatialdata`).
    labels_keys : sequence of str, optional
        Which labels elements to convert. Defaults to all of ``sdata.labels``.
    dist_between_slices : float, optional
        Physical spacing between adjacent z-slices. If ``None`` (default), it is
        read from the labels element's encoded pixel size (the ``z`` scale of its
        ``"global"`` transform); pass a value to override that.
    min_points : int
        Minimum contour vertices for a valid polygon.
    polygon_offset : float
        Added to every contour vertex, moving the polygons from mask-array index
        space into the transcripts' coordinate space. Defaults to ``1.0``; see
        above.
    pool_size : int, optional
        Number of worker processes; FOVs are contoured in parallel, one per
        process. If ``None`` (default), uses ``min(n_fovs, os.cpu_count())``.
        A value of 1 (or a single FOV) runs serially with no pool overhead.
        FOV masks are read lazily: at most ``pool_size`` FOVs (those currently
        being worked on) are held in memory at once, rather than loading every
        FOV up front.
    inplace : bool
        If True, add the shapes to ``sdata`` and return it; otherwise return a
        new SpatialData holding only the shapes.

    Returns
    -------
    spatialdata.SpatialData
    """
    if labels_keys is None:
        labels_keys = list(sdata.labels)

    n = len(labels_keys)
    if pool_size is None:
        pool_size = min(n, os.cpu_count() or 1)
    pool_size = max(1, min(pool_size, n)) if n else 1  # never more workers than FOVs

    def _make_payload(key):
        """Materialise a single FOV's worker payload (transforms + mask stack)."""
        labels_element = sdata.labels[key]
        # element key is "{fov}_labels" -> recover the fov token
        fov = key[: -len("_labels")] if key.endswith("_labels") else key

        # Pixel->global affine (offset + physical scale) from the labels element.
        fov_cs, M = _labels_global_affine(labels_element, fov)
        # z-spacing: encoded pixel size (M[2, 2]) unless explicitly overridden.
        z_size = float(M[2, 2]) if dist_between_slices is None else float(dist_between_slices)

        # Use the element's labelled axes (not positional order) to build a
        # (z, y, x) stack, so slicing is correct regardless of storage order.
        if "z" in labels_element.dims:
            stack = np.asarray(labels_element.transpose("z", "y", "x"))
        else:
            stack = np.asarray(labels_element.transpose("y", "x"))[None, ...]

        return (fov, fov_cs, M, z_size, stack, min_points, polygon_offset)

    shapes: dict = {}
    if pool_size == 1:
        # Serial: one FOV materialised at a time, released before the next.
        for key in labels_keys:
            shapes.update(_masks_to_shapes_worker(_make_payload(key)))
    else:
        # Use a fork context so workers inherit this module without re-importing
        # __main__ (Python >= 3.14 no longer defaults to fork on Linux).
        try:
            ctx = mp.get_context("fork")
        except ValueError:  # platform without fork (e.g. Windows)
            ctx = mp.get_context()

        # Materialise FOV stacks lazily and cap the number held in memory to the
        # pool size (only the FOVs currently being worked on): the generator
        # blocks on the semaphore until a result frees a slot, so masks are not
        # read in ahead of the workers.
        slots = threading.Semaphore(pool_size)

        def _lazy_payloads():
            for key in labels_keys:
                slots.acquire()
                yield _make_payload(key)

        with ctx.Pool(pool_size) as pool:
            for res in pool.imap_unordered(_masks_to_shapes_worker, _lazy_payloads()):
                shapes.update(res)
                slots.release()

    if inplace:
        for k, v in shapes.items():
            sdata.shapes[k] = v
            if sdata.is_backed():
                sdata.write_element(k)
        return sdata
    return SpatialData(shapes=shapes)


# --------------------------------------------------------------------------- #
# Extracting a single z-slice as a flat 2D SpatialData                         #
# --------------------------------------------------------------------------- #


# Coordinate systems that hold a single element type. They are useful on the
# full object (mirroring spatialdata_io.cosmx), but they break plotting: a
# ``render_images()`` call in a labels-only system has no image to take an extent
# from. ``select_zslice`` therefore leaves them behind — see its docstring.
TYPE_RESTRICTED_CS = ("global_only_image", "global_only_labels")


def _reduce_transformations_2d(element, drop=(STACK_3D_CS, *TYPE_RESTRICTED_CS)):
    """Return the element's transformations reduced to 2D ``(x, y)`` mappings,
    dropping any coordinate systems in ``drop`` (by default the 3D stack system
    and the type-restricted ``global_only_*`` systems).

    Each transform is projected onto its x/y part, so a 2D-sliced element no
    longer carries a 3D (``x, y, z``) transform — which otherwise breaks
    ``spatialdata``'s rasterizer (and thus ``spatialdata-plot``).
    """
    out = {}
    for cs, t in get_transformation(element, get_all=True).items():
        if cs in drop:
            continue
        matrix = t.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        out[cs] = Affine(matrix, input_axes=("x", "y"), output_axes=("x", "y"))
    return out


def select_zslice(
    sdata: SpatialData,
    z: int,
    *,
    points_z_column: str = "z",
    snap_z: Optional[bool] = None,
) -> SpatialData:
    """Extract a single z-slice of a (pseudo-3D) SpatialData as a flat 2D object.

    Builds a **new** :class:`spatialdata.SpatialData` containing only what is
    relevant to slice ``z``, with every element reduced to 2D so it plots and
    rasterizes without error:

    * **images / labels** — 3D ``(…, z, y, x)`` elements are sliced with
      ``isel(z=z)`` to 2D; already-2D elements are copied as-is. (A 3D element
      that has no slice ``z`` is skipped.)
    * **points** — kept where the slice they belong to is ``z``: their
      ``points_z_column`` rounds to ``z`` (within half a slice, ties going to the
      lower slice), not merely equals it; emitted as 2D points (the z coordinate
      column is dropped). Which slice a point belongs to follows the **same
      z-correction policy as** :func:`aggregate_zslice_shapes` — see ``snap_z``
      — so a slice selected here holds exactly the transcripts aggregation would
      count into it.
    * **shapes** — ``{fov}_z{z}_shapes`` elements for this ``z`` are kept; those
      for other slices are dropped. Shapes with no ``_z{z}_`` marker are kept
      as z-agnostic.
    * **coordinate systems** — every element's transforms are projected to 2D,
      and both the 3D stack system (:data:`STACK_3D_CS`) and the type-restricted
      :data:`TYPE_RESTRICTED_CS` systems are dropped, leaving only ``fov_*`` and
      ``"global"``. Every remaining system therefore holds *all* element types,
      which is what makes the result safely plottable: ``spatialdata-plot``
      auto-selects coordinate systems from an unordered set, and if it lands on a
      labels-only system a ``render_images()`` call fails with "does not contain
      any element in the coordinate system 'global_only_labels'". Use the full
      object if you need the type-restricted systems.

    Tables are not carried over (their region/instance links are slice-agnostic).

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        The (pseudo-3D) object, e.g. from :func:`softseg_to_spatialdata` +
        :func:`masks_to_shapes`.
    z : int
        The z-slice index to extract.
    points_z_column : str
        Column of the points holding the z location (``"z"`` after
        :func:`softseg_to_spatialdata`).
    snap_z : bool, optional
        How to treat points whose z rounds to no valid slice of their FOV (the
        slices recorded in ``sdata.attrs``). ``True`` folds them into the nearest
        valid slice, so they appear here when that slice is selected; ``False``
        leaves them out of every slice. If ``None`` (default) the policy recorded
        by :func:`softseg_to_spatialdata` is used, falling back to ``False`` —
        keeping this function consistent with what
        :func:`aggregate_zslice_shapes` would count. When no ``valid_z`` is
        recorded for a FOV, every integer is treated as a valid slice and this
        setting has no effect.

    Returns
    -------
    spatialdata.SpatialData
        A new, fully 2D object for slice ``z``.
    """
    snap_z = _resolve_snap_z(sdata, snap_z)
    images: dict = {}
    for name in sdata.images:
        el = sdata.images[name]
        dims = getattr(el, "dims", ())
        if "z" in dims:
            if z >= int(el.sizes["z"]):
                continue  # this slice does not exist for this element
            sl = el.isel(z=z)
        else:
            sl = el
        sl = sl.transpose("c", "y", "x")
        images[name] = Image2DModel.parse(
            sl.data, dims=("c", "y", "x"), transformations=_reduce_transformations_2d(sl)
        )

    labels: dict = {}
    for name in sdata.labels:
        el = sdata.labels[name]
        dims = getattr(el, "dims", ())
        if "z" in dims:
            if z >= int(el.sizes["z"]):
                continue
            sl = el.isel(z=z)
        else:
            sl = el
        sl = sl.transpose("y", "x")
        labels[name] = Labels2DModel.parse(
            sl.data, dims=("y", "x"), transformations=_reduce_transformations_2d(sl)
        )

    points: dict = {}
    for name in sdata.points:
        el = sdata.points[name]
        feature_key = el.attrs.get("spatialdata_attrs", {}).get("feature_key")
        pdf = el.compute() if hasattr(el, "compute") else pd.DataFrame(el)

        if points_z_column in pdf.columns:
            # Keep the points belonging to slice `z`, resolving "which slice"
            # exactly as aggregate_zslice_shapes does: against this FOV's valid
            # slices, honouring the recorded snap-vs-drop policy. Points that
            # belong to no slice get NaN and so match no `z`.
            zvals = pdf[points_z_column].to_numpy(dtype=float)
            fov = name[: -len("_points")] if name.endswith("_points") else name
            recorded = _recorded_valid_z(sdata, fov)
            if recorded:
                allowed = np.asarray(sorted(int(v) for v in recorded), dtype=float)
                slice_of_point, _ = _assign_points_to_slices(zvals, allowed, snap_z)
            else:  # nothing recorded: every integer slice is valid
                slice_of_point = np.floor(zvals + 0.5)
            pdf = pdf[slice_of_point == z]
        if len(pdf) == 0:
            continue

        # drop both the z coordinate and the raw global_z it came from, so the
        # 2D result carries no stale z information
        pdf = pdf.drop(
            columns=[points_z_column, "global_z"], errors="ignore"
        ).reset_index(drop=True)
        pdf.attrs = {}
        kwargs = {"feature_key": feature_key} if feature_key else {}
        points[name] = PointsModel.parse(
            pdf,
            coordinates={"x": "x", "y": "y"},
            transformations=_reduce_transformations_2d(el),
            **kwargs,
        )

    shapes: dict = {}
    for name in sdata.shapes:
        m = re.search(r"_z(\d+)_shapes$", name)
        if m and int(m.group(1)) != z:
            continue  # a different slice's shapes
        gdf = sdata.shapes[name].copy()
        tfs = _reduce_transformations_2d(gdf)
        gdf.attrs = {}
        shapes[name] = ShapesModel.parse(gdf, transformations=tfs)

    # carry the z-correction policy over, so the slice can be re-sliced or
    # aggregated without having to be told about it again
    return SpatialData(
        images=images,
        labels=labels,
        points=points,
        shapes=shapes,
        attrs=dict(getattr(sdata, "attrs", None) or {}),
    )


# --------------------------------------------------------------------------- #
# Aggregating points into the per-z-slice (pseudo-3D) shapes                   #
# --------------------------------------------------------------------------- #


def _discover_zslice_shapes(sdata: SpatialData, by) -> "dict[int, str]":
    """Resolve ``by`` to an ordered ``{z_index: shape_element_name}`` mapping.

    ``by`` may be an explicit list/tuple of ``"{fov}_z{z}_shapes"`` names, or a
    FOV prefix (e.g. ``"2"`` or ``2``) that is expanded to every matching
    ``f"{fov}_z{z}_shapes"`` element in ``sdata.shapes``.
    """
    if isinstance(by, (list, tuple)):
        names = list(by)
        out: dict[int, str] = {}
        for n in names:
            m = re.search(r"_z(\d+)_shapes$", n)
            out[int(m.group(1)) if m else len(out)] = n
    else:
        pat = re.compile(rf"^{re.escape(str(by))}_z(\d+)_shapes$")
        out = {int(m.group(1)): n for n in sdata.shapes if (m := pat.match(n))}
    if not out:
        raise KeyError(f"No z-slice shapes found for by={by!r} in {list(sdata.shapes)}")
    return dict(sorted(out.items()))


def _fov_cs_of(shape_element) -> str:
    """Return a shape element's FOV coordinate system (the one that is neither
    ``"global"`` nor the 3D stack system)."""
    names = list(get_transformation(shape_element, get_all=True))
    for n in names:
        if n not in ("global", STACK_3D_CS):
            return n
    return names[0]


def aggregate_zslice_shapes(
    sdata: SpatialData,
    values: str,
    by: Union[str, int, Sequence[str]],
    *,
    value_key: Optional[str] = None,
    agg_func: Union[str, list] = "count",
    target_coordinate_system: Optional[str] = None,
    fractions: bool = False,
    region_key: str = "region",
    instance_key: str = "instance_id",
    deepcopy: bool = True,
    table_name: Optional[str] = None,
    buffer_resolution: int = 16,
    z_column: str = "z",
    feature_key: Optional[str] = None,
    combine: str = "sum",
    snap_z: Optional[bool] = None,
    **kwargs: Any,
) -> ad.AnnData:
    """Aggregate points into the per-z-slice shapes of a (pseudo-3D) FOV.

    The project stores a 3D segmentation as one 2D shapes element per z-slice
    (``f"{fov}_z{z}_shapes"``, since geopandas has no 3D polygons). This runs
    :func:`spatialdata.aggregate` slice-by-slice and merges the results:

    1. Resolve ``by`` to the FOV's z-slice shapes and read each slice's z index.
    2. Assign every point of ``values`` to the slice its ``z_column`` rounds to.
       Points that do not round to a valid slice are dropped here (with a
       warning) unless ``snap_z`` applies, in which case they are snapped to the
       nearest valid slice. This is the step that discards transcripts sitting
       off the segmentation; :func:`softseg_to_spatialdata` only warns about
       them and records the policy in ``sdata.attrs``.
    3. For each slice, aggregate only the points assigned to it into that slice's
       shape (in the FOV coordinate system, in 2D).
    4. Concatenate the per-slice tables and combine rows for the same
       ``cell_id`` with ``combine`` (default ``"sum"``), yielding one table whose
       rows are the FOV's cells and whose values are pooled over all slices.

    All aggregation arguments mirror :func:`spatialdata.aggregate` and are passed
    through (``value_key``, ``agg_func``, ``fractions``, ``region_key``,
    ``instance_key``, ``deepcopy``, ``table_name``, ``buffer_resolution``, plus
    any extra ``kwargs``).

    .. note::
       The result covers **one FOV**. Because SoftSeg masks carry globally unique
       cell labels, a cell lying on a FOV boundary appears in both neighbouring
       masks, and this function only ever sees the part of it inside ``by``'s FOV
       — so its counts are that FOV's share, not the whole cell's. To recover
       whole-cell counts (what
       :meth:`~SoftSeg.SoftAssigner.SoftAssigner.convert_to_adata` produces, since
       it tallies across every FOV), aggregate each FOV separately and pool the
       resulting tables by ``cell_id``.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        Object holding the points and the per-slice shapes.
    values : str
        Name of the points element to aggregate (e.g. ``"2_points"``).
    by : str | int | sequence of str
        The FOV's 3D shape: a FOV prefix (``"2"``) or an explicit list of
        ``"{fov}_z{z}_shapes"`` names.
    target_coordinate_system : str, optional
        Coordinate system to aggregate in. Defaults to the shapes' FOV system,
        where points and polygons share raw 2D pixel coordinates.
    z_column : str
        Column in the points holding the z location used to pick the nearest
        slice (``"z"`` after :func:`softseg_to_spatialdata`, from ``global_z``).
    feature_key : str, optional
        Gene/feature column of the points. If ``None``, read from the points
        element metadata.
    combine : str
        Pandas aggregation used to pool a cell's rows across slices (``"sum"``).
    snap_z : bool, optional
        What to do with points whose ``z_column`` does not round to one of this
        FOV's valid slices. ``True`` snaps them to the nearest available slice
        (assigning them to a slice they do not belong to); ``False`` drops them.
        If ``None`` (default) the policy recorded by
        :func:`softseg_to_spatialdata` in ``sdata.attrs`` is used, falling back
        to ``False``. **This is where transcripts that miss the segmentation are
        actually discarded** — conversion only warns about them.

        The valid slices are those recorded in ``attrs`` for this FOV if present,
        intersected with the slices that have a shapes element; otherwise just
        the latter. The distinction matters because a slice holding no cells has
        no shapes element, so its points have nowhere legitimate to go.

    Returns
    -------
    anndata.AnnData
        One row per ``cell_id`` present across the slices, columns are the
        aggregated features, values pooled over slices. ``obs`` carries
        ``cell_id`` and the source ``by``.
    """
    slices = _discover_zslice_shapes(sdata, by)
    slice_idx = np.array(sorted(slices), dtype=float)

    # points -> pandas
    pts_elem = sdata.points[values]
    pdf = pts_elem.compute() if hasattr(pts_elem, "compute") else pd.DataFrame(pts_elem)

    if feature_key is None:
        feature_key = pts_elem.attrs.get("spatialdata_attrs", {}).get("feature_key")

    if target_coordinate_system is None:
        target_coordinate_system = _fov_cs_of(sdata.shapes[next(iter(slices.values()))])

    snap_z = _resolve_snap_z(sdata, snap_z)

    # A point may only land on a slice that both the segmentation covers (per the
    # recorded valid_z, when available) and that has a shapes element to
    # aggregate into.
    recorded = _recorded_valid_z(sdata, by)
    if recorded is not None:
        allowed = np.array(sorted(set(slice_idx.astype(int)) & set(int(v) for v in recorded)),
                           dtype=float)
        if len(allowed) == 0:
            raise ValueError(
                f"None of the z-slices with shapes for by={by!r} "
                f"({sorted(slice_idx.astype(int))}) are among the valid slices "
                f"recorded in sdata.attrs[{SOFTSEG_ATTRS!r}] ({sorted(recorded)})."
            )
    else:
        allowed = slice_idx

    # assign each point to the slice its z rounds to
    if z_column in pdf.columns:
        zvals = pdf[z_column].to_numpy(dtype=float)
        nearest, off = _assign_points_to_slices(zvals, allowed, snap_z)
        if off.any():
            warnings.warn(
                f"{int(off.sum())} of {len(pdf)} points of {values!r} have a "
                f"{z_column} that does not round to any valid z-slice of "
                f"by={by!r} (slices {allowed.min():g}-{allowed.max():g}). "
                + (
                    "Snapping them to the nearest slice."
                    if snap_z
                    else "Dropping them; pass snap_z=True to snap them to the "
                    "nearest slice instead."
                ),
                stacklevel=2,
            )
            if not snap_z:
                pdf = pdf[~off]
                nearest = nearest[~off]
    else:  # no z info: everything goes to the single/first slice
        nearest = np.full(len(pdf), allowed[0])

    forwarded = dict(
        value_key=value_key,
        agg_func=agg_func,
        fractions=fractions,
        region_key=region_key,
        instance_key=instance_key,
        deepcopy=deepcopy,
        buffer_resolution=buffer_resolution,
        **kwargs,
    )
    if table_name is not None:
        forwarded["table_name"] = table_name
    tname = table_name or "table"

    per_slice = []  # (DataFrame indexed by cell_id, columns = features)
    coords = {"x": "x", "y": "y"}
    pt_kwargs = {"feature_key": feature_key} if feature_key else {}

    # keep only the columns aggregate needs, so geopandas' internal sjoin doesn't
    # collide on stray index columns (e.g. "index"/"level_0") in the points frame
    keep_cols = ["x", "y"]
    for c in (feature_key, value_key):
        if c and c in pdf.columns and c not in keep_cols:
            keep_cols.append(c)

    for z, shape_name in slices.items():
        sub = pdf.loc[nearest == z, keep_cols].reset_index(drop=True)
        if len(sub) == 0:
            continue
        pts2d = PointsModel.parse(
            sub,
            coordinates=coords,
            transformations={target_coordinate_system: Identity()},
            **pt_kwargs,
        )
        agg = _sd_aggregate(
            values=pts2d,
            by=sdata.shapes[shape_name],
            target_coordinate_system=target_coordinate_system,
            **forwarded,
        )
        table = agg[tname]
        X = table.X.toarray() if hasattr(table.X, "toarray") else np.asarray(table.X)
        # map positional instance_id -> cell_id from the shape gdf
        cell_ids = sdata.shapes[shape_name]["cell_id"].to_numpy()[
            table.obs[instance_key].to_numpy().astype(int)
        ]
        per_slice.append(
            pd.DataFrame(X, index=cell_ids, columns=list(table.var_names))
        )

    if not per_slice:
        return ad.AnnData(
            X=np.empty((0, 0)),
            obs=pd.DataFrame({"cell_id": []}),
        )

    # combine: pool rows for the same cell_id across slices, align feature columns
    combined = (
        pd.concat(per_slice)
        .groupby(level=0)
        .agg(combine)
        .sort_index()
    )
    combined = combined.reindex(sorted(combined.columns), axis=1).fillna(0)

    adata = ad.AnnData(combined.to_numpy())
    adata.var_names = combined.columns.astype(str)
    adata.obs_names = combined.index.astype(str)
    adata.obs["cell_id"] = combined.index.to_numpy()
    adata.obs["by"] = str(by)
    return adata
