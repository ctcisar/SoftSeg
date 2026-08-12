"""
Warp the boundaries of segmentation Shapes in a SpatialData object.

Standalone utility (independent of the rest of SoftSeg). It targets the two
polygon shapes elements produced by the st-qc simulation benchmark:

* ``cell_borders`` — GeoDataFrame of cell-boundary polygons (geometry only).
* ``cells``        — GeoDataFrame of cell polygons plus per-cell metadata
                     (celltype, areas, scales, ...).

Both describe the same set of cells, so by default they are deformed with a
single *shared* warp field — the same point in space moves the same way in
both elements, keeping ``cell_borders`` and ``cells`` mutually consistent.
Metadata columns on ``cells`` are carried through untouched, and each element's
coordinate transformations are preserved via the spatialdata transformation API.

Two warp styles are provided:

* ``smooth_field_warp`` — one smooth, spatially-coherent deformation field
  applied to every vertex. Neighbouring cells deform consistently. Mimics
  distortion from imperfect registration / tissue movement. (Default, and the
  right choice when warping ``cell_borders`` and ``cells`` together.)

* ``boundary_noise_warp`` — independent low-frequency noise added to each
  cell's own boundary (radially, about its centroid). Cells deform
  independently, so it does NOT keep the two elements aligned — use it on a
  single element to model per-cell segmentation uncertainty.

Both warpers are pure functions of an ``(N, 2)`` vertex array, so you can write
your own and pass it in.

Example
-------
>>> import spatialdata as sd
>>> from SoftSeg.warp_shapes import warp_shapes, smooth_field_warp
>>> sdata = sd.read_zarr("sdata_evaled.zarr")
>>> warped = warp_shapes(sdata, smooth_field_warp(amplitude=5, length_scale=80, seed=0))
>>> warped.shapes["cell_borders_warped"], warped.shapes["cells_warped"]
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np
import shapely

try:  # spatialdata only needed by warp_shapes(), not the warpers themselves
    import spatialdata as sd
    from spatialdata.models import ShapesModel
    from spatialdata.transformations import get_transformation, set_transformation
except Exception:  # pragma: no cover
    sd = None
    ShapesModel = None
    get_transformation = None
    set_transformation = None


# A warper maps an (N, 2) array of [x, y] vertices to a new (N, 2) array.
Warper = Callable[[np.ndarray], np.ndarray]

# Shapes elements warped by default.
DEFAULT_ELEMENTS = ("cell_borders", "cells")


def smooth_field_warp(
    amplitude: float = 5.0,
    length_scale: float = 80.0,
    n_components: int = 8,
    seed: Optional[int] = None,
) -> Warper:
    """
    Build a smooth, global deformation field shared by all vertices.

    The displacement is a sum of random sinusoids (cheap band-limited noise).
    Because every vertex is displaced by the same continuous function of its
    position, adjacent cells warp coherently, the same physical location moves
    identically across elements, and shapes stay non-degenerate for reasonable
    ``amplitude``.

    Parameters
    ----------
    amplitude : float
        Approximate maximum displacement, in shape coordinate units (pixels).
    length_scale : float
        Spatial wavelength of the deformation; larger = smoother / more gradual.
    n_components : int
        Number of random sinusoidal components summed per axis.
    seed : int, optional
        Seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    base_freq = 2.0 * np.pi / float(length_scale)
    freqs = base_freq * (0.5 + rng.random((2, n_components)))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(2, n_components))
    directions = rng.normal(size=(2, n_components, 2))
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    weights = rng.normal(size=(2, n_components))
    weights *= amplitude / np.sqrt(np.sum(weights**2, axis=1, keepdims=True))

    def _warp(coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        proj = np.einsum("ij,kcj->kci", coords, directions)  # (axis, comp, N)
        waves = np.sin(proj * freqs[:, :, None] + phases[:, :, None])
        disp = np.einsum("kc,kci->ik", weights, waves)  # (N, 2)
        return coords + disp

    return _warp


def boundary_noise_warp(
    amplitude: float = 4.0,
    smoothness: int = 6,
    seed: Optional[int] = None,
) -> Warper:
    """
    Build a per-shape radial boundary perturbation.

    Each shape's vertices are pushed in/out along the radial direction from the
    shape centroid, by an offset that varies smoothly around the boundary (a
    low-frequency Fourier series in the boundary angle). Shapes deform
    independently of one another.

    Parameters
    ----------
    amplitude : float
        Approximate maximum radial displacement, in coordinate units.
    smoothness : int
        Highest angular frequency (number of Fourier modes). Lower = blobbier.
    seed : int, optional
        Seed for reproducibility; each shape additionally gets a deterministic
        offset so shapes warp differently even with one seed.
    """
    state = {"i": 0}

    def _warp(coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        local = np.random.default_rng(None if seed is None else (seed, state["i"]))
        state["i"] += 1

        centroid = coords.mean(axis=0)
        rel = coords - centroid
        angles = np.arctan2(rel[:, 1], rel[:, 0])

        modes = np.arange(1, smoothness + 1)
        a = local.normal(size=smoothness)
        b = local.normal(size=smoothness)
        offset = np.cos(np.outer(angles, modes)) @ a + np.sin(np.outer(angles, modes)) @ b
        peak = np.max(np.abs(offset))
        if peak > 0:
            offset = offset / peak * amplitude

        radial_dir = rel / (np.linalg.norm(rel, axis=1, keepdims=True) + 1e-12)
        return coords + radial_dir * offset[:, None]

    return _warp


def _warp_geometry(geom, warper: Warper, repair: bool = True):
    """Apply ``warper`` to a single shapely geometry, returning a new geometry.

    A large warp can fold a small polygon's boundary onto itself; if ``repair``
    is set, any resulting invalid geometry is fixed with a zero-width buffer.
    """
    if geom is None or geom.is_empty:
        return geom
    # shapely.transform feeds every ring's coords (N, 2) through the callable.
    warped = shapely.transform(geom, warper)
    if repair and not warped.is_valid:
        warped = warped.buffer(0)
    return warped


def _warp_element(src, warper: Warper, repair: bool = True):
    """
    Return a warped, schema-valid copy of one shapes GeoDataFrame.

    Non-geometry columns are preserved, and the element's coordinate
    transformations are re-applied through the spatialdata transformation API
    (``get_transformation`` / ``set_transformation``).
    """
    out = src.copy()
    out["geometry"] = out.geometry.apply(lambda g: _warp_geometry(g, warper, repair))

    if ShapesModel is None:
        return out

    # Capture the existing transformations, then clear element metadata so the
    # re-parse doesn't see the transform both in ``attrs`` and as an argument.
    transformations = None
    if get_transformation is not None:
        try:
            transformations = get_transformation(src, get_all=True)
        except Exception:
            transformations = None

    out.attrs = {}
    out = ShapesModel.parse(out)
    if transformations is not None and set_transformation is not None:
        set_transformation(out, transformations, set_all=True)
    return out


def rasterize_shapes(
    sdata,
    element_name: str,
    *,
    pixel_size: float = 1.0,
    bounds=None,
    id_column: Optional[str] = None,
    background: int = 0,
    out_path: Optional[str] = None,
    dtype=None,
):
    """
    Rasterize a shapes element of a SpatialData object into a labelled image.

    Each output pixel holds the id of the cell whose polygon covers that
    location, with ``background`` (default 0) elsewhere. This matches the
    package's mask convention where background is 0 and mask value ``m``
    corresponds to cell id ``m - 1`` — i.e. with the default ``id_column=None``
    the label written for the i-th row of the element is ``i + 1``.

    The element's coordinate transformation is applied first. The element's
    attached coordinate system is read automatically from ``sdata`` and the
    geometries are mapped into it via :func:`spatialdata.transform`, so any
    ``Scale`` / ``Translation`` / ``Affine`` on the element is honoured and
    ``pixel_size`` / ``bounds`` are interpreted in that coordinate system's
    units (not the element's raw intrinsic units). For this dataset every
    transform is ``Identity``, so the two coincide.

    The entire area enclosed by each polygon's outer boundary is filled with
    the cell id — interior rings/holes are filled too, not left as background.
    Where warped polygons overlap, later rows in the element overwrite earlier
    ones.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        The object holding the shapes element.
    element_name : str
        Key into ``sdata.shapes`` of the element to rasterize,
        e.g. ``"cell_borders_warped"``.
    pixel_size : float
        Size of one pixel, in units of the element's coordinate system.
    bounds : (xmin, ymin, xmax, ymax), optional
        Extent to rasterize, in the element's coordinate system. Defaults to the
        transformed shapes' ``total_bounds``.
    id_column : str, optional
        Column to take cell ids from. If None, uses positional index + 1.
    background : int
        Fill value for pixels not covered by any polygon.
    out_path : str, optional
        If given, write the label image to this path as a TIFF.
    dtype : numpy dtype, optional
        Output dtype. Defaults to the smallest unsigned int that fits the
        largest id.

    Returns
    -------
    (label_img, transform) : (numpy.ndarray, dict)
        The 2D label image (shape ``(height, width)``, row=y, col=x) and a dict
        ``{"xmin","ymin","pixel_size","coordinate_system"}`` mapping coordinate-
        system coords to pixels via ``col = (x - xmin) / pixel_size``,
        ``row = (y - ymin) / pixel_size``.
    """
    from skimage.draw import polygon as draw_polygon

    if element_name not in sdata.shapes:
        raise KeyError(
            f"{element_name!r} is not a shapes element; available: {list(sdata.shapes)}"
        )
    gdf = sdata.shapes[element_name]

    # Read the element's own coordinate system(s) from the object and bake that
    # transformation into the geometry coordinates before rasterizing.
    coordinate_system = None
    if get_transformation is not None:
        attached = list(get_transformation(gdf, get_all=True))
        if len(attached) > 1:
            raise ValueError(
                f"{element_name!r} maps to multiple coordinate systems {attached}; "
                "rasterization is ambiguous."
            )
        coordinate_system = attached[0] if attached else None
        if coordinate_system is not None and sd is not None:
            gdf = sd.transform(gdf, to_coordinate_system=coordinate_system)

    if bounds is None:
        xmin, ymin, xmax, ymax = gdf.total_bounds
    else:
        xmin, ymin, xmax, ymax = bounds

    width = int(np.ceil((xmax - xmin) / pixel_size))
    height = int(np.ceil((ymax - ymin) / pixel_size))

    if id_column is None:
        ids = np.arange(1, len(gdf) + 1)
    else:
        ids = np.asarray(gdf[id_column])

    if dtype is None:
        max_id = int(max(ids.max(), background)) if len(ids) else background
        dtype = (
            np.uint16 if max_id <= np.iinfo(np.uint16).max
            else np.uint32 if max_id <= np.iinfo(np.uint32).max
            else np.uint64
        )

    label = np.full((height, width), background, dtype=dtype)

    def _to_px(coords):
        coords = np.asarray(coords)
        cc = (coords[:, 0] - xmin) / pixel_size
        rr = (coords[:, 1] - ymin) / pixel_size
        return rr, cc

    def _fill(geom, value):
        if geom is None or geom.is_empty:
            return
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            # Fill the entire area enclosed by the outer boundary, so the whole
            # cell (including any interior ring/hole) takes the cell id.
            rr, cc = _to_px(poly.exterior.coords)
            fr, fc = draw_polygon(rr, cc, shape=label.shape)
            label[fr, fc] = value

    for geom, value in zip(gdf.geometry, ids):
        _fill(geom, int(value))

    if out_path is not None:
        import tifffile

        tifffile.imwrite(out_path, label)

    return label, {
        "xmin": float(xmin),
        "ymin": float(ymin),
        "pixel_size": float(pixel_size),
        "coordinate_system": coordinate_system,
    }


def plot_warp_overlay(
    original,
    warped,
    *,
    ax=None,
    bbox=None,
    orig_color="0.5",
    warp_color="crimson",
    linewidth=0.8,
    title=None,
):
    """
    Overlay warped shape boundaries on the originals.

    Draws each original polygon outline in one colour and the corresponding
    warped outline in another, on a shared, equal-aspect axis. Pass the
    matching elements yourself, e.g. ``sdata.shapes["cell_borders"]`` and
    ``warped.shapes["cell_borders_warped"]``.

    Parameters
    ----------
    original, warped : geopandas.GeoDataFrame
        The before/after shapes elements (any geometry that has an exterior).
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; a new figure/axis is created if omitted.
    bbox : (xmin, ymin, xmax, ymax), optional
        Zoom to this window (in shape coordinates) instead of the full extent.
    orig_color, warp_color : color
        Outline colours for original and warped boundaries.
    linewidth : float
        Outline width.
    title : str, optional
        Axis title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    original.boundary.plot(ax=ax, color=orig_color, linewidth=linewidth, zorder=1)
    warped.boundary.plot(ax=ax, color=warp_color, linewidth=linewidth, zorder=2)

    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    ax.set_aspect("equal")
    ax.invert_yaxis()  # image convention: y increases downward
    ax.legend(
        handles=[
            Line2D([0], [0], color=orig_color, lw=2, label="original"),
            Line2D([0], [0], color=warp_color, lw=2, label="warped"),
        ],
        loc="upper right",
    )
    if title:
        ax.set_title(title)
    return ax


def warp_shapes(
    sdata,
    warper: Warper = None,
    *,
    elements: Iterable[str] = DEFAULT_ELEMENTS,
    suffix: str = "_warped",
    repair: bool = True,
    inplace: bool = False,
):
    """
    Warp the boundaries of segmentation shapes in a SpatialData object.

    Operates on the ``cell_borders`` and ``cells`` GeoDataFrame elements (by
    default), applying the *same* ``warper`` to both so they stay consistent.
    Geometry is warped vertex-wise; all non-geometry columns and each element's
    coordinate transformations are preserved.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        The loaded object.
    warper : Warper, optional
        Function mapping an ``(N, 2)`` vertex array to a new one. Defaults to
        :func:`smooth_field_warp` with default parameters. The same callable is
        reused for every element, so a stateless field warper deforms both
        elements identically in space.
    elements : iterable of str
        Shapes keys to warp. Defaults to ``("cell_borders", "cells")``.
    suffix : str
        Suffix for the new element names when ``inplace`` is False.
    repair : bool
        If True (default), repair any polygon that a large warp renders invalid
        (e.g. a self-intersecting boundary) with a zero-width buffer.
    inplace : bool
        If True, replace each element in ``sdata`` and return ``sdata``.
        Otherwise return a new SpatialData with the warped elements added under
        ``f"{name}{suffix}"``.

    Returns
    -------
    spatialdata.SpatialData
        ``sdata`` if ``inplace``, else a new object holding the warped shapes.
    """
    if warper is None:
        warper = smooth_field_warp()

    missing = [e for e in elements if e not in sdata.shapes]
    if missing:
        raise KeyError(
            f"shapes element(s) {missing} not found; available: {list(sdata.shapes)}"
        )

    warped = {name: _warp_element(sdata.shapes[name], warper, repair) for name in elements}

    if inplace:
        for name, gdf in warped.items():
            sdata.shapes[name] = gdf
        return sdata

    return sd.SpatialData(shapes={f"{name}{suffix}": gdf for name, gdf in warped.items()})
