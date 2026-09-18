"""Helpers for working with a SoftSeg dataset held as a :class:`spatialdata.SpatialData`.

Everything lives on the single :class:`SpatialDataHelpers` class, in four
labelled groups:

1. **Coordinate-system conventions** -- the systems that exist alongside
   ``"global"``, and how to recover a FOV's own system from an element.
2. **The z-slice layer used by** :class:`~SoftSeg.SoftAssigner.SoftAssigner` --
   which slice a transcript sits on, under the ``valid_z``/``snap_z`` policy
   recorded on the object.
3. **Segmentation masks -> per-z-slice shapes** -- a 3D segmentation cannot be
   held as shapes directly (geopandas has no 3D polygons), so it is split into
   one shapes element per slice and reunited through these.
4. **Standalone 3D utilities** the assigner does not itself use: pulling a single
   z-slice out as a flat 2D object, and aggregating points into the per-slice
   shapes of a whole FOV.

Building a SpatialData out of loose per-FOV CSV/TIFF files is *not* here: see
:class:`SoftSeg.SupportFuncs.DatasetFormatter`. Data from a supported platform
should use the standard readers (``spatialdata_io.cosmx``, ``xenium``, ...).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import re
import threading
import warnings
from typing import Any, Mapping, Optional, Sequence, Union

import anndata as ad
import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from spatialdata import SpatialData
from spatialdata import aggregate as _sd_aggregate
from spatialdata.models import PointsModel, ShapesModel
from spatialdata.transformations import Affine, Identity, get_transformation

# --------------------------------------------------------------------------- #
# The SpatialData layer                                                        #
# --------------------------------------------------------------------------- #


class SpatialDataHelpers:
    """Everything needed to work a SoftSeg dataset held as a SpatialData.

    Grouped as: the coordinate-system conventions, the z-slice layer
    :class:`~SoftSeg.SoftAssigner.SoftAssigner` runs on, and the standalone 3D
    utilities it does not itself use.
    """

    # ----------------------------------------------------------------------- #
    # 1. Coordinate-system conventions                                        #
    # ----------------------------------------------------------------------- #

    # The shared 3D coordinate system into which every per-slice, per-FOV shapes
    # element is placed at its correct (x, y, z) location. This is the system
    # that "accounts for" splitting a 3D stack into separate 2D shapes elements
    # (geopandas has no 3D polygons).
    STACK_3D_CS = "global_3d"

    # Key under which this package's own bookkeeping lives in SpatialData.attrs.
    SOFTSEG_ATTRS = "softseg"

    @staticmethod
    def _element_fov_cs(element) -> Optional[str]:
        """The element's own FOV coordinate system, read off the element itself.

        An element sits in its FOV's system plus the shared ones, so the FOV
        system is whatever is left after discarding ``"global"``, the
        type-restricted ``"global_only_*"`` systems and the 3D stack system.
        Taking the name from the element, rather than rebuilding a
        ``f"fov_{fov}"`` string, means it is whatever the object actually
        carries -- however that object was built.
        """
        names = list(get_transformation(element, get_all=True))
        for name in names:
            if (
                name == "global"
                or name.startswith("global_only")
                or name == SpatialDataHelpers.STACK_3D_CS
            ):
                continue
            return name
        return names[0] if names else None

    @staticmethod
    def save_element(sdata: SpatialData, name: str) -> bool:
        """Save an element back to the store the sdata came from.

        Does nothing (returning False) for an sdata with no store on disk -- an
        in-memory object has no location to be saved to, and its elements live
        only as long as it does.

        Rewriting an element in place is a delete followed by a write: zarr
        refuses to overwrite a path it is currently backing an element from. The
        replacement must therefore already be fully in memory before this is
        called, or the delete is refused to avoid pulling the store out from
        under it. The pair is not atomic -- an interruption between the two
        leaves the element only in memory, and re-running the step rewrites it.
        """
        if not sdata.is_backed():
            return False
        try:
            sdata.delete_element_from_disk(name)
        except (ValueError, KeyError):
            pass  # not on disk yet: this is the element's first write
        sdata.write_element(name)
        return True

    @staticmethod
    def remove_element(sdata: SpatialData, name: str) -> bool:
        """Drop an element from the object and from the store it came from.

        The in-memory element has to go first: zarr refuses to delete a path that
        is still backing a lazily-read element. Returns False if the object never
        held the element.
        """
        for group in (sdata.points, sdata.labels, sdata.images, sdata.shapes,
                      sdata.tables):
            if name in group:
                del group[name]
                break
        else:
            return False

        if sdata.is_backed():
            try:
                sdata.delete_element_from_disk(name)
            except (ValueError, KeyError):
                pass  # it was never written
        return True

    # ----------------------------------------------------------------------- #
    # 2. The z-slice layer used by SoftAssigner                               #
    # ----------------------------------------------------------------------- #

    @staticmethod
    def _softseg_attrs(sdata: SpatialData) -> dict:
        """This module's bookkeeping from ``sdata.attrs``, or ``{}`` if absent.

        A SpatialData assembled by hand (or round-tripped through something that
        drops ``attrs``) simply has no recorded policy, and callers fall back to
        their own defaults.
        """
        attrs = getattr(sdata, "attrs", None) or {}
        recorded = attrs.get(SpatialDataHelpers.SOFTSEG_ATTRS) or {}
        return recorded if isinstance(recorded, Mapping) else {}

    @staticmethod
    def _resolve_snap_z(sdata: SpatialData, snap_z: Optional[bool]) -> bool:
        """The z-correction policy in force: explicit argument, else what was
        recorded at conversion, else drop."""
        if snap_z is not None:
            return bool(snap_z)
        return bool(SpatialDataHelpers._softseg_attrs(sdata).get("snap_z", False))

    @staticmethod
    def _recorded_valid_z(sdata: SpatialData, fov) -> Optional[list]:
        """The valid z-slices recorded for a FOV, or ``None`` if none were recorded.

        ``fov`` is either a FOV token or a list of ``"{fov}_z{z}_shapes"`` names; in
        the latter case the FOV is taken from the first name.
        """
        valid = SpatialDataHelpers._softseg_attrs(sdata).get("valid_z") or {}
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

    @staticmethod
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

    # ----------------------------------------------------------------------- #
    # 3. Segmentation masks -> per-z-slice Shapes                             #
    # ----------------------------------------------------------------------- #

    @staticmethod
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
            parts.extend(SpatialDataHelpers._polygon_parts(sub))
        return parts

    @staticmethod
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
            contours = cv2.findContours(
                binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[-2]

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
                polys.extend(SpatialDataHelpers._polygon_parts(poly))
            if not polys:
                continue

            geom = polys[0] if len(polys) == 1 else MultiPolygon(polys)
            geoms[int(m) - 1] = geom  # mask value m -> cell id m-1
        return geoms

    @staticmethod
    def _labels_global_affine(labels_element) -> tuple[str, np.ndarray]:
        """Recover a labels element's FOV coordinate-system name and its pixel->global
        mapping as a homogeneous ``(x, y, z) -> (x, y, z)`` 4x4 affine matrix.

        The FOV system is read off the element (:meth:`_element_fov_cs`), so the
        element's own transforms are the only thing consulted. The matrix is the
        transform attached to ``"global"``: it carries both the FOV offset and,
        when a pixel size was encoded, the physical scale. Its ``z`` scale
        (``M[2, 2]``) is the real-world spacing between z-slices. A 2D element's
        matrix is promoted to 4x4 with a unit, zero-offset z axis. Falls back to
        identity if transforms can't be read.
        """
        fov_cs = SpatialDataHelpers._element_fov_cs(labels_element)
        matrix = np.eye(4)
        try:
            transforms = get_transformation(labels_element, get_all=True)
            t = transforms["global"]
            if "z" in getattr(labels_element, "dims", ()):
                matrix = np.asarray(
                    t.to_affine_matrix(
                        input_axes=("x", "y", "z"), output_axes=("x", "y", "z")
                    )
                )
            else:  # 2D: embed into 4x4 with an identity z axis
                m2 = np.asarray(
                    t.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
                )
                matrix = np.eye(4)
                matrix[np.ix_([0, 1], [0, 1])] = m2[:2, :2]
                matrix[[0, 1], 3] = m2[:2, 2]
        except Exception:
            pass
        return fov_cs, matrix

    @staticmethod
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

    @staticmethod
    def _global_2d_from_matrix(M: np.ndarray) -> Affine:
        """The 2D ``(x, y) -> (x, y)`` projection of the labels' pixel->global affine,
        used as the shapes' ``"global"`` transform (physical x/y when pixel_size set).
        """
        matrix = np.array(
            [
                [M[0, 0], M[0, 1], M[0, 3]],
                [M[1, 0], M[1, 1], M[1, 3]],
                [0.0, 0.0, 1.0],
            ]
        )
        return Affine(matrix, input_axes=("x", "y"), output_axes=("x", "y"))

    @staticmethod
    def _masks_to_shapes_worker(payload):
        """Build the per-z-slice shapes for a single FOV. Module-level so it can be
        dispatched to a multiprocessing ``Pool``.

        ``payload`` = ``(fov, fov_cs, M, z_size, stack, min_points, polygon_offset)``
        where ``stack`` is the FOV mask as a ``(z, y, x)`` numpy array. Returns
        ``{shape_element_name: parsed_shapes_gdf}``.
        """
        fov, fov_cs, M, z_size, stack, min_points, polygon_offset = payload
        global_2d = SpatialDataHelpers._global_2d_from_matrix(M)
        out: dict = {}
        for z in range(stack.shape[0]):
            geoms = SpatialDataHelpers._slice_to_polygons(
                stack[z], min_points=min_points, offset=polygon_offset
            )
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
                    SpatialDataHelpers.STACK_3D_CS: SpatialDataHelpers._stack_affine_from_matrix(
                        M, z, z_size
                    ),
                },
            )
        return out

    @staticmethod
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
        pool_size = (
            max(1, min(pool_size, n)) if n else 1
        )  # never more workers than FOVs

        def _make_payload(key):
            """Materialise a single FOV's worker payload (transforms + mask stack)."""
            labels_element = sdata.labels[key]
            # element key is "{fov}_labels" -> recover the fov token
            fov = key[: -len("_labels")] if key.endswith("_labels") else key

            # Pixel->global affine (offset + physical scale) from the labels element.
            fov_cs, M = SpatialDataHelpers._labels_global_affine(labels_element)
            # z-spacing: encoded pixel size (M[2, 2]) unless explicitly overridden.
            z_size = (
                float(M[2, 2])
                if dist_between_slices is None
                else float(dist_between_slices)
            )

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
                shapes.update(
                    SpatialDataHelpers._masks_to_shapes_worker(_make_payload(key))
                )
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
                for res in pool.imap_unordered(
                    SpatialDataHelpers._masks_to_shapes_worker, _lazy_payloads()
                ):
                    shapes.update(res)
                    slots.release()

        if inplace:
            for k, v in shapes.items():
                sdata.shapes[k] = v
                if sdata.is_backed():
                    sdata.write_element(k)
            return sdata
        return SpatialData(shapes=shapes)

    # ----------------------------------------------------------------------- #
    # 4. Standalone 3D utilities (not used by SoftAssigner)                   #
    # ----------------------------------------------------------------------- #

    # --- Extracting a single z-slice as a flat 2D SpatialData ---

    # Coordinate systems that hold a single element type. They are useful on the
    # full object (mirroring spatialdata_io.cosmx), but they break plotting: a
    # ``render_images()`` call in a labels-only system has no image to take an extent
    # from. ``select_zslice`` therefore leaves them behind -- see its docstring.
    TYPE_RESTRICTED_CS = ("global_only_image", "global_only_labels")

    @staticmethod
    def _reduce_transformations_2d(element, drop=(STACK_3D_CS, *TYPE_RESTRICTED_CS)):
        """Return the element's transformations reduced to 2D ``(x, y)`` mappings,
        dropping any coordinate systems in ``drop`` (by default the 3D stack system
        and the type-restricted ``global_only_*`` systems).

        Each transform is projected onto its x/y part, so a 2D-sliced element no
        longer carries a 3D (``x, y, z``) transform -- which otherwise breaks
        ``spatialdata``'s rasterizer (and thus ``spatialdata-plot``).
        """
        out = {}
        for cs, t in get_transformation(element, get_all=True).items():
            if cs in drop:
                continue
            matrix = t.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
            out[cs] = Affine(matrix, input_axes=("x", "y"), output_axes=("x", "y"))
        return out

    @staticmethod
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

        * **images / labels** -- 3D ``(..., z, y, x)`` elements are sliced with
          ``isel(z=z)`` to 2D; already-2D elements are copied as-is. (A 3D element
          that has no slice ``z`` is skipped.)
        * **points** -- kept where the slice they belong to is ``z``: their
          ``points_z_column`` rounds to ``z`` (within half a slice, ties going to the
          lower slice), not merely equals it; emitted as 2D points (the z coordinate
          column is dropped). Which slice a point belongs to follows the **same
          z-correction policy as** :func:`aggregate_zslice_shapes` -- see ``snap_z``
          -- so a slice selected here holds exactly the transcripts aggregation would
          count into it.
        * **shapes** -- ``{fov}_z{z}_shapes`` elements for this ``z`` are kept; those
          for other slices are dropped. Shapes with no ``_z{z}_`` marker are kept
          as z-agnostic.
        * **coordinate systems** -- every element's transforms are projected to 2D,
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
            The (pseudo-3D) object, e.g. from
            :class:`SoftSeg.SupportFuncs.DatasetFormatter` + :func:`masks_to_shapes`.
        z : int
            The z-slice index to extract.
        points_z_column : str
            Column of the points holding the z location (``"z"`` after conversion).
        snap_z : bool, optional
            How to treat points whose z rounds to no valid slice of their FOV (the
            slices recorded in ``sdata.attrs``). ``True`` folds them into the nearest
            valid slice, so they appear here when that slice is selected; ``False``
            leaves them out of every slice. If ``None`` (default) the policy recorded
            at conversion is used, falling back to ``False`` -- keeping this function
            consistent with what :func:`aggregate_zslice_shapes` would count. When no
            ``valid_z`` is recorded for a FOV, every integer is treated as a valid
            slice and this setting has no effect.

        Returns
        -------
        spatialdata.SpatialData
            A new, fully 2D object for slice ``z``.
        """
        from spatialdata.models import Image2DModel, Labels2DModel

        snap_z = SpatialDataHelpers._resolve_snap_z(sdata, snap_z)
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
                sl.data,
                dims=("c", "y", "x"),
                transformations=SpatialDataHelpers._reduce_transformations_2d(sl),
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
                sl.data,
                dims=("y", "x"),
                transformations=SpatialDataHelpers._reduce_transformations_2d(sl),
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
                recorded = SpatialDataHelpers._recorded_valid_z(sdata, fov)
                if recorded:
                    allowed = np.asarray(sorted(int(v) for v in recorded), dtype=float)
                    slice_of_point, _ = SpatialDataHelpers._assign_points_to_slices(
                        zvals, allowed, snap_z
                    )
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
                transformations=SpatialDataHelpers._reduce_transformations_2d(el),
                **kwargs,
            )

        shapes: dict = {}
        for name in sdata.shapes:
            m = re.search(r"_z(\d+)_shapes$", name)
            if m and int(m.group(1)) != z:
                continue  # a different slice's shapes
            gdf = sdata.shapes[name].copy()
            tfs = SpatialDataHelpers._reduce_transformations_2d(gdf)
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

    # --- Aggregating points into the per-z-slice (pseudo-3D) shapes ---

    @staticmethod
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
            raise KeyError(
                f"No z-slice shapes found for by={by!r} in {list(sdata.shapes)}"
            )
        return dict(sorted(out.items()))

    @staticmethod
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
           off the segmentation; conversion only warns about them and records the
           policy in ``sdata.attrs``.
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
           -- so its counts are that FOV's share, not the whole cell's. To recover
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
            slice (``"z"`` after conversion, from ``global_z``).
        feature_key : str, optional
            Gene/feature column of the points. If ``None``, read from the points
            element metadata.
        combine : str
            Pandas aggregation used to pool a cell's rows across slices (``"sum"``).
        snap_z : bool, optional
            What to do with points whose ``z_column`` does not round to one of this
            FOV's valid slices. ``True`` snaps them to the nearest available slice
            (assigning them to a slice they do not belong to); ``False`` drops them.
            If ``None`` (default) the policy recorded at conversion in
            ``sdata.attrs`` is used, falling back to ``False``. **This is where
            transcripts that miss the segmentation are actually discarded** --
            conversion only warns about them.

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
        slices = SpatialDataHelpers._discover_zslice_shapes(sdata, by)
        slice_idx = np.array(sorted(slices), dtype=float)

        # points -> pandas
        pts_elem = sdata.points[values]
        pdf = (
            pts_elem.compute()
            if hasattr(pts_elem, "compute")
            else pd.DataFrame(pts_elem)
        )

        if feature_key is None:
            feature_key = pts_elem.attrs.get("spatialdata_attrs", {}).get("feature_key")

        if target_coordinate_system is None:
            target_coordinate_system = SpatialDataHelpers._element_fov_cs(
                sdata.shapes[next(iter(slices.values()))]
            )

        snap_z = SpatialDataHelpers._resolve_snap_z(sdata, snap_z)

        # A point may only land on a slice that both the segmentation covers (per the
        # recorded valid_z, when available) and that has a shapes element to
        # aggregate into.
        recorded = SpatialDataHelpers._recorded_valid_z(sdata, by)
        if recorded is not None:
            allowed = np.array(
                sorted(set(slice_idx.astype(int)) & set(int(v) for v in recorded)),
                dtype=float,
            )
            if len(allowed) == 0:
                raise ValueError(
                    f"None of the z-slices with shapes for by={by!r} "
                    f"({sorted(slice_idx.astype(int))}) are among the valid slices "
                    f"recorded in sdata.attrs[{SpatialDataHelpers.SOFTSEG_ATTRS!r}] ({sorted(recorded)})."
                )
        else:
            allowed = slice_idx

        # assign each point to the slice its z rounds to
        if z_column in pdf.columns:
            zvals = pdf[z_column].to_numpy(dtype=float)
            nearest, off = SpatialDataHelpers._assign_points_to_slices(
                zvals, allowed, snap_z
            )
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
            X = (
                table.X.toarray()
                if hasattr(table.X, "toarray")
                else np.asarray(table.X)
            )
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
        combined = pd.concat(per_slice).groupby(level=0).agg(combine).sort_index()
        combined = combined.reindex(sorted(combined.columns), axis=1).fillna(0)

        adata = ad.AnnData(combined.to_numpy())
        adata.var_names = combined.columns.astype(str)
        adata.obs_names = combined.index.astype(str)
        adata.obs["cell_id"] = combined.index.to_numpy()
        adata.obs["by"] = str(by)
        return adata
