import ast
import logging
import multiprocessing
import random
from collections import Counter
from contextlib import closing
from copy import deepcopy
from datetime import datetime
from itertools import repeat
from pathlib import Path

import anndata as ad
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.patches import Circle
#  from memory_profiler import profile
from spatialdata import get_centroids, read_zarr
from spatialdata.models import PointsModel
from spatialdata.transformations import get_transformation
from tqdm.auto import tqdm

from .SpatialDataHelpers import SpatialDataHelpers

# The SoftAssigner instance the current worker process is bound to. Set once per
# worker by ``_init_worker`` so a pooled run pickles the assigner (and with it
# any in-memory part of the SpatialData) once per process rather than once per
# FOV -- see ``SoftAssigner.__getstate__``.
_WORKER_ASSIGNER = None


def _init_worker(assigner):
    global _WORKER_ASSIGNER
    _WORKER_ASSIGNER = assigner


def _run_worker(args):
    """Call ``func(assigner, *rest)`` on the assigner this worker is bound to.

    ``func`` is the plain (unbound) method — e.g. ``SoftAssigner.blur_fov`` —
    which pickles as a reference to itself, so no copy of the assigner (and
    therefore none of the SpatialData) rides along with each task.
    """
    func, rest = args[0], args[1:]
    return func(_WORKER_ASSIGNER, *rest)


def sigm(x, x0=0, k=0.05):
    return 1 / (1 + np.e ** (-1 * k * (x - x0)))


def signed_distance(geom, xs, ys):
    """Signed distance from points to a polygon.

    Positive inside the geometry, negative outside, magnitude being the distance
    to the nearest edge. Vectorised over ``xs``/``ys`` (1D arrays). A
    ``MultiPolygon`` -- a cell whose slice has several disjoint pieces -- is
    measured as a whole, against the nearest edge of any of its parts.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    dist = shapely.distance(geom.boundary, shapely.points(xs, ys))
    return np.where(shapely.contains_xy(geom, xs, ys), dist, -dist)


class SoftAssigner:
    def __init__(
        self,
        sdata,
        pool_size=1,
        conf_thresh=0.7,
        decay_func=None,
    ):
        """Initialize internal parameters.

        sdata: the dataset, as a single `spatialdata.SpatialData` object (or a
           path to a zarr store holding one), in the layout produced by
           `SpatialDataHelpers.softseg_to_spatialdata`:
             - `sdata.labels[f"{fov}_labels"]` — the FOV's segmentation mask,
               2D `(y, x)` or 3D `(z, y, x)`.
             - `sdata.points[f"{fov}_points"]` — the FOV's transcript table.
             - `sdata.shapes[f"{fov}_z{z}_shapes"]` — (optional) per-z-slice cell
               polygons. Generated on demand with
               `SpatialDataHelpers.masks_to_shapes` when absent.

        This object is both the input and the output: results are written back
        into it as new columns on the points elements and as tables, and every
        element that changes is saved to the zarr store the object came from.
        An sdata with no store on disk is worked on purely in memory.

        Passing a zarr-backed `sdata` is required when `pool_size > 1`: workers
        re-open the store themselves and save their own FOV's element, which is
        the only way their results get back to the parent.
        """
        self._set_sdata(sdata)
        self.pool_size = pool_size
        self.logger = logging.getLogger()
        self.conf_thresh = conf_thresh
        if decay_func is None:
            self.decay_func = sigm
        else:
            self.decay_func = decay_func

        # The run log sits next to the store, the one location this object
        # already owns; an in-memory sdata gets no log file and leaves logging
        # configuration to the caller.
        if self._sdata_path is not None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logging.basicConfig(
                filename=f"{str(self._sdata_path).rstrip('/')}.{stamp}.run.log",
                level=logging.DEBUG,
            )

    # ----------------------------------------------------------------- #
    # SpatialData access                                                 #
    # ----------------------------------------------------------------- #

    def _set_sdata(self, sdata):
        """Bind the assigner to a SpatialData object or an on-disk zarr store."""
        if isinstance(sdata, (str, Path)):
            self._sdata_path = str(sdata)
            self._sdata = None  # opened lazily
        else:
            self._sdata = sdata
            path = getattr(sdata, "path", None)
            backed = sdata.is_backed() if hasattr(sdata, "is_backed") else False
            self._sdata_path = str(path) if (backed and path is not None) else None

    @property
    def sdata(self):
        """The bound SpatialData, opened from disk on first use if backed."""
        if self._sdata is None:
            if self._sdata_path is None:
                raise RuntimeError(
                    "This SoftAssigner has no SpatialData attached; pass one to "
                    "__init__ or call _set_sdata()."
                )
            self._sdata = read_zarr(Path(self._sdata_path))
        return self._sdata

    def _reload_sdata(self):
        """Drop the in-memory view so the next access re-reads it from disk.

        Worker processes save their own elements to the store, which the parent's
        already-open object knows nothing about -- and its lazy elements still
        point at zarr paths the workers have since rewritten. Re-reading after a
        pooled run is what makes their results visible here.
        """
        if self._sdata_path is not None:
            self._sdata = None

    def save_element(self, name):
        """Save an element back to the store the sdata came from.

        Does nothing for an sdata with no store on disk -- an in-memory object
        has no location to be saved to, and its elements live only as long as it
        does.

        Rewriting an element in place is a delete followed by a write: zarr
        refuses to overwrite a path it is currently backing an element from. The
        replacement must therefore already be fully in memory (built through
        `.compute()`, as `set_transcripts` does) before this is called, or the
        delete is refused to avoid pulling the store out from under it. The pair
        is not atomic -- an interruption between the two leaves the element only
        in memory, and re-running the step rewrites it.
        """
        saved = SpatialDataHelpers.save_element(self.sdata, name)
        if saved:
            self.logger.debug(f"[{datetime.now()}] saved element {name} to disk.")
        return saved

    def __getstate__(self):
        """Don't ship the dataset to worker processes when it lives on disk."""
        state = self.__dict__.copy()
        if state.get("_sdata_path") is not None:
            state["_sdata"] = None  # each worker re-opens the zarr store itself
        return state

    @staticmethod
    def _fov_of(key, suffix):
        return key[: -len(suffix)] if key.endswith(suffix) else key

    def _labels_key(self, fov):
        return f"{fov}_labels"

    def _points_key(self, fov):
        return f"{fov}_points"

    def get_labels(self, fov):
        """The FOV's segmentation mask as a spatialdata labels element."""
        return self.sdata.labels[self._labels_key(fov)]

    def is_3d(self, fov):
        """True if this FOV's segmentation mask has a z axis."""
        return "z" in getattr(self.get_labels(fov), "dims", ())

    def get_mask(self, fov):
        """The FOV's segmentation mask as a numpy array, `(z, y, x)` or `(y, x)`.

        Axes are selected by label rather than position, so the result is
        correctly oriented regardless of how the element is stored.
        """
        el = self.get_labels(fov)
        if "z" in el.dims:
            return np.asarray(el.transpose("z", "y", "x"))
        return np.asarray(el.transpose("y", "x"))

    def get_transcripts(self, fov):
        """The FOV's transcript table as a pandas dataframe.

        Carries the columns of the source transcript CSV plus what
        `softseg_to_spatialdata` adds (`global_x`/`global_y` and the slice index
        `z`) and what this class adds as it runs (`cell_ids`, and an assignment
        column once overlapping regions have been evaluated). The `index` column,
        holding the transcript id, is placed first.
        """
        el = self.sdata.points[self._points_key(fov)]
        df = el.compute() if hasattr(el, "compute") else pd.DataFrame(el)
        df = df.reset_index(drop=True)
        if "index" in df.columns:
            df = df[["index"] + [c for c in df.columns if c != "index"]]
        return df

    def set_transcripts(self, fov, df):
        """Write a transcript table back into the FOV's points element.

        The element is replaced by one parsed from `df` -- carrying over the
        coordinate columns, the feature key and every coordinate transformation
        the old one had -- and then saved, so the new columns land in the
        element's parquet in the store. `df` must be an in-memory dataframe (what
        `get_transcripts` returns); the element it replaces is what backs the
        parquet being rewritten, so nothing may still be reading from it.

        Columns holding python objects, such as the `{cell_id: score}` dicts in
        `cell_ids`, have to be stored as their `str()` repr -- parquet has no
        type for them -- and are read back the same way, via `ast.literal_eval`.
        """
        key = self._points_key(fov)
        old = self.sdata.points[key]
        transformations = get_transformation(old, get_all=True)
        feature_key = old.attrs.get("spatialdata_attrs", {}).get("feature_key")

        df = df.reset_index(drop=True)
        coordinates = {"x": "x", "y": "y"}
        if "z" in df.columns:
            coordinates["z"] = "z"
        kwargs = {}
        if feature_key and feature_key in df.columns:
            kwargs["feature_key"] = feature_key

        self.sdata.points[key] = PointsModel.parse(
            df,
            coordinates=coordinates,
            transformations=transformations,
            **kwargs,
        )
        self.save_element(key)

    def save_table(self, name, adata):
        """Put an AnnData into the sdata as a table and save it."""
        self.sdata.tables[name] = adata
        self.save_element(name)
        return adata

    def has_column(self, fov, column):
        """True if the FOV's transcript table already carries `column`.

        Reads the parquet's schema only, not its contents.
        """
        key = self._points_key(fov)
        if key not in self.sdata.points:
            return False
        return column in self.sdata.points[key].columns

    def get_fov_affine(self, fov):
        """The FOV's pixel -> "global" mapping, as a 4x4 `(x, y, z)` affine.

        This is the transform the SpatialData's coordinate systems carry: it
        folds in the FOV's offset within the experiment and, when a `pixel_size`
        was encoded at conversion time, the physical scale of a voxel.
        """
        _, matrix = SpatialDataHelpers._labels_global_affine(self.get_labels(fov))
        return matrix

    def get_dist_between_slices(self, fov):
        """Distance between adjacent z-slices, from the sdata's coordinate system.

        Both scales are read off the labels element's "global" transform — the
        voxel depth and the in-plane pixel size that `softseg_to_spatialdata`
        encoded from its `pixel_size`. The z spacing is the same quantity
        `SpatialDataHelpers.masks_to_shapes` uses to place each slice in the 3D
        stack coordinate system, so distances measured across slices here agree
        with the geometry stored in the object.

        The result is returned **in pixels**, as the ratio of the two: cell
        polygons live in raw pixel coordinates, so an out-of-plane distance has
        to be in pixels as well before it can be combined with an in-plane one.
        A dataset with no pixel size encoded is isotropic by definition and gets
        1.0, one z-slice being one pixel deep.

        Returns None for a 2D FOV, which has no slices to be spaced apart.
        """
        if not self.is_3d(fov):
            return None
        matrix = self.get_fov_affine(fov)
        z_spacing = float(matrix[2, 2])
        # physical length of one in-plane pixel step (the x axis of the linear part)
        in_plane = float(np.hypot(matrix[0, 0], matrix[1, 0]))
        if not in_plane:
            return z_spacing
        return z_spacing / in_plane

    def valid_slices(self, fov):
        """The z-slice indices this FOV's segmentation covers.

        Taken from the `valid_z` policy `softseg_to_spatialdata` recorded in
        `sdata.attrs` when present, else every plane of the mask.
        """
        recorded = SpatialDataHelpers._recorded_valid_z(self.sdata, fov)
        if recorded:
            return sorted(int(v) for v in recorded)
        el = self.get_labels(fov)
        if "z" in getattr(el, "dims", ()):
            return list(range(int(el.sizes["z"])))
        return [0]

    def transcript_slices(self, fov, tr):
        """Map each transcript of `tr` to the z-slice index it sits on.

        Uses the same rule as `SpatialDataHelpers.aggregate_zslice_shapes` and
        `select_zslice`: a transcript belongs to the valid slice its z rounds to,
        and one that rounds to no valid slice is either snapped to the nearest
        slice or marked NaN (belonging to no slice), per the `snap_z` policy
        recorded in `sdata.attrs`. A 2D FOV puts everything on slice 0.
        """
        allowed = np.asarray(self.valid_slices(fov), dtype=float)
        z_col = "z" if "z" in tr.columns else None
        if z_col is None and "global_z" in tr.columns:
            z_col = "global_z"
        if not self.is_3d(fov) or z_col is None:
            return np.full(len(tr), allowed[0])

        zvals = tr[z_col].to_numpy(dtype=float)
        nearest, off = SpatialDataHelpers._assign_points_to_slices(
            zvals, allowed, SpatialDataHelpers._resolve_snap_z(self.sdata, None)
        )
        if off.any():
            self.logger.info(
                f"fov_{fov:0>4}: {int(off.sum())} transcripts do not land on a "
                f"segmented z-slice."
            )
        return nearest

    def get_cell_sizes(self, fov):
        """`{cell_id: size}` for the FOV, size being the cell's voxel count.

        Counted straight off the labels array, so it is the exact number of
        voxels the segmentation gives the cell -- which is what `min_size` and
        `max_size` are compared against.
        """
        counts = np.bincount(np.asarray(self.get_mask(fov)).ravel())
        # mask value m -> cell id m-1; 0 is background
        return {m - 1: int(counts[m]) for m in range(1, len(counts)) if counts[m]}

    def get_cell_centroids(self, fov):
        """`{cell_id: (x, y, z)}` for the FOV, in the "global" coordinate system.

        Delegates to `spatialdata.get_centroids`, which computes the exact
        voxel-weighted centre of each label -- across z for a 3D mask -- and maps
        it through the element's transform, so the FOV's offset within the
        experiment and any encoded pixel size are already applied. Coordinates
        follow spatialdata's raster convention, where voxel `i` is centred at
        `i + 0.5`.
        """
        centroids = get_centroids(self.get_labels(fov), coordinate_system="global")
        if hasattr(centroids, "compute"):
            centroids = centroids.compute()

        axes = [ax for ax in ("x", "y", "z") if ax in centroids.columns]
        out = {}
        for label, row in centroids.iterrows():
            point = [float(row[ax]) for ax in axes]
            while len(point) < 3:  # a 2D mask has no z
                point.append(0.0)
            out[int(label) - 1] = tuple(point)
        return out

    def _size_excluded_cells(self, fov, min_size=None, max_size=None):
        """Cell ids whose mask falls outside the requested size range."""
        if min_size is None and max_size is None:
            return set()
        return {
            cell_id
            for cell_id, n in self.get_cell_sizes(fov).items()
            if (min_size is not None and n < min_size)
            or (max_size is not None and n > max_size)
        }

    def _border_cell_ids(self, fov):
        """Cell ids whose mask touches the edge of the FOV on any z-slice.

        These cells are cut off by the FOV boundary, so any centroid or size
        measured here would only describe their visible fragment. A cell is
        counted as touching if its label appears anywhere on the edge row or
        column of any slice.
        """
        stack = self.get_mask(fov)
        if stack.ndim == 2:
            stack = stack[None, ...]
        touching = set()
        for plane in stack:
            touching.update(np.unique(plane[0, :]))
            touching.update(np.unique(plane[-1, :]))
            touching.update(np.unique(plane[:, 0]))
            touching.update(np.unique(plane[:, -1]))
        touching.discard(0)
        return {int(m) - 1 for m in touching}

    def get_cell_shapes(self, fov, min_size=None, max_size=None):
        """The FOV's cells as polygons, aggregated across the z axis.

        A 3D segmentation is held in the SpatialData as one 2D shapes element
        per z-slice (geopandas has no 3D polygons), so this reunites the slices
        of each cell:

            {cell_id: {z_index: shapely geometry}}

        Existing `f"{fov}_z{z}_shapes"` elements are reused; if the FOV has none,
        they are contoured on the fly with
        `SpatialDataHelpers.masks_to_shapes` (without modifying `sdata`).

        Coordinates: the polygons are in the FOV's own pixel space, the same
        space the transcripts' `x`/`y` are in. A contour traced from the labels
        array comes out in mask-array indices, while a transcript at `(x, y)`
        sits at mask pixel `[y - 1, x - 1]`; `masks_to_shapes` closes that
        one-pixel gap by shifting every vertex by its `polygon_offset` (1.0 by
        default), so a transcript can be tested against a polygon directly, with
        no correction of its own. Placing the polygons in the experiment-wide
        frame instead is the job of the FOV's `"global"` transform
        (`get_fov_affine`), which is not applied here.

        Cell ids follow the SoftSeg convention `cell_id = mask_value - 1`.
        Cells outside the `min_size`/`max_size` voxel-count range are omitted.
        """
        source = self.sdata
        try:
            # reuse the shapes already in the object rather than re-contouring
            slices = SpatialDataHelpers._discover_zslice_shapes(source, fov)
            self.logger.debug(
                f"fov_{fov:0>4}: reusing {len(slices)} existing z-slice shapes."
            )
        except KeyError:
            self.logger.info(
                f"fov_{fov:0>4}: no z-slice shapes in the SpatialData, "
                f"contouring {self._labels_key(fov)}."
            )
            source = SpatialDataHelpers.masks_to_shapes(
                self.sdata,
                labels_keys=[self._labels_key(fov)],
                pool_size=1,
                inplace=False,
            )
            slices = SpatialDataHelpers._discover_zslice_shapes(source, fov)

        excluded = self._size_excluded_cells(fov, min_size, max_size)

        by_cell = {}
        for z, name in slices.items():
            gdf = source.shapes[name]
            for cell_id, geom in zip(gdf["cell_id"], gdf.geometry):
                cell_id = int(cell_id)
                if cell_id in excluded or geom is None or geom.is_empty:
                    continue
                by_cell.setdefault(cell_id, {})[int(z)] = geom
        return by_cell

    def get_all_fovs(self):
        """Returns a list of all valid fov indicies for this object."""
        labels = {self._fov_of(k, "_labels") for k in self.sdata.labels}
        points = {self._fov_of(k, "_points") for k in self.sdata.points}
        return sorted(labels & points, key=str)

    def get_complete_fovs(self, column="cell_ids"):
        """Returns a list of all fov indicies this object has already run on.

        A FOV counts as complete once its transcript table carries `column`.
        """
        return [f for f in self.get_all_fovs() if self.has_column(f, column)]

    def get_incomplete_fovs(self, column="cell_ids"):
        """Returns a list of all fov indicies that still need to be run by this object."""
        return [f for f in self.get_all_fovs() if not self.has_column(f, column)]

    def random_complete_fov(self):
        fovs = self.get_complete_fovs()
        random.shuffle(fovs)
        while len(fovs) > 0:
            yield fovs.pop()

    def random_cell_in_fov(self, fov):
        """
        Yields random cell ids from cells that have transcripts mapped to them.
        """

        im = self.get_mask(fov)
        tr = self.get_transcripts(fov)

        # strip out all possible cell ids from tr
        eligible = set()
        for r, row in tr.iterrows():
            assigned = ast.literal_eval(row["cell_ids"])
            eligible.update(list(assigned.keys()))

        candidates = list(np.unique(im)[1:])
        # exclude first element because it's always 0
        # which is background, not a cell

        random.shuffle(candidates)

        # do this to save memory
        del im
        del tr

        while len(candidates) > 0:
            cell = candidates.pop()
            # check against transcripts
            if str(cell - 1) in eligible:
                yield cell - 1

    def cell_to_fov(self, cell_ids):
        """
        Given
            int: cell_ids OR
            [int]: cell_ids
        Returns the FOV that the cell is in, in the form
            {int: cell id, str: fov}
        This is inefficient so avoid it if you can.
        """

        # convert to list if it's a singluar cell
        if isinstance(cell_ids, int):
            cell_ids = [cell_ids]

        results = {}
        for f in self.get_complete_fovs():
            trs = self.get_transcripts(f)
            for i, row in trs.iterrows():
                for c, v in ast.literal_eval(row["cell_ids"]).items():
                    if int(c) in cell_ids:
                        results[int(c)] = f
                        if len(results) == len(cell_ids):
                            return results
                        cell_ids.remove(c)
        return results

    def plot_completed_cell(
        self,
        fov,
        cells,
        gene_col_name="gene",
        dist_between_slices=None,
        project_neighbor_slices=True,
        tr_hi=None,
    ):
        """
        dist_between_slices: spacing between z-slices used when projecting a
           cell's contour onto a neighbouring slice. If None (default) it is read
           off the sdata's coordinate system; pass `project_neighbor_slices=False`
           to switch the projection off entirely.

        tr_hi formatting:
        OPTIONAL dict key: color to use, values are dicts below:
        dict key: column to compare to transcript dataframe
           value: single value to match against OR list of values
        """
        im = self.get_mask(fov)
        if im.ndim == 2:  # treat a flat mask as a single-slice stack
            im = im[None, ...]
        if dist_between_slices is None and project_neighbor_slices:
            dist_between_slices = self.get_dist_between_slices(fov)
        if not project_neighbor_slices:
            dist_between_slices = None
        tr = self.get_transcripts(fov)

        # collect relevant transcripts
        sel_tr = []
        for r, row in tr.iterrows():
            assigned = ast.literal_eval(row["cell_ids"])
            if "lank" not in row[gene_col_name] and any(
                [str(cell) in assigned.keys() for cell in cells]
            ):
                cs = [0, 0, 0]
                for i in range(len(cells)):
                    if str(cells[i]) in assigned.keys():
                        cs[i] = assigned[str(cells[i])]

                sel_tr.append(
                    {
                        "x": row["x"],
                        "y": row["y"],
                        "z": row["global_z"],
                        "c0": cs[0],
                        "c1": cs[1],
                        "c2": cs[2],
                        "gene": row[gene_col_name],
                        "index": row["index"],
                    }
                )
        sel_tr = pd.DataFrame(sel_tr)

        # find our bounding box
        x_0 = int(max(np.nanmin(sel_tr["x"]) - 10, 0))
        x_1 = int(min(np.nanmax(sel_tr["x"]) + 10, np.shape(im)[-2]))
        y_0 = int(max(np.nanmin(sel_tr["y"]) - 10, 0))
        y_1 = int(min(np.nanmax(sel_tr["y"]) + 10, np.shape(im)[-1]))

        im8 = [(im == cell + 1).astype(np.uint8)[:, y_0:y_1, x_0:x_1] for cell in cells]

        # go through each z slice
        for z in range(np.shape(im8)[1]):
            clen = np.shape(im8)[0]

            c_valid = []
            c_neighbor = []
            cnt = []
            for c in range(clen):
                c_neighbor.append(False)
                if len(np.unique(im8[c][z, :, :])) < 2:
                    print(f"Cell {cells[c]} not present on slice {z}")
                    cnt.append(None)
                    c_valid.append(False)
                else:
                    cnt.append(
                        cv2.findContours(
                            im8[c][z, :, :], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                        )[0]
                    )
                    c_valid.append(True)

            if dist_between_slices is None and not any(c_valid):
                print("No valid cells in this slice.")
                continue

            # go through and find applicable neighbors, if we're dealing with that
            if dist_between_slices is not None:
                for i in range(len(cnt)):
                    if not c_valid[i]:
                        if z > 0:
                            # check z-1
                            if len(np.unique(im8[i][z - 1, :, :])) == 2:
                                c_neighbor[i] = True
                                c_valid[i] = True
                                cnt[i] = cv2.findContours(
                                    im8[i][z - 1, :, :],
                                    cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE,
                                )[0]
                        if z < np.shape(im8[i])[0] - 1:
                            # check z+1
                            if len(np.unique(im8[i][z + 1, :, :])) == 2:
                                c_neighbor[i] = True
                                c_valid[i] = True
                                cnt[i] = cv2.findContours(
                                    im8[i][z + 1, :, :],
                                    cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE,
                                )[0]

            if not any(c_valid) and not any(c_neighbor):
                print("No valid cells in this slice, including neighbor slices.")
                continue

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title(f"fov_{fov:0>4}, {cells}, z-slice {z}")

            # plot our value map
            _, ylen, xlen = np.shape(im8[0])

            raw_dist = np.empty((ylen, xlen, clen), dtype=np.float32)
            for i in range(ylen):
                for j in range(xlen):
                    for c in range(clen):
                        if c_valid[c]:
                            dist = cv2.pointPolygonTest(cnt[c][0], (j, i), True)
                            if not c_neighbor[c]:
                                raw_dist[i, j, c] = dist
                            else:
                                if dist < 0:
                                    raw_dist[i, j, c] = -1 * np.sqrt(
                                        dist**2 + dist_between_slices**2
                                    )
                                else:
                                    raw_dist[i, j, c] = max(
                                        -1 * np.sqrt(dist**2 + dist_between_slices**2),
                                        dist_between_slices * -1,
                                    )

            drawing = np.zeros((ylen, xlen, 3))
            for i in range(ylen):
                for j in range(xlen):
                    drawing[i, j] = (
                        (self.decay_func(raw_dist[i, j, 0]) if c_valid[0] else 0),
                        (
                            self.decay_func(raw_dist[i, j, 1])
                            if clen > 1 and c_valid[1]
                            else 0
                        ),
                        (
                            self.decay_func(raw_dist[i, j, 2])
                            if clen > 2 and c_valid[2]
                            else 0
                        ),
                    )

            for i in range(len(cnt)):
                col = (1, 1, 1)
                if c_neighbor[i]:
                    col = (0.5, 0.5, 0.5)
                draw_im = cv2.drawContours(drawing, cnt[i], 0, col, 1)

            ax.imshow(draw_im)

            # go through and plot the relevant transcripts
            for r, row in sel_tr.iterrows():
                if int(row["z"]) == z:
                    facecolor = (row["c1"], row["c2"], row["c0"])
                    if tr_hi is not None:
                        for k, vs in tr_hi.items():
                            if isinstance(vs, dict):
                                # then k is the color to use
                                for ind, vals in vs.items():
                                    if isinstance(vals, list):
                                        if row[ind] in vals:
                                            facecolor = k
                                    else:
                                        if row[ind] == vals:
                                            facecolor = k
                            else:
                                if isinstance(vs, list):
                                    if row[k] in vs:
                                        facecolor = (1, 1, 1)
                                else:
                                    if row[k] == vs:
                                        facecolor = (1, 1, 1)
                    circle = Circle(
                        (int(row["x"]) - x_0, int(row["y"]) - y_0),
                        radius=1,
                        linewidth=0.0,
                        facecolor=facecolor,
                    )
                    ax.add_patch(circle)

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            plt.show(block=False)

        del im
        del tr

    def _blur_one_shape(
        self,
        cell_id,
        geom,
        rows,
        x,
        y,
        cell_ids,
        max_dist,
        projected_dist=None,
    ):
        """Score one cell's polygon against the transcripts on its z-slice.

        rows: indices into x/y/cell_ids of the transcripts sitting on this slice.
        projected_dist: if not None, this polygon comes from a *neighbouring*
           z-slice and every distance is lengthened by this out-of-plane offset.

        Transcripts further than max_dist outside the polygon are left alone;
        the rest get `{cell_id: decay_func(signed distance)}` merged into their
        entry of cell_ids.
        """
        if len(rows) == 0:
            return

        # Only transcripts inside the polygon's bounding box (grown by max_dist)
        # can possibly be within max_dist of it. Projecting onto a neighbouring
        # slice only ever lengthens a distance, so this stays conservative.
        minx, miny, maxx, maxy = geom.bounds
        xs, ys = x[rows], y[rows]
        near = (
            (xs >= minx - max_dist)
            & (xs <= maxx + max_dist)
            & (ys >= miny - max_dist)
            & (ys <= maxy + max_dist)
        )
        rows = rows[near]
        if len(rows) == 0:
            return

        dist = signed_distance(geom, x[rows], y[rows])

        if projected_dist is not None:
            # the transcript is one slice away from the contour, so its true
            # separation is the hypotenuse of (in-plane distance, slice spacing).
            # A transcript inside the projected contour is simply the slice
            # spacing away from it.
            hypot = -1 * np.sqrt(dist**2 + projected_dist**2)
            dist = np.where(dist < 0, hypot, np.maximum(hypot, -1 * projected_dist))

        keep = dist > -1 * max_dist
        name = str(cell_id)
        for j, d in zip(rows[keep], dist[keep]):
            # plain floats, so the dicts stay round-trippable through
            # str() -> ast.literal_eval() when the table is written out
            cell_ids[j][name] = float(self.decay_func(float(d)))

    def blur_fov(
        self,
        f,
        min_size,
        max_dist,
        dist_between_slices=None,
        project_neighbor_slices=True,
        disable_tqdm=False,
    ):
        """Run the first step of soft-segmentation, where masks are blurred and
                multiple float values assigned to each transcript, corresponding to which
                cells they may be members of and their relative likelihoods.

                Both the segmentation and the transcripts come from the bound SpatialData
                object. A 3D segmentation is handled through
                `SpatialDataHelpers.masks_to_shapes`, which splits the mask into one
                polygon set per z-slice; `get_cell_shapes` reunites those slices per cell,
                and each transcript is matched to its slice with the same rule
                `SpatialDataHelpers.aggregate_zslice_shapes` uses.

                f: fov number
                min_size: minimum size for eligible masks, in voxels.
                max_dist: the maximum distance between a transcript and a mask to be considered eligible.
                dist_between_slices: for 3d data, slices with no contours within 1 zslice
                 of a slice with a valid contour will project that contour with this added
                 distance. If None (default), the spacing is read off the sdata's
                 coordinate system (`get_dist_between_slices`).
                project_neighbor_slices: set False to disable that projection entirely.

        writes: the `cell_ids` column of the FOV's points element, saved to the store
                returns: dataframe of transcript information (None during a pooled run)
        """
        tr = self.get_transcripts(f)

        if len(tr) > 0:
            self.logger.info(f"[{datetime.now()}] starting fov_{f:0>4}...")

            # override if if it's alraedy there'
            if "cell_ids" in tr.keys():
                tr.drop(["cell_ids"], axis=1, inplace=True)

            # the spacing between z-slices lives in the sdata's coordinate
            # system, so read it from there unless we were handed one
            if self.is_3d(f) and project_neighbor_slices:
                if dist_between_slices is None:
                    dist_between_slices = self.get_dist_between_slices(f)
            else:
                dist_between_slices = None

            # slices in the segmentation may not line up with the transcripts'
            # z values, so resolve each transcript to a slice up front
            elig_z = self.valid_slices(f)
            slice_of_tr = self.transcript_slices(f, tr)
            tr_by_slice = {z: np.flatnonzero(slice_of_tr == z) for z in elig_z}

            x = tr["x"].to_numpy(dtype=float)
            y = tr["y"].to_numpy(dtype=float)
            cell_ids = [{} for _ in range(len(tr))]

            # each cell as {z: polygon}, i.e. its 2D slices aggregated back
            # together along the z axis
            shapes_by_cell = self.get_cell_shapes(f, min_size=min_size)

            if not disable_tqdm:
                pbar = tqdm(total=len(shapes_by_cell))

            # ascending cell id, so each transcript's dict of candidates is
            # ordered by cell id -- `assign_to_cell` returns the first key that
            # clears its thresholds, so the order is part of the result
            for cell_id in sorted(shapes_by_cell):
                by_z = shapes_by_cell[cell_id]

                for z in elig_z:
                    rows = tr_by_slice[z]
                    if len(rows) == 0:
                        continue

                    geom = by_z.get(z)
                    if geom is not None:
                        self._blur_one_shape(
                            cell_id, geom, rows, x, y, cell_ids, max_dist
                        )
                        continue

                    if dist_between_slices is None:
                        continue

                    # no contour on this slice: borrow a neighbouring one.
                    # Realistically we should not have a case where there are two
                    # neighboring slices that both have contours surrounding a
                    # slice with no contours. So just do basic check.
                    neighbor = z - 1 if (z - 1) in by_z else (z + 1)
                    if neighbor not in by_z:
                        self.logger.debug(
                            f"Cell ID {cell_id} zslice {z} has no eligible neighbors. Skipping zslice {z}."
                        )
                        continue
                    self.logger.debug(
                        f"Assigning cell ID {cell_id} zslice {z} nearest neighbor as {neighbor}."
                    )
                    self._blur_one_shape(
                        cell_id,
                        by_z[neighbor],
                        rows,
                        x,
                        y,
                        cell_ids,
                        max_dist,
                        projected_dist=dist_between_slices,
                    )

                if not disable_tqdm:
                    pbar.update(1)

            del shapes_by_cell

            if not disable_tqdm:
                pbar.close()

            # parquet has no type for a python dict, so the per-transcript
            # {cell_id: score} maps are stored as their str() repr and read back
            # with ast.literal_eval
            tr["cell_ids"] = [str(d) for d in cell_ids]
            self.set_transcripts(f, tr)

            self.logger.info(
                f"[{datetime.now()}] saved fov_{f:0>4}\n\t{len(tr)} transcripts"
            )

            tr = tr.set_index("index")

            """
            intensities = []
            for i, row in tr.iterrows():
                for k, v in row["cell_ids"].items():
                    intensities.append(v)

            print(f"Completed fov_{f:0>4}.")
            print(
                f"\t{len(intensities)} assigned to cells, {len(tr)} transcripts total."
            )
            print(f"\ttime taken:{(time.time() - t1)/60} minutes.")
            """

            # Going to assume that disabled tqdm is a part of a batch run
            if not disable_tqdm:
                return tr
            else:
                del tr
                return None

        else:
            print(f"Skipping fov_{f:0>4}, no transcripts found.")
            return None

    def blur_all_fovs(
        self,
        min_size,
        max_dist,
        dist_between_slices=None,
        project_neighbor_slices=True,
        sel_fovs=None,
    ):
        """Run the first step of soft-segmentation, where masks are blurred and
        multiple float values assigned to each transcript, corresponding to which
        cells they may be members of and their relative likelihoods.

        This method will run on all eligible FOVs, using the multiprocessing pool.

        pool_size: the number of threads to be used.
        min_size: minimum size for eligible masks.
        max_dist: the maximum distance between a transcript and a mask to be considered eligible.
        dist_between_slices: as in blur_fov; read per-FOV off the sdata's
            coordinate system when left as None.

        writes: the `cell_ids` column of every FOV's points element, saved to the store.
        """
        if sel_fovs is None:
            sel_fovs = self.get_incomplete_fovs()

        self._run_over_fovs(
            zip(
                repeat(SoftAssigner.blur_fov),
                sel_fovs,
                repeat(min_size),
                repeat(max_dist),
                repeat(dist_between_slices),
                repeat(project_neighbor_slices),
                repeat(True),
            ),
            len(sel_fovs),
            self.pool_size,
        )

    def combine_soft_transcripts(self, sel_fovs=None):
        """Every FOV's soft-assignment table, concatenated into one dataframe.

        Also records the total transcript count on the instance, which
        `calculate_confident_threshold` uses to size its working arrays.
        """
        if sel_fovs is None:
            sel_fovs = self.get_complete_fovs()

        full_tr = pd.concat(
            [self.get_transcripts(f) for f in sel_fovs], ignore_index=True
        )
        self.num_transcripts = len(full_tr)
        return full_tr

    def count_transcripts(self, sel_fovs=None):
        """Record the total number of transcripts across the given FOVs.

        Reads each FOV's row count rather than materialising the tables, so it
        is the cheap way to set `num_transcripts` when the concatenated table
        from `combine_soft_transcripts` is not itself needed.
        """
        if sel_fovs is None:
            sel_fovs = self.get_complete_fovs()

        total = 0
        for f in sel_fovs:
            el = self.sdata.points[self._points_key(f)]
            total += int(
                el.shape[0].compute() if hasattr(el.shape[0], "compute") else len(el)
            )
        self.num_transcripts = total
        return total

    def assign_to_cell(self, assigned, min_thresh=None):
        """
        modularizing the transcript assignment method to make things more easily
        changable in the future.
        assigned: dict():
              key: cell id
            value: confidence assigned to the transcript w this cell

        returns: id of cell (if valid), or None (if no valid target)
        """
        total = sum([float(v) for v in assigned.values()])
        if total == 0:
            return None
        for k, v in assigned.items():
            if (min_thresh is None or float(v) / total > min_thresh) and float(
                v
            ) >= 0.5:
                return k
        return None

    def calculate_confident_threshold(self, show_plots=False):
        """
        Finds the ideal threshold for what should be considered a 'confident' assignment.

        returns:
        "aggresive": defines threshold as elbow point of assigned transcripts vs
            multiply-assigned transcripts plot.
        "conservative": defines threshold as lowest value where no transcript is
            multiply-assigned.
        """
        if hasattr(self, "num_transcripts"):
            tr_tally = self.num_transcripts
        else:
            tr_tally = 10000000
        n_wid = 12  # initial guess for max number of cells a transcript is assigned to
        cells = [np.empty((tr_tally, n_wid))]
        cutoff = [np.empty((tr_tally, n_wid))]
        mat_ind = 0
        c_row = 0
        c_row_total = 0
        cell_list = set()

        sel_fovs = self.get_complete_fovs()

        for f in sel_fovs:
            self.logger.info(f"Reading in fov {f}\t{datetime.now()}")
            tr = self.get_transcripts(f)
            for r, row in tr.iterrows():
                assigned = ast.literal_eval(row["cell_ids"])
                ind = 0

                # make the mat wider if needed
                if np.shape(cutoff[mat_ind])[1] < len(assigned):
                    n_wid = len(assigned)  # - np.shape(cutoff)[1]
                    cells = np.append(
                        cells[mat_ind], np.empty((tr_tally, n_wid)), axis=1
                    )
                    cutoff = np.append(
                        cutoff[mat_ind], np.empty((tr_tally, n_wid)), axis=1
                    )
                    n_wid = np.shape(cutoff)[1]

                if np.shape(cutoff[mat_ind])[0] == c_row:  # go to next ind
                    cells.append(np.empty((tr_tally, n_wid)))
                    cutoff.append(np.empty((tr_tally, n_wid)))
                    mat_ind += 1
                    c_row_total += c_row
                    c_row = 0

                # transfer in our floats...
                for k, v in assigned.items():
                    cells[mat_ind][c_row, ind] = k
                    cutoff[mat_ind][c_row, ind] = v
                    ind += 1
                    cell_list.update(k)

                c_row += 1
            self.logger.info(
                f"Just completed fov_{f:0>4}, mat number {mat_ind}, current c_row {c_row + c_row_total}"
            )

        # now merge the lists, if we need to
        combined = np.empty((0, n_wid))
        for c in cells:
            combined = np.append(combined, c, axis=0)
        del cells
        cells = combined

        combined_cutoff = np.empty((0, n_wid))
        for c in cutoff:
            combined_cutoff = np.append(combined_cutoff, c, axis=0)
        del cutoff
        cutoff = combined_cutoff

        assigned_tr = []
        cell_dist = []
        num = 60  # resolution for estimate
        rang = np.linspace(0.01, 0.60, num=num)

        for thresh in rang:
            # this is going to be a bit slower but whatever
            this_tr = 0
            this_cells = []
            for i in range(len(cells)):
                cell = self.assign_to_cell(
                    {cells[i][j]: cutoff[i][j] for j in range(len(cells[i]))}, thresh
                )
                if cell is not None:
                    this_tr += 1
                    this_cells.append(cell)

            assigned_tr.append(this_tr)

            # now some basic filtering like we do when clustering
            # remove cells that do not reach a specific transcript count
            min_count = 10

            counts = Counter(this_cells)
            tally = 0
            for k, v in counts.items():
                if k == 0.0:
                    continue
                if v > min_count:
                    tally += 1

            cell_dist.append(tally)

        dif1 = np.diff(cell_dist)
        dif2 = np.diff(dif1)

        if show_plots:
            fig, ax = plt.subplots()
            im = ax.scatter(assigned_tr, cell_dist, c=rang)
            # ax.set_xscale("log")
            # ax.set_yscale("log")
            ax.set_xlabel("Total transcripts assigned")
            ax.set_ylabel(f"Number of cells with {min_count} transcripts assigned")
            fig.colorbar(im, ax=ax)
            plt.show(block=False)

            fig, ax = plt.subplots()
            im = ax.scatter(assigned_tr[1:], dif1, c=rang[1:])
            # ax.set_xscale("log")
            # ax.set_yscale("log")
            ax.set_xlabel("Total transcripts assigned")
            ax.set_ylabel(
                f"Number of cells with {min_count} transcripts assigned, 1st derivative"
            )
            fig.colorbar(im, ax=ax)
            plt.show(block=False)

            fig, ax = plt.subplots()
            im = ax.scatter(assigned_tr[2:], dif2, c=rang[2:])
            # ax.set_xscale("log")
            # ax.set_yscale("log")
            ax.set_xlabel("Total transcripts assigned")
            ax.set_ylabel(
                f"Number of cells with {min_count} transcripts assigned, 2nd derivative"
            )
            fig.colorbar(im, ax=ax)
            plt.show(block=False)

        percentile = np.argmax(dif2)

        ass_per = [cell_dist[i] / assigned_tr[i] * 100 for i in range(num)]
        fir = ass_per.index(min(ass_per))

        if show_plots:
            plt.show(block=False)

        # defaults to saving the conservative threshold
        self.conf_thresh = rang[fir]

        return {
            "aggresive": {
                "threshold": rang[percentile + 2],
                "total_assigned": assigned_tr[percentile + 2],
                "unique_cells": cell_dist[percentile + 2],
            },
            "conservative": {
                "threshold": rang[fir],
                "total_assigned": assigned_tr[fir],
                "unique_cells": cell_dist[fir],
            },
        }

    def convert_to_adata(
        self,
        gene_col_name="gene",
        min_thresh=None,
        assigned_col=None,
        sel_fovs=None,
        table_name=None,
    ):
        """
        Converts all completed analyses to adata format.

        The result is stored as a table in the sdata (and saved to the store)
        under `table_name`, defaulting to "cxg", or
        "cxg_resegmented_{assigned_col}" when reading from an assignment column.

        When determining what transcripts go into what cells:
        - If assigned_col is provided, will pull the cell id from that column.
        - Otherwise, self.assign_to_cell will be passed the "cell_ids" column.
          If min_thresh is provided, that will be passed on to that method.

        Each cell is then labelled with its size and centroid, measured from the
        segmentation in the sdata: the cell's polygons are taken slice by slice
        and aggregated along the z axis, area-weighted, then mapped through the
        FOV's "global" transform. `size` is the summed polygon area in pixels;
        `x_coords`/`y_coords`/`z_coords` are in the units of the sdata's "global"
        coordinate system, which carries both the FOV's offset within the
        experiment and the physical pixel size, if one was encoded.
        """

        cxg_dict = {}
        if sel_fovs is None:
            sel_fovs = self.get_complete_fovs()

        print("Reading cell by gene data from transcript tables.")
        for f in tqdm(sel_fovs):
            tr = self.get_transcripts(f)
            self.logger.info(f"[{datetime.now()}] started reading in fov_{f:0>4}.")

            if assigned_col is not None:
                if assigned_col not in tr.columns:
                    self.logger.info(
                        f"{assigned_col} not in df for fov_{f:0>4}, skipping."
                    )
                    continue
                tr = tr[~pd.isnull(tr[assigned_col])]
                tallies = Counter(list(zip(tr[gene_col_name], tr[assigned_col])))
            else:
                cell_col = tr.cell_ids.apply(
                    lambda x: self.assign_to_cell(ast.literal_eval(x), min_thresh)
                )
                inds = cell_col.apply(lambda x: x is not None)
                tallies = Counter(list(zip(tr[gene_col_name][inds], cell_col[inds])))

            for tup, tally in tallies.items():
                gene, cell = tup
                if cell == "other":
                    # this technically shouldn't be possible but sometimes it happens
                    # bug has since been patched but some old results still have this
                    continue
                cell = int(cell)
                if cell in cxg_dict.keys():
                    cxg_dict[cell][gene] = tally
                else:
                    cxg_dict[cell] = {gene: tally}
            # print(cxg_dict)
            del tr
            self.logger.info(f"[{datetime.now()}] completed reading in fov_{f:0>4}.")

        cxg_df = pd.DataFrame.from_dict(cxg_dict, orient="index")
        # genes and cells otherwise come out in whatever order the transcript
        # tables happened to mention them first, which makes the var/obs order
        # depend on row order rather than on the data
        cxg_df = cxg_df.reindex(sorted(cxg_df.columns), axis=1).sort_index()

        adata = ad.AnnData(cxg_df)

        adata.obs["fov"] = pd.Series(
            [None] * len(adata), index=adata.obs.index, dtype=object
        )
        adata.obs["size"] = pd.DataFrame(np.zeros((len(adata), 1)))
        adata.obs["x_coords"] = pd.DataFrame(np.zeros((len(adata), 1)))
        adata.obs["y_coords"] = pd.DataFrame(np.zeros((len(adata), 1)))
        adata.obs["z_coords"] = pd.DataFrame(np.zeros((len(adata), 1)))

        with pd.option_context("display.max_seq_items", None):
            self.logger.debug(f"All valid cell ids: {adata.obs_names}")

        print("Finding cell centroid data and copying to anndata object")
        for f in tqdm(sel_fovs):
            if self._labels_key(f) not in self.sdata.labels:
                self.logger.info(f"[{datetime.now()}] skipped converting fov_{f:0>4}.")
                continue

            sizes = self.get_cell_sizes(f)
            centroids = self.get_cell_centroids(f)
            # cells clipped by the FOV edge are only partly visible here, so
            # leave them to the FOV that holds the whole cell
            border = self._border_cell_ids(f)

            for k, size in sizes.items():
                if k in border or str(k) not in adata.obs_names:
                    self.logger.debug(f"[{datetime.now()}] cell id {k} not valid.")
                    continue

                sel = adata.obs.index.isin([str(k)])
                adata.obs.loc[sel, "fov"] = f
                adata.obs.loc[sel, "size"] = size

                centroid = centroids.get(k)
                if centroid is not None:
                    adata.obs.loc[sel, "x_coords"] = centroid[0]
                    adata.obs.loc[sel, "y_coords"] = centroid[1]
                    adata.obs.loc[sel, "z_coords"] = centroid[2]

            self.logger.info(f"[{datetime.now()}] completed converting fov_{f:0>4}.")

        adata.X = np.nan_to_num(adata.X)

        if table_name is None:
            table_name = "cxg"
            if assigned_col is not None:
                table_name += f"_resegmented_{assigned_col}"

        self.save_table(table_name, adata)
        print(f"Saved table {table_name!r}")
        return adata

    def get_scoring_matrix(self, adata, cats, normed=True):
        """
        Need to run this before evaluate_overlapping_regions
        cats is the possible cell types as they are described in adata
        formatted as
        { "column name":["cell type", "cell type"], ...}
        """
        all_avg = np.average(adata.X, axis=0)

        filtered = {}
        for k, v in cats.items():
            for cat in v:
                filtered[cat] = deepcopy(adata[adata.obs[k] == cat])

        avgs = {}
        for k, v in filtered.items():
            avgs[k] = np.average(v.X, axis=0)

        filtered["other"] = [0] * len(adata.X)
        avgs["other"] = all_avg

        df_avg = pd.DataFrame.from_dict(avgs, orient="index", columns=adata.var.index)
        if normed:
            norm_total = sum([len(x) for k, x in filtered.items()])
            df_normed = df_avg.multiply(
                [(norm_total - len(x)) / norm_total * 100 for x in filtered.values()],
                axis=0,
            )

            df_comp = df_normed
        else:
            df_comp = df_avg

        self.score_mat = df_comp.fillna(0).to_dict()

        # generating cell_to_type dict, while we have our hands on cats
        self.cell_to_type = {}  # key: cell ID, value: cell type
        self.tr_to_gene = {}  # key: transcript ID, value: gene name
        for r, row in adata.obs.iterrows():
            for key_type, cell_types in cats.items():
                for cell_type in cell_types:
                    if row[key_type] == cell_type:
                        self.cell_to_type[r] = cell_type

    def score_tr_assignment(self, assignment, mse_score=False):
        score = 0
        for cell, trs in assignment.items():
            # it's possible to have untyped cells in a comparison
            # if it is untyped, treat it as an "average" cell.
            if cell in self.cell_to_type.keys():
                cell_type = self.cell_to_type[cell]
            else:
                cell_type = "other"

            if mse_score:
                genes = [self.tr_to_gene[tr] for tr in trs]
                gene_tallies = Counter(genes)
                for gene in self.score_mat.keys():
                    if gene in gene_tallies:
                        score += (
                            self.score_mat[gene][cell_type] - gene_tallies[gene]
                        ) ** 2
                    else:
                        score += self.score_mat[gene][cell_type] ** 2

                # we want higher score = better, so invert the scale
                score = score * -1
            else:
                for tr in trs:
                    # there may be genes that do not contribute to score
                    # (ie: blanks)
                    # TODO: add optional holdout feature here?
                    gene = self.tr_to_gene[tr]
                    if gene in self.score_mat.keys():
                        score += self.score_mat[gene][cell_type]
        return score

    def score_dataset(self, adata, include_other=True):
        """
        Computes an overall score for a dataset by summing, for each cell, the
        score_mat value for each (gene, cell_type) pair weighted by the gene's
        count in that cell. Requires get_scoring_matrix to have been run first.

        Returns a dict with:
            "total": raw sum of scores across all cells
            "per_cell": total / number of cells
            "per_transcript": total / total transcript count
        """
        score_df = pd.DataFrame.from_dict(self.score_mat)
        # restrict to genes present in both adata and score_mat
        shared_genes = [g for g in adata.var_names if g in score_df.columns]
        score_df = score_df[shared_genes]

        counts = adata[:, shared_genes].to_df()

        total = 0.0
        for cell_id, row in counts.iterrows():
            cell_type = self.cell_to_type.get(cell_id, "other")
            if cell_type not in score_df.index:
                cell_type = "other"

            if cell_type != "other" or include_other:
                scores = score_df.loc[cell_type]
                total += float(row.dot(scores))

        n_cells = len(adata)
        n_transcripts = float(counts.values.sum())

        return {
            "total": total,
            "per_cell": total / n_cells if n_cells > 0 else 0.0,
            "per_transcript": total / n_transcripts if n_transcripts > 0 else 0.0,
        }

    def trs_at_thresh(self, thresh, sel_cells, todo_trs, all_trs, sel_index=0):
        """
        Given a set of ambiguous transcripts and their relative assignment scores,
        figure out which transcripts should belong to which members of sel_cells at a given
        threshold value for a given "primary cell".

        thresh: float, given threshold value
        sel_cells: a list of cells, matches the keys in todo_trs
        todo_trs: dict: key: cell id
                  value: np array; each row is a transcript
                        [:,0] -> transcript IDs
                        [:,1] -> assigment score
        all_trs: pandas dataframe, raw reading in the saved csv file
        sel_index: the index in sel_cells that the thresh should apply to.
                   by default, we assume the 0th element.

        RETURNS: dict: key: cell id
                     value: list of transcript IDs
        """

        # we're going to be destroying this object,
        # don't want to cause issues outside the scope of this function
        sel_cells = sel_cells.copy()

        i_cell = str(sel_cells[sel_index])
        sel_cells.remove(
            sel_cells[sel_index]
        )  # sel_cells is now the other cells we need to look at
        result = {}

        # find set of transcripts that are beneat threshold for first sel_cell
        i_trs = list(todo_trs[i_cell][todo_trs[i_cell][:, 1] > thresh][:, 0])
        result[i_cell] = i_trs
        assigned = i_trs.copy()

        # Now we start to do slow stuff if there's more than two in the comparison...
        # For each of the remaining cell ids, repeat threshold inversion
        # (ie find the transcript with the lowest score assigned to previous cell,
        # then get the value for the next cell on that same transcript,
        # use this threshold to find the next set of candidates)
        # and assign transcripts that are still unclaimed
        while len(sel_cells) > 1:
            # find transcript with lowest assignment score for prev cell
            min_tr = todo_trs[i_cell][np.argmin(todo_trs[i_cell][:, 1]), 0]
            min_assigned = ast.literal_eval(
                all_trs[all_trs.index == min_tr]["cell_ids"].values[0]
            )

            # get the value for the next thresh
            c_cell = str(sel_cells[0])
            if c_cell == "other":
                print(f"FATAL PROBLEM.\nsel_cells {sel_cells}\ntodo_trs {todo_trs}")
            inv_thresh = min_assigned[c_cell]

            # take slice of transcripts that are above this new threshold
            interim_result = list(
                todo_trs[c_cell][todo_trs[c_cell][:, 1] > inv_thresh][:, 0]
            )

            # remove items that have been assigned already
            interim_result = [tr for tr in interim_result if tr not in assigned]
            assigned.extend(interim_result)

            result[c_cell] = interim_result

            i_cell = c_cell
            sel_cells.remove(sel_cells[0])

        if len(sel_cells) == 0:
            # CRITICAL ERROR
            return False

        # for the last cell, just assign everything that's left
        last_cell = str(sel_cells[0])
        last_trs = list(todo_trs[last_cell][:, 0])
        result[last_cell] = [tr for tr in last_trs if tr not in assigned]

        return result

    def dict_merge(self, d1, d2, inds):
        """
        Makes addition of two dicts of format
           key: (index)
           value: list
        by looking up inds in both dicts and concatenating their respective lists
        """
        result = {}
        for i in inds:
            list1 = d1.get(i)
            list2 = d2.get(i)
            if list1 is not None:
                if list2 is not None:
                    result[i] = list1 + list2
                else:
                    result[i] = list1
            elif list2 is not None:
                result[i] = list2
        return result

    def trs_at_default(self, todo_trs):
        """
        Given a list of unconfident transcripts, return the assignments as they would be performed under
        strict segmentation (ie, inside the original image boundaries). Basically: assigns all transcripts
        with a assignment score > 0.5.

        RETURNS: dict: key: cell id
                     value: list of transcript IDs
        """
        result = {}
        for cell, mat in todo_trs.items():
            interim_result = mat[mat[:, 1] >= 0.5][:, 0]
            result[cell] = list(interim_result)
        # in the "default" case for a one-cell comparison, we need to spoof the "other" cell
        if len(todo_trs.keys()) == 1:
            cell = list(todo_trs.keys())[0]
            interim_result = todo_trs[cell][todo_trs[cell][:, 1] < 0.5][:, 0]
            result["other"] = list(interim_result)
        return result

    def evaluate_overlapping_regions_single_fov(
        self,
        f,
        gene_col_name="gene",
        min_thresh=None,
        default_thresh=5,
        only_tagged_cells=None,
        use_conf_trs=False,
        use_other_cells=False,
        use_mse_score=False,
        assigned_col="assignment",
        omit_blanks=False,
        auto_assign_single_target=False,
        save_delta_tallies=False,
        disable_tqdm=False,
        overwrite=False,
    ):
        """
        Scores and re-assigns border region transcripts for a single FOV.
        f (int): current FOV
        default_thresh (float): new segmentation must out-score original segmentation
            by a factor of this much in order to be considered "better".
        min_thresh (float): threshold to be used for confident transcript identification
        only_tagged_cells (list): If provided, only cells in this list will be considered for re-evaluation.
        use_conf_trs (bool): If true, confident transcripts will be added to each cell
            when scoring transcript assignments. If false, only ambiguous transcripts will
            be used. NOTE: Setting this to True causes a non-trivial slowdown.
        assigned_col (string): The name of the column for the new assignment to be added to
            for a given FOV's transcript table.
        omit_blanks (bool): if True, all blanks will be categorically ignored. Note that you
            may not want to ignore blanks in cell assignment if you want to quantify
            any kind of spatial error.
        auto_assign_single_target (bool): if True, unconfident transcripts that have a single
            eligible target cell will automatically be assigned to that cell.
        disable_tqdm (bool): if True, this method will not print output or  create
            its own pbar entities.
        overwrite (bool): if True, this method will overwite assigned_col if it already exists.
        """
        if self.has_column(f, assigned_col) and not overwrite:
            self.logger.info(
                f"[{datetime.now()}] skipping fov_{f:0>4}, {assigned_col} already present"
            )
            return

        # transcript id is the row label here: the scoring below addresses
        # transcripts by it, and it is restored to a column before saving
        tr = self.get_transcripts(f).set_index("index")

        self.logger.info(
            f"[{datetime.now()}] evaluate_overlapping_regions starting fov_{f:0>4}..."
        )

        conf_trs = {}  # key: cell, value: list of transcript ids
        unconf_trs = {}
        # key: (tuple of possible cell assignments)
        # value: dict
        #        key: cell
        #        value: list of (transcript id, gradient value)
        #               ^ will be converted to np.array later

        skip_cached = []  # comparison tuples that we know we don't care about

        seg_is_default = {}
        # key: unconf_tup
        # value: True if using original masks, False if using novel mask

        assigned_trs = {}
        # key: cell
        # value: list of transcript IDs

        # if this is part of a run of all FOVs, we don't want to present a tdqm bar
        # for just one FOV (they will be run in parallel and it will break)
        if not disable_tqdm:
            print(
                "\treading transcripts: identify confident trs and valid overlapping regions..."
            )
            pbar = tqdm(total=len(tr))
        else:
            pbar = None

        for index, row in tr.iterrows():
            if pbar is not None:
                pbar.update(1)

            # omit blanks
            if omit_blanks and "lank" in row[gene_col_name]:
                continue

            assigned = ast.literal_eval(row["cell_ids"])

            # skip over any transcript that has no possible cell assignments
            if len(assigned.keys()) == 0:
                continue

            if only_tagged_cells is None or any(
                [str(cell) in assigned.keys() for cell in only_tagged_cells]
            ):
                # TODO: for now, only_tagged_cells will always be false
                # In the future, plan to add support for list of suspect cells

                # try to assign to a single cell
                conf_cell = self.assign_to_cell(assigned, min_thresh)
                self.tr_to_gene[index] = row[gene_col_name]
                if conf_cell is not None:
                    # transcript has confident assignment
                    if conf_cell in conf_trs.keys():
                        conf_trs[conf_cell].append(index)
                    else:
                        conf_trs[conf_cell] = [index]
                else:
                    unconf_tup = tuple(sorted([k for k in assigned.keys()]))
                    if unconf_tup not in unconf_trs.keys():
                        unconf_trs[unconf_tup] = {}

                    # adding values of each possible assignment to proper dict entry
                    for cell, prob in assigned.items():
                        tr_val = (index, prob)
                        if cell in unconf_trs[unconf_tup].keys():
                            unconf_trs[unconf_tup][cell].append(tr_val)
                        else:
                            unconf_trs[unconf_tup][cell] = [tr_val]

                    # dealing with a non-confident transcript
                    if unconf_tup in skip_cached:
                        continue

                    # throw out this comparison if we don't have a type for
                    # all cells present and we are not explicitly treating them
                    # as "other"
                    if not use_other_cells and any(
                        [
                            (
                                cell not in self.cell_to_type.keys()
                                or self.cell_to_type[cell] == "Unassigned"
                            )
                            for cell in unconf_tup
                        ]
                    ):
                        skip_cached.append(unconf_tup)
                        seg_is_default[unconf_tup] = [True]
                        self.logger.info(
                            f"Throwing out tuple {unconf_tup}, contains untyped cell."
                        )
                        continue

                    # we only care about this unconf_tup comparison
                    # if there are at least 2 different celltypes present
                    # AND it is not a single cell's unconfident transcripts
                    if (
                        len(
                            set(
                                [
                                    self.cell_to_type[cell]
                                    for cell in unconf_tup
                                    if cell in self.cell_to_type.keys()
                                ]
                            )
                        )
                        < 2
                    ):
                        if len(unconf_tup) == 1 and use_other_cells:
                            self.logger.info(
                                f"Tuple {unconf_tup} would be thrown out, but we are evaluating it independently."
                            )
                        elif (
                            len(unconf_tup) == 1 and auto_assign_single_target
                        ):  # implicit: not using "other" comparison
                            # treat this as an assigned transcript
                            # while not confident, there is no dispute about
                            # where this transcript belongs
                            if unconf_tup[0] in assigned_trs.keys():
                                assigned_trs[unconf_tup[0]].append(index)
                            else:
                                assigned_trs[unconf_tup[0]] = [index]
                            self.logger.info(
                                f"Assigning transcript {index} to unconfident but unambiguous cell assignment {unconf_tup}."
                            )
                            seg_is_default[unconf_tup] = [True]
                            continue
                        else:
                            skip_cached.append(unconf_tup)
                            seg_is_default[unconf_tup] = [True]
                            self.logger.info(
                                f"Throwing out tuple {unconf_tup}, contains single cell type."
                            )
                            continue

        if pbar is not None:
            pbar.close()

        # convert unconf_trs values to np arrays to save time later
        for unconf_tup in unconf_trs.keys():
            for cell in unconf_trs[unconf_tup].keys():
                unconf_trs[unconf_tup][cell] = np.array(unconf_trs[unconf_tup][cell])

        if len(unconf_trs.keys()) == 0:
            self.logger.info(f"No unconfident transcripts in fov_{f:0>4}, skipping.")
            return

        max_len = max([len(k) for k in unconf_trs.keys()])

        if not disable_tqdm:
            print("\tassigning unconfident transcripts...")
            pbar = tqdm(total=len(unconf_trs))
        else:
            pbar = None

        # start with one way comparisons, then work our way up
        for cur_len in range(1, max_len + 1):
            for unconf_tup, tr_by_cell in unconf_trs.items():
                if len(unconf_tup) != cur_len:
                    continue

                if pbar is not None:
                    pbar.update(1)

                # if this tuple is only in there because we want to treat it as
                # default, don't actually compute any threshes for it
                if unconf_tup in seg_is_default:
                    continue

                elg_cells = list(unconf_tup)

                # coming up with "default" score for comparison
                best_score = np.inf * -1
                best_assignment = self.trs_at_default(tr_by_cell)
                if use_conf_trs:
                    best_assignment = self.dict_merge(
                        best_assignment, conf_trs, elg_cells
                    )
                    best_assignment = self.dict_merge(
                        best_assignment, assigned_trs, elg_cells
                    )

                default_score = self.score_tr_assignment(best_assignment, use_mse_score)

                # handle this more simply if we only have one transcript in this region
                total_trs = sum([len(trs) for trs in tr_by_cell.values()]) / 2
                if total_trs == 1:
                    # manually run this one trascript though all eligible cells
                    for cell in tr_by_cell.keys():
                        assignment = {cell: [tr_by_cell[cell][0, 0]]}
                        score = self.score_tr_assignment(assignment, use_mse_score)
                        if (
                            score > best_score
                            and score > default_score * default_thresh
                        ):
                            best_score = score
                            if "other" not in assignment.keys():
                                best_assignment = assignment
                            else:
                                best_assignment = {
                                    k: v for k, v in assignment.items() if k != "other"
                                }
                else:
                    # coming up with range of thresholds to check
                    maxgrad = max(
                        [max(tr_by_cell[cell][:, 1]) for cell in tr_by_cell.keys()]
                    )
                    mingrad = min(
                        [min(tr_by_cell[cell][:, 1]) for cell in tr_by_cell.keys()]
                    )
                    step = max_len - cur_len + 2

                    if total_trs < step:
                        self.logger.info(
                            f"Comparison region {unconf_tup} has fewer than {step} transcripts, changing step to {total_trs}"
                        )
                        step = total_trs - 1

                    if maxgrad == mingrad:
                        # possible to have multiple transcripts with the same value
                        # just manually fudge this to assign both or neither
                        # (arange does not like having min and max be the same value)
                        maxgrad += 0.01
                        mingrad -= 0.01
                        step = 1

                    # in the case where we are looking at ambiguous transcropts
                    # with one possible cell assignment, we add a nonexistent "other"
                    # cell to compare against
                    if use_other_cells and cur_len == 1:
                        tr_by_cell.update({"other": tr_by_cell[unconf_tup[0]]})
                        elg_cells.append("other")

                    for thresh in np.arange(
                        mingrad, maxgrad, (maxgrad - mingrad) / step
                    ):
                        for prim_ind in range(cur_len):
                            assignment = self.trs_at_thresh(
                                thresh, elg_cells, tr_by_cell, tr, prim_ind
                            )

                            if assignment is False:
                                self.logger.error(
                                    f"CRITICAL PROBLEM.\nfov: {f}\nelg_cells: {elg_cells}\nunconf_tup: {unconf_tup}\ntr_by_cell: {tr_by_cell}"
                                )
                                break

                            if use_conf_trs:
                                assignment = self.dict_merge(
                                    assignment, conf_trs, elg_cells
                                )
                                assignment = self.dict_merge(
                                    assignment, assigned_trs, elg_cells
                                )

                            score = self.score_tr_assignment(assignment, use_mse_score)
                            if (
                                score > best_score
                                and score > default_score * default_thresh
                            ):
                                best_score = score
                                # strip the "other" back out if this was a
                                # one-way comparison
                                if "other" not in assignment.keys():
                                    best_assignment = assignment
                                else:
                                    best_assignment = {
                                        k: v
                                        for k, v in assignment.items()
                                        if k != "other"
                                    }

                seg_is_default[unconf_tup] = best_score == -1

                for cell, trs in best_assignment.items():
                    if len(trs) > 0 and cell != "other":
                        if not use_conf_trs:
                            if cell in assigned_trs.keys():
                                assigned_trs[cell].extend([float(t) for t in trs])
                            else:
                                assigned_trs[cell] = [float(t) for t in trs]
                        else:
                            # if we are using the conf assignments, we need to remove them from the assignment pool.
                            actual_trs = []
                            already_used = self.dict_merge(
                                conf_trs, assigned_trs, [cell]
                            )
                            if cell not in already_used.keys():
                                continue
                            already_used = already_used[cell]
                            for t in trs:
                                if t not in already_used:
                                    actual_trs.append(t)
                            if cell in assigned_trs.keys():
                                assigned_trs[cell].extend(
                                    [float(t) for t in actual_trs]
                                )
                            else:
                                assigned_trs[cell] = [float(t) for t in actual_trs]
        # print([k for k in unconf_trs.keys()])

        # assign things that were left as default
        for unconf_tup, is_default in seg_is_default.items():
            if is_default:
                if unconf_tup in unconf_trs:
                    og_assignment = self.trs_at_default(unconf_trs[unconf_tup])
                    for cell, trs in og_assignment.items():
                        # "other" assignments were for one-way comparisons
                        if cell != "other":
                            if cell in assigned_trs:
                                assigned_trs[cell].extend(trs)
                            else:
                                assigned_trs[cell] = trs
                else:
                    print(f"{unconf_tup} does not have corresponding trs list")

        if pbar is not None:
            pbar.close()

        self.logger.info(f"[{datetime.now()}] updating tr for fov_{f:0>4}")

        # the reassignments themselves do not need saving separately: they are
        # exactly what the assigned_col column below records, per transcript

        # below creates dict of {key: tr ID, value: cell ID}
        inv_assignment = {
            k: v
            for d in [{tr: cell for tr in trs} for cell, trs in assigned_trs.items()]
            for k, v in d.items()
        }
        # now add in confident assignments
        inv_assignment.update(
            {
                k: v
                for d in [{tr: cell for tr in trs} for cell, trs in conf_trs.items()]
                for k, v in d.items()
            }
        )
        # create new column for tr dataframe where row is transcript ID and value is cell assignment
        new_col = tr.apply(
            lambda b: (
                inv_assignment[b.name] if b.name in inv_assignment.keys() else np.nan
            ),
            axis=1,
        )
        tr[assigned_col] = new_col

        # adding some simple tracking columns to make later analysis easier
        tr["og_cell"] = tr.apply(
            lambda b: (
                str(int(self.assign_to_cell(ast.literal_eval(b["cell_ids"]))))
                if b["cell_ids"] is not None
                and self.assign_to_cell(ast.literal_eval(b["cell_ids"])) is not None
                else "None"
            ),
            axis=1,
        )
        tr["og_type"] = tr.apply(
            lambda b: (
                self.cell_to_type[b["og_cell"]]
                if b["og_cell"] in self.cell_to_type
                else "None"
            ),
            axis=1,
        )
        tr[f"{assigned_col}_type"] = tr.apply(
            lambda b: (
                self.cell_to_type[b[assigned_col]]
                if b[assigned_col] in self.cell_to_type
                else "None"
            ),
            axis=1,
        )

        # tally deltas while the table is still addressed by transcript id
        if save_delta_tallies:
            deltas = {}
            final_trs = {}
            for c in tr[assigned_col].astype("category").cat.categories:
                ex = tr[tr[assigned_col] == c].index
                final_trs[str(int(c))] = list(ex)

            for cell, trs in assigned_trs.items():
                if cell not in deltas.keys():
                    deltas[cell] = [0, 0, 0, 0]

                for trid in trs:
                    original = ast.literal_eval(tr.loc[[trid]]["cell_ids"].values[0])
                    original = self.assign_to_cell(original)

                    if original != str(cell):
                        deltas[cell][0] += 1
                        deltas[cell][2] += 1
                        if original is not None:
                            if original not in deltas.keys():
                                deltas[original] = [0, 1, 1, 0]
                            else:
                                deltas[original][1] += 1
                                deltas[original][2] += 1

                deltas[cell][3] = len(final_trs[cell])

            for cell in list(set(final_trs.keys()) - set(deltas.keys())):
                deltas[cell] = [0, 0, 0, len(final_trs[cell])]

            # one table per FOV, so pooled workers each save their own element
            cells = sorted(deltas)
            tally = ad.AnnData(
                np.array([deltas[c] for c in cells], dtype=float).reshape(len(cells), 4)
            )
            tally.obs_names = [str(c) for c in cells]
            tally.var_names = ["gained", "lost", "changed", "final"]
            tally.obs["cell_id"] = cells
            self.save_table(f"{f}_deltas_{assigned_col}", tally)

        # last bit of cleanup before we save it:
        tr = tr.loc[:, ~tr.columns.str.contains("^Unnamed")]
        self.set_transcripts(f, tr.reset_index())

        # this is too big to keep in memory if we're a part of a pool
        # that's running everything. So, delete if we are in a pool.
        if multiprocessing.current_process().daemon:
            return
        else:
            return assigned_trs, seg_is_default

    def __func_wrapper__(self, args):
        # needed to use imap; want to use imap to have tqdm progress bar
        # pass target function as first arg, the rest get passed through
        return args[0](*args[1:])

    # Start methods to try, in order. "fork" is deliberately not among them: by
    # the time a pool is created the parent has been reading (and often writing)
    # the SpatialData's zarr store through dask, and a forked child inherits that
    # machinery's locks without the threads that would release them -- the pool
    # then hangs, which is exactly what happens if you convert a dataset and run
    # blur_all_fovs in the same session.
    _START_METHODS = ("forkserver", "spawn")

    def _mp_context(self):
        for method in self._START_METHODS:
            try:
                return multiprocessing.get_context(method)
            except ValueError:  # not available on this platform
                continue
        return multiprocessing.get_context()

    def _pool(self, processes):
        """A worker pool whose processes are each bound to this assigner.

        The assigner is handed over once, when a worker starts, instead of being
        pickled alongside every task -- which matters now that it owns a whole
        SpatialData. A zarr-backed object is not copied at all: __getstate__
        drops it and each worker re-opens the store on first use.

        Because the workers do not fork, the assigner has to be picklable: keep
        `decay_func` a module-level function rather than a lambda if you plan to
        run with `pool_size > 1`.
        """
        return self._mp_context().Pool(
            processes=processes, initializer=_init_worker, initargs=(self,)
        )

    def _check_parallel_allowed(self):
        """Raise unless this object can be worked on by several processes.

        A worker returns its results by saving its own FOV's element back to the
        store; with nowhere to save, the work would be done and then lost when
        the process exits. There are two ways to be in that position, and they
        need different fixes.
        """
        if self._sdata_path is None:
            raise ValueError(
                "This SpatialData has never been saved, so worker processes "
                "would have nowhere to write their results and every FOV's "
                "output would be lost. Save it first -- "
                "`sdata.write('/path/to/store.zarr')`, or convert it with "
                "`save_loc=` -- and build the SoftAssigner from the saved "
                "object. To work in memory instead, set pool_size=1."
            )
        if not self.sdata.is_backed():
            raise ValueError(
                f"This SpatialData came from {self._sdata_path!r} but is not "
                "currently backed by it, so worker processes could not save "
                "their results. Re-open the store "
                f"(`spatialdata.read_zarr({self._sdata_path!r})`) and build the "
                "SoftAssigner from that object. To work in memory instead, set "
                "pool_size=1."
            )

    def _run_over_fovs(self, tasks, total, processes):
        """Run `(func, *args)` tasks, in a worker pool or serially, with a pbar.

        `processes <= 1` skips multiprocessing entirely, which keeps single-
        threaded runs usable from a plain script (a pool's workers do not fork,
        so they re-import __main__ and need the usual
        `if __name__ == "__main__":` guard around your script).

        Workers pass their results back by saving their own FOV's element to the
        store, so a parallel run requires one: see `_check_parallel_allowed`.
        Afterwards the store is re-read, since the elements the workers rewrote
        are newer than the ones held here.
        """
        if processes > 1:
            self._check_parallel_allowed()

        with tqdm(total=total) as pbar:
            if processes <= 1:
                for task in tasks:
                    task[0](self, *task[1:])
                    pbar.update(1)
                return
            with closing(self._pool(processes)) as pool:
                for _ in pool.imap_unordered(_run_worker, tasks):
                    pbar.update(1)
        self._reload_sdata()

    def evaluate_all_overlapping_regions(
        self,
        sel_fovs=None,
        gene_col_name="gene",
        min_thresh=None,
        default_thresh=5,
        only_tagged_cells=None,
        use_conf_trs=False,
        use_other_cells=False,
        use_mse_score=False,
        assigned_col="assignment",
        omit_blanks=False,
        auto_assign_single_target=False,
        save_delta_tallies=False,
        overwrite=False,
    ):
        """
        Runner for evaluate_overlapping_regions_single
        Runs said method in parallel on multiple fovs.
        """
        if sel_fovs is None:
            sel_fovs = self.get_complete_fovs()

        # jank workaround because multiprocessing seems to leak ram.
        # divide our list of fovs into sub-lists with max_fov_pool elements
        # then run multiprocessing sequentially on each of these.
        max_fov_pool = 100
        fov_pool = []
        offset = int(len(sel_fovs) % max_fov_pool != 0)
        for i in range(len(sel_fovs) // max_fov_pool + offset):
            end = min(len(sel_fovs), (i + 1) * max_fov_pool)
            fov_pool.append(sel_fovs[i * max_fov_pool : end])

        self.logger.info(f"Pooling fovs for parallel processing, as:\n{fov_pool}")

        for subset_fovs in fov_pool:
            self.logger.info(f"[{datetime.now()}] starting sub-pool: {subset_fovs}")
            self._run_over_fovs(
                zip(
                    repeat(SoftAssigner.evaluate_overlapping_regions_single_fov),
                    subset_fovs,
                    repeat(gene_col_name),
                    repeat(min_thresh),
                    repeat(default_thresh),
                    repeat(only_tagged_cells),
                    repeat(use_conf_trs),
                    repeat(use_other_cells),
                    repeat(use_mse_score),
                    repeat(assigned_col),
                    repeat(omit_blanks),
                    repeat(auto_assign_single_target),
                    repeat(save_delta_tallies),
                    repeat(True),
                    repeat(overwrite),
                ),
                len(subset_fovs),
                min(self.pool_size, len(subset_fovs)),
            )

    def combined_changed_transcripts(
        self, assigned_col="assignment", gene_col_name="gene"
    ):
        """Every transcript whose cell type changed under `assigned_col`.

        Returns one dataframe covering all FOVs. This is a cross-FOV summary of
        transcripts rather than of cells, so it is handed back rather than stored
        as a table -- write it wherever you need it.
        """
        wanted = [
            gene_col_name,
            "og_cell",
            "og_type",
            assigned_col,
            f"{assigned_col}_type",
        ]

        sel_trs = []
        for f in self.get_complete_fovs():
            tr = self.get_transcripts(f).set_index("index")
            if any(col not in tr.columns for col in wanted):
                continue

            # narrow first, and drop to plain objects on the way: the gene column
            # comes back from parquet as a Categorical, which refuses fillna with
            # a value outside its categories
            tr = tr[wanted].astype(object).fillna("None")

            # keep transcripts whose type changed (and are not both null)
            tr = tr[tr[f"{assigned_col}_type"] != tr["og_type"]]

            # the FOV is implicit in which element the rows came from
            tr.insert(1, "fov", f)
            sel_trs.append(tr)

        if not sel_trs:
            return pd.DataFrame(columns=wanted[:1] + ["fov"] + wanted[1:])
        return pd.concat(sel_trs)
