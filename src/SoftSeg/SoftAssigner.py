import ast
import json
import logging
import multiprocessing
import random
import warnings
from collections import Counter
from contextlib import closing
from datetime import datetime
from itertools import repeat
from collections.abc import Mapping
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


# The scoring matrix's QC bounds have no default: what counts as too few
# transcripts depends on the panel, so a number that suits one dataset throws
# away most of another. They belong to the dataset, and are carried in its
# attrs -- see `SoftAssigner.qc_bounds`. This marks an argument nobody passed,
# which is not the same as one passed as None (that turns the filter off).
UNSET = object()


# How every per-transcript assignment column says "no cell here" -- see
# `SoftAssigner.as_label_column` for why this spelling.
MISSING = pd.NA
MISSING_DTYPE = "string"

# Everything that has meant the same thing in a column written before that:
# "None" is what og_cell/og_type/{col}_type held, "none" is what a column that
# had been through `SupportFuncs.ParamSweeper._normalize_missing` held, and the
# rest are how a null can arrive back as text. `normalize_labels` collapses them.
MISSING_SPELLINGS = ("None", "none", "nan", "NaN", "NA", "<NA>", "")


def parse_cell_ids(value):
    """Read a stored ``{cell_id: score}`` map back into a dict.

    The maps are written as JSON, which parses an order of magnitude faster than
    a python repr does -- `ast.literal_eval` compiles every string it is handed,
    and this is called once per transcript. Stores written before the switch hold
    a repr instead, so those fall back to `literal_eval`; the attempt costs
    nothing on the JSON path.
    """
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return ast.literal_eval(value)


def dump_cell_ids(assigned):
    """Serialise a ``{cell_id: score}`` map for storage.

    Built directly rather than through ``json.dumps``: the maps are always
    ``{str: float}``, so the encoder's generality costs about twice what writing
    the text out does, once per transcript.
    """
    return "{" + ",".join(f'"{k}":{v}' for k, v in assigned.items()) + "}"


def sigm(x, x0=0, k=0.05):
    return 1 / (1 + np.e ** (-1 * k * (x - x0)))


def bounding_circle(geom):
    """A circle containing the whole geometry, as ``(cx, cy, radius)``.

    Centred on the bounding box, with the radius reaching the outermost vertex.
    Reading the vertices is the one costly part of screening points, so this is
    computed once per polygon and handed to `signed_distance` from then on.
    """
    minx, miny, maxx, maxy = geom.bounds
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    coords = shapely.get_coordinates(geom)
    radius = float(np.sqrt(((coords - (cx, cy)) ** 2).sum(axis=1)).max())
    return cx, cy, radius


def possibly_within(geom, xs, ys, max_dist, circle=None):
    """Mask of the points that could be within ``max_dist`` of ``geom``.

    Conservative: a point it excludes is provably farther than ``max_dist``, so
    excluding it changes no result. It is allowed to keep points that turn out
    to be far -- the exact measurement settles those.

    Two containers are tested, and a point has to be near both:

    * the geometry's bounding box. The geometry sits inside it, so nothing can
      be nearer than the box is.
    * a circle around the geometry's vertices. Every edge is inside it, so no
      edge can be nearer than the circle is. This is the tighter of the two for
      a roughly round cell, where the box is loose at the corners.

    Both are compared as squared distances, which keeps a square root out of the
    inner loop.
    """
    minx, miny, maxx, maxy = geom.bounds
    dx = np.maximum(np.maximum(minx - xs, xs - maxx), 0.0)
    dy = np.maximum(np.maximum(miny - ys, ys - maxy), 0.0)
    near_box = (dx * dx + dy * dy) <= max_dist * max_dist

    cx, cy, radius = circle if circle is not None else bounding_circle(geom)
    reach = radius + max_dist
    ex, ey = xs - cx, ys - cy
    return near_box & ((ex * ex + ey * ey) <= reach * reach)


def signed_distance(geom, xs, ys, max_dist=None, circle=None):
    """Signed distance from points to a polygon.

    Positive inside the geometry, negative outside, magnitude being the distance
    to the nearest edge. Vectorised over ``xs``/``ys`` (1D arrays). A
    ``MultiPolygon`` -- a cell whose slice has several disjoint pieces -- is
    measured as a whole, against the nearest edge of any of its parts.

    ``max_dist`` is the distance beyond which the caller stops caring. Given it,
    points that :func:`possibly_within` rules out never reach the measurement,
    which is the expensive part -- an exact distance costs about fifty times what
    the sign does. They come back as ``-inf``, which compares as "too far"
    wherever the result is used. Pass ``circle`` from :func:`bounding_circle` to
    screen without re-reading the geometry's vertices every call.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    if max_dist is None:
        dist = shapely.distance(geom.boundary, shapely.points(xs, ys))
        return np.where(shapely.contains_xy(geom, xs, ys), dist, -dist)

    out = np.full(xs.shape, -np.inf)
    near = np.flatnonzero(possibly_within(geom, xs, ys, max_dist, circle=circle))
    if len(near) == 0:
        return out

    qx, qy = xs[near], ys[near]
    dist = shapely.distance(geom.boundary, shapely.points(qx, qy))
    out[near] = np.where(shapely.contains_xy(geom, qx, qy), dist, -dist)
    return out


class SoftAssigner:
    def __init__(
        self,
        sdata,
        pool_size=1,
        conf_thresh=0.7,
        decay_func=None,
        save_to_disk=True,
        qc_bounds=None,
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

        save_to_disk: whether results are written to the store as they are
           produced. True by default. Set it False to try something out without
           touching what is on disk -- every result still lands in the sdata in
           memory, and is lost when the object goes, unless `save_element` is
           called for it later with the flag back on. Note that a pooled run
           needs it: a worker returns its results by saving its own element.

        qc_bounds: the QC the scoring matrix is built under, as any of
           `{"min_counts", "max_counts", "min_genes", "min_cells"}`. Recorded
           in the sdata, so it travels with the dataset and a later assigner over
           the same store scores it the same way -- see `qc_bounds` and
           `filter_for_scoring`. Left as None the stored bounds stand, and a
           store that has none filters nothing.

        Passing a zarr-backed `sdata` is required when `pool_size > 1`: workers
        re-open the store themselves and save their own FOV's element, which is
        the only way their results get back to the parent.
        """
        self._set_sdata(sdata)
        self._cell_to_type = None  # filled by get_scoring_matrix, or read back
        self.save_to_disk = save_to_disk
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

        if qc_bounds is not None:
            self.set_qc_bounds(qc_bounds)

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
            SpatialDataHelpers.quiet_ome_zarr()
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

        Does nothing either when this assigner was built with
        `save_to_disk=False`, which turns writing off wholesale.

        Rewriting an element in place is a delete followed by a write: zarr
        refuses to overwrite a path it is currently backing an element from. The
        replacement must therefore already be fully in memory (built through
        `.compute()`, as `set_transcripts` does) before this is called, or the
        delete is refused to avoid pulling the store out from under it. The pair
        is not atomic -- an interruption between the two leaves the element only
        in memory, and re-running the step rewrites it.
        """
        if not self.save_to_disk:
            self.logger.debug(
                f"[{datetime.now()}] save_to_disk is off; keeping {name} in memory."
            )
            return False

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

    def set_transcripts(self, fov, df, save=True):
        """Write a transcript table back into the FOV's points element.

        fov: fov number, naming the element to replace.
        df: the table to store, as `get_transcripts` returns it.

        `save=False` updates the element in memory but leaves the store alone,
        for a caller making many successive edits -- a parameter sweep, say --
        that would otherwise rewrite the whole element each time. A later
        `save_element` on the same name persists them.

        The element is replaced by one parsed from `df` -- carrying over the
        coordinate columns, the feature key and every coordinate transformation
        the old one had -- and then saved, so the new columns land in the
        element's parquet in the store. `df` must be an in-memory dataframe (what
        `get_transcripts` returns); the element it replaces is what backs the
        parquet being rewritten, so nothing may still be reading from it.

        Columns holding python objects, such as the `{cell_id: score}` dicts in
        `cell_ids`, have to be stored as their `str()` repr -- parquet has no
        type for them -- and are read back with `parse_cell_ids`.
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
        if save:
            self.save_element(key)

    def save_table(self, name, adata, save=True):
        """Put an AnnData into the sdata as a table, saved unless deferred.

        name (string): the table's element name.
        adata (AnnData): what to store under it.
        save (bool): if True (default) the table is written to the store.
        """
        self.sdata.tables[name] = adata
        if save:
            self.save_element(name)
        return adata

    def require_blurred(self, fov):
        """Raise unless this FOV has been through `blur_fov`.

        Everything downstream reads the `cell_ids` column, and without it the
        failure would otherwise surface as a bare KeyError partway through
        whatever was already running.
        """
        if not self.has_column(fov, "cell_ids"):
            raise KeyError(
                f"fov_{fov:0>4} has no 'cell_ids' column, so it has not been "
                "through the blurring step yet. Run "
                f"`blur_fov({fov!r}, min_size, max_dist)` for this FOV, or "
                "`blur_all_fovs(min_size, max_dist)` for every outstanding one, "
                "before assigning transcripts."
            )

    def region_table_names(self, assigned_col):
        """The two tables `evaluate_overlapping_regions_single_fov` writes."""
        return (f"assigned_trs_{assigned_col}", f"seg_is_default_{assigned_col}")

    def region_rows(self, fov, assigned_trs, seg_is_default):
        """One FOV's resolved assignments and region verdicts, as table rows.

        `assigned_trs` becomes one row per (cell, transcript) pair and
        `seg_is_default` one row per ambiguous region, each tagged with the FOV
        they came from. Kept separate from writing them so that a pooled run can
        hand these back from the worker and merge them in the parent, rather than
        having every worker rewrite the same shared element.
        """
        cells, ids = [], []
        for cell, transcripts in assigned_trs.items():
            cells.extend([str(cell)] * len(transcripts))
            ids.extend(float(t) for t in transcripts)
        trs = pd.DataFrame(
            {
                "fov": pd.Series([str(fov)] * len(cells), dtype=object),
                "cell_id": pd.Series(cells, dtype=object),
                "transcript_id": pd.Series(ids, dtype=float),
            }
        )

        regions = list(seg_is_default)
        seg = pd.DataFrame(
            {
                "fov": pd.Series([str(fov)] * len(regions), dtype=object),
                "region": pd.Series(
                    [",".join(str(c) for c in r) for r in regions], dtype=object
                ),
                "is_default": pd.Series(
                    [bool(seg_is_default[r]) for r in regions], dtype=bool
                ),
            }
        )
        return trs, seg

    def _merge_region_rows(self, name, frames, overwrite):
        """A table's existing rows with these FOVs' rows merged in.

        Rows for a FOV already in the table are replaced when `overwrite` is set,
        and left alone otherwise -- in which case that FOV's incoming rows are
        dropped. Returns None when nothing is left to write, so an existing table
        is not rewritten identically.
        """
        existing = self.sdata.tables.get(name)
        old = None if existing is None else existing.obs.reset_index(drop=True)

        # nothing to record and nothing recorded: an empty element for data that
        # does not exist is just noise. An existing table is still rewritten, so
        # `overwrite` can clear a FOV's rows.
        if existing is None and not any(len(rows) for rows in frames.values()):
            return None

        keep = dict(frames)
        if old is not None and len(old):
            present = set(old["fov"])
            if overwrite:
                old = old[~old["fov"].isin({str(f) for f in keep})]
            else:
                skipped = [f for f in keep if str(f) in present]
                for f in skipped:
                    del keep[f]
                if skipped:
                    self.logger.info(
                        f"[{datetime.now()}] {name}: keeping the rows already "
                        f"there for {skipped}; pass overwrite to replace them."
                    )
        if not keep:
            return None

        parts = [old] if old is not None and len(old) else []
        parts.extend(keep.values())
        obs = pd.concat(parts, ignore_index=True)
        obs.index = [str(i) for i in range(len(obs))]
        return ad.AnnData(np.zeros((len(obs), 0)), obs=obs)

    def save_region_tables(self, rows, assigned_col, overwrite=True, save=True):
        """Store how an assignment was reached, as two tables for the whole run.

        The assignment column on the points says where each transcript ended up;
        these say how it got there. One pair of tables per `assigned_col`, each
        holding every FOV's rows and carrying a `fov` column, so a whole run
        reads as a single table. Both keep their content in `obs`, since neither
        is a measurement over variables.

        rows (dict): `{fov: (trs_rows, seg_rows)}`, as `region_rows` builds them.
        overwrite (bool): whether rows already present for one of these FOVs are
            replaced. False leaves them, and drops the incoming rows.

        returns: the names written, which is empty if every FOV was left alone.
        """
        written = []
        for name, i in zip(self.region_table_names(assigned_col), (0, 1)):
            merged = self._merge_region_rows(
                name, {f: r[i] for f, r in rows.items()}, overwrite
            )
            if merged is None:
                continue
            self.save_table(name, merged, save=save)
            written.append(name)
        return written

    def as_label_column(self, values, index):
        """One of the assignment columns, with one spelling for "not assigned".

        `assigned_col`, `og_cell`, `og_type` and `f"{assigned_col}_type"` all
        name a cell or a cell type per transcript, and all four have to say "no
        cell here" the same way -- a column that spelt it differently to its
        neighbours used to reach `int()` in `SupportFuncs.ParamSweeper` and raise
        "invalid literal for int(): 'None'".

        The spelling is `MISSING` (`pd.NA`) in a `MISSING_DTYPE` ("string")
        column. That is what parquet gives back for any absent value, whether it
        went in as None, NaN or pd.NA, so writing it means the column reads back
        as it was written rather than changing type on the first reload. It also
        costs a bit in the validity bitmap rather than a stored value, unlike the
        sentinel string "None" this used to write -- a small saving in practice,
        since parquet dictionary-encodes a repeated sentinel anyway.

        values: the column's contents, or None for an all-absent column.
        index: the transcript table's index, for an all-absent column.
        """
        if values is None:
            return pd.Series(MISSING, index=index, dtype=MISSING_DTYPE)
        if isinstance(values, pd.Series):
            # already carries the table's index; converting in place avoids
            # realigning it against a copy of itself
            return values.astype(MISSING_DTYPE)
        return pd.Series(values, index=index, dtype=MISSING_DTYPE)

    @staticmethod
    def normalize_labels(df, column):
        """Rewrite one assignment column's absent values as `MISSING`, in place.

        For a table written before `as_label_column` settled the spelling, where
        "no cell here" could be any of `MISSING_SPELLINGS` as well as a null.
        Afterwards the column matches what this class writes now: `MISSING_DTYPE`
        throughout, with `MISSING` wherever a value is absent. A column already
        in that form is left as it is.

        A cell id that reads back as a float (12.0 from an older store, say) is
        written as "12" rather than "12.0", since the downstream consumers of
        these columns put them through `int()`.

        df (DataFrame): the table to edit, e.g. from `get_transcripts`.
        column (string): which of its columns to rewrite.

        returns: the same dataframe, for chaining.
        """
        values = df[column].astype(object)
        absent = pd.isna(values).to_numpy(dtype=bool)
        absent |= values.isin(MISSING_SPELLINGS).to_numpy(dtype=bool)

        # a whole-numbered float is a cell id that lost its string on the way
        # through some other format, not a distinct label
        numeric = [
            i
            for i, v in enumerate(values)
            if isinstance(v, (float, np.floating)) and not pd.isna(v) and v.is_integer()
        ]
        if numeric:
            values.iloc[numeric] = [str(int(values.iloc[i])) for i in numeric]

        values[absent] = MISSING
        df[column] = values.astype(MISSING_DTYPE)
        return df

    CELL_TYPE_TABLE = "cell_to_type"

    @property
    def cell_to_type(self):
        """Which cell type each cell was given -- `{cell id: cell type}`.

        `get_scoring_matrix` builds this off the cell-by-gene table's cell type
        column, and stores it in the sdata as the `CELL_TYPE_TABLE` table. An
        assigner that has not run that method reads it back from there, so a
        store that has been scored once can be picked up again -- to evaluate
        more FOVs, say -- without rebuilding the scoring matrix first.

        Note that only this mapping is stored, not `score_mat`: resolving an
        overlapping region needs both, so reading this back lets you see and use
        the cell types, but scoring still wants `get_scoring_matrix`.
        """
        if self._cell_to_type is None:
            self._cell_to_type = self.load_cell_to_type()
        return self._cell_to_type

    @cell_to_type.setter
    def cell_to_type(self, mapping):
        self._cell_to_type = mapping

    def save_cell_to_type(self, save=True):
        """Store the cell-to-type mapping in the sdata as a table."""
        cells = list(self._cell_to_type)
        table = ad.AnnData(np.zeros((len(cells), 0)))
        table.obs_names = [str(c) for c in cells]
        table.obs["cell_id"] = pd.Series(
            [str(c) for c in cells], index=table.obs_names, dtype=object
        )
        table.obs["cell_type"] = pd.Series(
            [str(self._cell_to_type[c]) for c in cells],
            index=table.obs_names, dtype=object,
        )
        return self.save_table(self.CELL_TYPE_TABLE, table, save=save)

    def load_cell_to_type(self):
        """Read the cell-to-type mapping back out of the sdata."""
        if self.CELL_TYPE_TABLE not in self.sdata.tables:
            raise KeyError(
                f"No cell types available: this assigner has not run "
                "`get_scoring_matrix`, and the sdata has no "
                f"{self.CELL_TYPE_TABLE!r} table left by one that did. Run "
                "`get_scoring_matrix()` first."
            )
        obs = self.sdata.tables[self.CELL_TYPE_TABLE].obs
        return dict(zip(obs["cell_id"], obs["cell_type"]))

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

    BLUR_ATTRS = "blur"

    def blur_params(self):
        """The blur parameters recorded in the sdata, or None if there are none.

        `blur_all_fovs` writes them, and reads them back to tell whether the
        `cell_ids` columns already in the store were produced the way it is being
        asked to produce them.
        """
        recorded = (getattr(self.sdata, "attrs", None) or {}).get(self.BLUR_ATTRS)
        return dict(recorded) if isinstance(recorded, Mapping) else None

    def set_blur_params(self, params, save=True):
        """Record the blur parameters in the sdata, or clear them with None.

        `blur_all_fovs` clears the key before it starts and writes it again once
        every FOV is done, so a run that dies partway through leaves no key --
        and a store with no key is one whose `cell_ids` columns nothing vouches
        for.
        """
        attrs = dict(getattr(self.sdata, "attrs", None) or {})
        if params is None:
            attrs.pop(self.BLUR_ATTRS, None)
        else:
            attrs[self.BLUR_ATTRS] = dict(params)
        self.sdata.attrs = attrs
        if save and self.save_to_disk and self.sdata.is_backed():
            self.sdata.write_attrs()
        return attrs.get(self.BLUR_ATTRS)

    QC_ATTRS = "qc_cells"
    QC_KEYS = ("min_counts", "max_counts", "min_genes", "min_cells")

    def qc_bounds(self):
        """The scoring-matrix QC recorded in the sdata, or None if there is none.

        These belong to the dataset rather than to the package: what counts as
        too few transcripts for a cell depends on the panel, so the bounds that
        suit a 500-gene MERFISH run would discard every cell of a simulation
        whose cells carry thousands. Recording them here means a dataset is
        scored the same way every time without anyone having to remember the
        numbers.
        """
        recorded = (getattr(self.sdata, "attrs", None) or {}).get(self.QC_ATTRS)
        return dict(recorded) if isinstance(recorded, Mapping) else None

    def set_qc_bounds(self, bounds, save=True):
        """Record the scoring-matrix QC in the sdata, or clear it with None.

        bounds (dict): any of `QC_KEYS`; anything absent is left unbounded.
        """
        if bounds is not None:
            unknown = set(bounds) - set(self.QC_KEYS)
            if unknown:
                raise ValueError(
                    f"Unknown QC bound(s) {sorted(unknown)}; "
                    f"expected any of {list(self.QC_KEYS)}."
                )

        attrs = dict(getattr(self.sdata, "attrs", None) or {})
        if bounds is None:
            attrs.pop(self.QC_ATTRS, None)
        else:
            attrs[self.QC_ATTRS] = dict(bounds)
        self.sdata.attrs = attrs
        if save and self.save_to_disk and self.sdata.is_backed():
            self.sdata.write_attrs()
        return attrs.get(self.QC_ATTRS)

    def resolve_qc_bounds(self, **passed):
        """What QC to apply: whatever was passed, else what the sdata records.

        A bound given as None is an explicit "no filter" and overrides the
        stored one; a bound left unpassed defers.
        """
        stored = self.qc_bounds() or {}
        return {
            key: stored.get(key) if value is UNSET else value
            for key, value in passed.items()
        }

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

    def get_cell_shapes(self, fov, min_size=None, max_size=None, persist=True):
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
        Cells outside the `min_size`/`max_size` voxel-count range are omitted --
        that filtering is applied to what is returned, never to what is stored,
        so the kept shapes stay valid for any size range.

        Newly contoured shapes are added to the sdata, and saved as well when
        `persist` is set. Contouring a FOV's mask is by far the most expensive
        part of blurring it, and the result depends only on the segmentation, so
        it is worth keeping: a second blur of the same FOV -- another point in a
        parameter sweep, say -- reads the shapes back instead of rebuilding them.
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
            built = SpatialDataHelpers.masks_to_shapes(
                self.sdata,
                labels_keys=[self._labels_key(fov)],
                pool_size=1,
                inplace=False,
            )
            # Keep them. They go into the object either way, so even a run that
            # is not writing anything only contours each FOV once.
            for name, gdf in built.shapes.items():
                self.sdata.shapes[name] = gdf
                if persist:
                    self.save_element(name)
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
        """Returns a list of all fov indicies that still need to be run by this object.

        The complement of `get_complete_fovs`: a FOV is outstanding while its
        transcript table has no `column`.
        """
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
            assigned = parse_cell_ids(row["cell_ids"])
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
                for c, v in parse_cell_ids(row["cell_ids"]).items():
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
            assigned = parse_cell_ids(row["cell_ids"])
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
        rows_x,
        x,
        y,
        hits,
        max_dist,
        projected_dist=None,
        circle=None,
    ):
        """Score one cell's polygon against the transcripts on its z-slice.

        rows: indices into x/y/cell_ids of the transcripts on this slice, in
           ascending x order.
        rows_x: their x values, so the band of candidates can be found without
           touching the rest.
        circle: this polygon's bounding circle, from `bounding_circle`, so the
           far transcripts can be screened out without re-reading its vertices.
        projected_dist: if not None, this polygon comes from a *neighbouring*
           z-slice and every distance is lengthened by this out-of-plane offset.

        Transcripts further than max_dist outside the polygon are left alone;
        for the rest, `(cell_id, transcript indices, scores)` is appended to
        `hits`. The per-transcript dicts are built from those in one pass at the
        end of `blur_fov`, which keeps this out of python entirely -- a dict
        write per cell per transcript is the one part of the blur that cannot be
        done on arrays.
        """
        if len(rows) == 0:
            return

        # Only transcripts inside the polygon's bounding box (grown by max_dist)
        # can possibly be within max_dist of it. Projecting onto a neighbouring
        # slice only ever lengthens a distance, so this stays conservative.
        #
        # The x range is taken by binary search rather than by testing every
        # transcript on the slice: a cell covers a small part of a FOV, so the
        # band is a small part of the column, and the alternative costs one pass
        # over every transcript for every cell.
        minx, miny, maxx, maxy = geom.bounds
        lo = np.searchsorted(rows_x, minx - max_dist, side="left")
        hi = np.searchsorted(rows_x, maxx + max_dist, side="right")
        if lo >= hi:
            return

        rows = rows[lo:hi]
        ys = y[rows]
        near = (ys >= miny - max_dist) & (ys <= maxy + max_dist)
        rows = rows[near]
        if len(rows) == 0:
            return

        # the caller discards anything further than max_dist, so say so and let
        # the cheap bounds throw those out before the distance is measured.
        # A projected distance is only ever longer than the in-plane one, so the
        # same cutoff stays conservative there too.
        dist = signed_distance(
            geom, x[rows], y[rows], max_dist=max_dist, circle=circle
        )

        if projected_dist is not None:
            # the transcript is one slice away from the contour, so its true
            # separation is the hypotenuse of (in-plane distance, slice spacing).
            # A transcript inside the projected contour is simply the slice
            # spacing away from it.
            hypot = -1 * np.sqrt(dist**2 + projected_dist**2)
            dist = np.where(dist < 0, hypot, np.maximum(hypot, -1 * projected_dist))

        keep = dist > -1 * max_dist
        if not keep.any():
            return

        # Applied one element at a time, and to python floats, deliberately.
        # Handing `decay_func` the whole array instead is about 6% quicker over
        # a FOV, but `np.e ** array` and `np.e ** float` can differ by an ulp,
        # and `assign_to_cell` compares the result against 0.5 and against a
        # share of the transcript's total -- so a value on either boundary could
        # land differently. Not worth trading an exact answer for.
        kept = dist[keep]
        scores = np.fromiter(
            (float(self.decay_func(float(d))) for d in kept), float, len(kept)
        )

        hits.append((cell_id, rows[keep], scores))

    def blur_fov(
        self,
        f,
        min_size=25,
        max_dist=20,
        dist_between_slices=None,
        project_neighbor_slices=True,
        disable_tqdm=False,
        save=True,
        overwrite=False,
        check_params=True,
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
            of a slice with a valid contour will project that contour with this
            added distance. If None (default), the spacing is read off the sdata's
            coordinate system (`get_dist_between_slices`).
        project_neighbor_slices: set False to disable that projection entirely.
        disable_tqdm: if True, this method will not print output or create its
            own pbar entities.
        save: if True (default) the transcript table is written back to the
            store. False leaves the edit in memory for a later `save_element`.
        overwrite: whether to redo a FOV that already has a `cell_ids` column.
            False (default) skips it, since blurring the same mask with the same
            parameters gives the same answer. `blur_all_fovs` sets this when the
            parameters it was called with differ from the ones recorded in the
            store.
        check_params: warn when this FOV is about to be blurred with parameters
            that do not match `attrs["blur"]`, or when nothing is recorded there
            -- either way the FOV ends up disagreeing with the rest of the
            dataset. `blur_all_fovs` turns this off, since it owns that key and
            clears it for the duration of a run.

        writes: the `cell_ids` column of the FOV's points element, as JSON text,
            saved unless `save` is False. Any `og_cell`/`og_type` left by an
            earlier run is dropped, since both follow from `cell_ids`.
        returns: dataframe of transcript information (None during a pooled run)
        """
        if self.has_column(f, "cell_ids") and not overwrite:
            self.logger.info(
                f"[{datetime.now()}] skipping fov_{f:0>4}, cell_ids already present"
            )
            return None

        if check_params:
            params = {
                "min_size": min_size,
                "max_dist": max_dist,
                "dist_between_slices": dist_between_slices,
                "project_neighbor_slices": bool(project_neighbor_slices),
            }
            recorded = self.blur_params()
            if recorded is None:
                warnings.warn(
                    f"Blurring fov_{f:0>4} with no dataset-wide blur parameters "
                    f"recorded in attrs[{self.BLUR_ATTRS!r}]. Nothing says the "
                    "other FOVs were blurred this way; run blur_all_fovs to do "
                    "the whole dataset consistently and record how.",
                    stacklevel=2,
                )
            elif recorded != params:
                warnings.warn(
                    f"Blurring fov_{f:0>4} with {params}, which differs from the "
                    f"{recorded} recorded for this dataset. This FOV will not "
                    "match the others.",
                    stacklevel=2,
                )

        tr = self.get_transcripts(f)

        if len(tr) > 0:
            self.logger.info(f"[{datetime.now()}] starting fov_{f:0>4}...")

            # override if if it's alraedy there'
            if "cell_ids" in tr.keys():
                tr.drop(["cell_ids"], axis=1, inplace=True)

            # og_cell/og_type are derived from cell_ids, so whatever a previous
            # run left behind no longer describes this segmentation
            stale = [c for c in ("og_cell", "og_type") if c in tr.columns]
            if stale:
                tr.drop(stale, axis=1, inplace=True)

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
            x = tr["x"].to_numpy(dtype=float)
            y = tr["y"].to_numpy(dtype=float)

            # (cell id, transcript indices, scores) per cell per slice, turned
            # into the per-transcript dicts once the whole FOV is scored
            hits = []

            # each slice's transcripts, ordered by x, so a cell can take the
            # band it covers by binary search instead of testing all of them
            tr_by_slice = {}
            for z in elig_z:
                rows = np.flatnonzero(slice_of_tr == z)
                rows = rows[np.argsort(x[rows], kind="stable")]
                tr_by_slice[z] = (rows, x[rows])

            # each cell as {z: polygon}, i.e. its 2D slices aggregated back
            # together along the z axis
            shapes_by_cell = self.get_cell_shapes(f, min_size=min_size, persist=save)

            # One bounding circle per polygon, computed here and reused for
            # every slice it is scored against. Deriving it inside the scoring
            # loop instead would re-read the same vertices on every call, which
            # costs more than the screening saves.
            circles = {
                (cell_id, z): bounding_circle(geom)
                for cell_id, by_z in shapes_by_cell.items()
                for z, geom in by_z.items()
            }

            if not disable_tqdm:
                pbar = tqdm(total=len(shapes_by_cell))

            # ascending cell id, so each transcript's dict of candidates is
            # ordered by cell id -- `assign_to_cell` returns the first key that
            # clears its thresholds, so the order is part of the result
            for cell_id in sorted(shapes_by_cell):
                by_z = shapes_by_cell[cell_id]

                for z in elig_z:
                    rows, rows_x = tr_by_slice[z]
                    if len(rows) == 0:
                        continue

                    geom = by_z.get(z)
                    if geom is not None:
                        self._blur_one_shape(
                            cell_id, geom, rows, rows_x, x, y, hits, max_dist,
                            circle=circles[(cell_id, z)],
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
                        rows_x,
                        x,
                        y,
                        hits,
                        max_dist,
                        projected_dist=dist_between_slices,
                        circle=circles[(cell_id, neighbor)],
                    )

                if not disable_tqdm:
                    pbar.update(1)

            del shapes_by_cell

            if not disable_tqdm:
                pbar.close()

            # Assemble the per-transcript maps. `hits` is in ascending cell id
            # order, so each transcript's candidates come out in that order too
            # -- which `assign_to_cell` depends on, since it takes the first key
            # clearing its thresholds. Iterating python lists rather than numpy
            # arrays keeps this loop off numpy scalars, which are slow to unbox.
            cell_ids = [{} for _ in range(len(tr))]
            for cell_id, rows_hit, scores in hits:
                name = str(cell_id)
                for j, v in zip(rows_hit.tolist(), scores.tolist()):
                    cell_ids[j][name] = v
            del hits

            # parquet has no type for a python dict, so the per-transcript
            # {cell_id: score} maps are stored as JSON text
            tr["cell_ids"] = [dump_cell_ids(d) for d in cell_ids]
            self.set_transcripts(f, tr, save=save)

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
        min_size=25,
        max_dist=20,
        dist_between_slices=None,
        project_neighbor_slices=True,
        sel_fovs=None,
        save=True,
    ):
        """Run the first step of soft-segmentation, where masks are blurred and
        multiple float values assigned to each transcript, corresponding to which
        cells they may be members of and their relative likelihoods.

        This method will run on all eligible FOVs, using the multiprocessing pool.

        pool_size: the number of threads to be used.
        min_size: minimum size for eligible masks.
        max_dist: the maximum distance between a transcript and a mask to be considered eligible.
            Both default to the values a full vizgen MERFISH dataset was
            processed and validated with; they are in voxels and pixels, so a
            dataset at a different resolution will want its own.
        dist_between_slices: as in blur_fov; read per-FOV off the sdata's
            coordinate system when left as None.
        project_neighbor_slices: as in blur_fov.
        sel_fovs: FOVs to run on. Defaults to all of them; which ones actually
            need doing is decided per FOV, below.

        The parameters are recorded under `attrs["blur"]`, and compared against
        what is already there. Matching them means the FOVs already carrying a
        `cell_ids` column were blurred this same way, so those are left alone and
        only the outstanding ones run -- which is what makes an interrupted run
        resumable. Differing from them invalidates every existing column, so all
        the FOVs are redone.

        Each worker saves its own element, which is how its results get back
        here, so this always writes; `blur_fov`'s deferred `save` is not
        available through this runner.

        writes: the `cell_ids` column of every FOV's points element, and
            `attrs["blur"]`, saved to the store.
        """
        if sel_fovs is None:
            sel_fovs = self.get_all_fovs()

        params = {
            "min_size": min_size,
            "max_dist": max_dist,
            "dist_between_slices": dist_between_slices,
            "project_neighbor_slices": bool(project_neighbor_slices),
        }
        recorded = self.blur_params()
        overwrite = recorded != params
        if overwrite and recorded is not None:
            self.logger.info(
                f"Blur parameters changed from {recorded} to {params}; "
                "re-blurring every FOV."
            )

        # Cleared for the duration: until every FOV is done there is no single
        # set of parameters that describes the dataset, and a run that does not
        # finish should not leave one behind.
        self.set_blur_params(None, save=save)

        self._run_over_fovs(
            zip(
                repeat(SoftAssigner.blur_fov),
                sel_fovs,
                repeat(min_size),
                repeat(max_dist),
                repeat(dist_between_slices),
                repeat(project_neighbor_slices),
                repeat(True),
                repeat(save),
                repeat(overwrite),
                repeat(False),  # this method owns attrs["blur"]; see check_params
            ),
            len(sel_fovs),
            # a worker hands its results back by saving them, so with saving off
            # there is nothing to run in parallel
            self.pool_size if save else 1,
        )

        self.set_blur_params(params, save=save)

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

        Reads every completed FOV's `cell_ids`, then sweeps 60 candidate
        thresholds over the lot, counting at each how many transcripts get
        assigned and how many cells end up with more than 10 of them.

        show_plots (bool): if True, draw the assigned-transcripts against
            unique-cells curve and its first two derivatives.

        Sets `self.conf_thresh` to the conservative threshold, and
        `self.num_transcripts` to the number of transcripts it read.

        returns:
        "aggresive": defines threshold as elbow point of assigned transcripts vs
            multiply-assigned transcripts plot.
        "conservative": defines threshold as lowest value where no transcript is
            multiply-assigned.
        """
        sel_fovs = self.get_complete_fovs()

        # Read every transcript's {cell: score} map, walking the column as an
        # array rather than with `iterrows` -- the per-row Series that yields is
        # most of the cost and none of the work.
        per_fov = []
        width = 1
        for f in sel_fovs:
            self.logger.info(f"Reading in fov {f}\t{datetime.now()}")
            tr = self.get_transcripts(f)
            parsed = [parse_cell_ids(raw) for raw in tr["cell_ids"].to_numpy()]
            width = max(width, max((len(a) for a in parsed), default=1))
            per_fov.append(parsed)
            self.logger.info(
                f"Just completed fov_{f:0>4}, {len(parsed)} transcripts"
            )

        # One row per transcript, padded to the widest candidate list. The size
        # comes from the data: the previous version reserved 10 million rows when
        # `num_transcripts` had not been set, and every row it left unwritten
        # stayed uninitialised memory that the sweep below then scored as though
        # it were a transcript.
        n_tr = sum(len(p) for p in per_fov)
        cells = np.full((n_tr, width), -1.0)
        cutoff = np.zeros((n_tr, width))  # padding scores 0: never selectable
        r = 0
        for parsed in per_fov:
            for assigned in parsed:
                for ind, (k, v) in enumerate(assigned.items()):
                    cells[r, ind] = float(k)
                    cutoff[r, ind] = float(v)
                r += 1
        del per_fov
        self.num_transcripts = n_tr

        assigned_tr = []
        cell_dist = []
        num = 60  # resolution for estimate
        rang = np.linspace(0.01, 0.60, num=num)

        # now some basic filtering like we do when clustering
        # remove cells that do not reach a specific transcript count
        min_count = 10

        # `assign_to_cell` takes the first candidate whose score is at least 0.5
        # and more than `thresh` of the transcript's total. Both tests are the
        # same for every threshold bar the comparison itself, so they are done
        # over the whole array at once rather than a dict per transcript.
        totals = cutoff.sum(axis=1)
        scored = totals > 0
        share = cutoff / np.where(scored, totals, 1.0)[:, None]
        strong = cutoff >= 0.5
        rows = np.arange(n_tr)

        for thresh in rang:
            ok = strong & (share > thresh)
            hit = ok.any(axis=1) & scored
            chosen = cells[rows, ok.argmax(axis=1)][hit]

            assigned_tr.append(int(hit.sum()))

            uniq, counts = np.unique(chosen, return_counts=True)
            cell_dist.append(int(((counts > min_count) & (uniq != 0.0)).sum()))

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

    def generate_cxg_table(
        self,
        gene_col_name="gene",
        min_thresh=None,
        assigned_col=None,
        sel_fovs=None,
        table_name=None,
    ):
        """
        Builds the cell-by-gene table from all completed analyses.

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
                # Absent is MISSING, so the nulls drop out here. "None" is how
                # a column written before `as_label_column` spelt it, and an
                # older store still has to be readable -- left in, the sentinel
                # reaches the int() below and raises.
                labels = tr[assigned_col]
                keep = labels.notna().to_numpy(dtype=bool)
                keep &= (labels != "None").fillna(False).to_numpy(dtype=bool)
                tr = tr[keep]
                tallies = Counter(list(zip(tr[gene_col_name], tr[assigned_col])))
            else:
                cell_col = tr.cell_ids.apply(
                    lambda x: self.assign_to_cell(parse_cell_ids(x), min_thresh)
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

        # AnnData wants string observation names and a float matrix, and converts
        # both itself with a warning apiece. Doing it here instead -- after the
        # sort above, which wants the numeric ids -- gives the same table quietly.
        cxg_df.index = cxg_df.index.astype(str)
        adata = ad.AnnData(cxg_df.astype(float))

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

    # obs columns `generate_cxg_table` writes itself: bookkeeping and geometry,
    # never cell types, so they are never candidates when `cats` is inferred
    CXG_OBS_COLUMNS = ("fov", "size", "x_coords", "y_coords", "z_coords")

    def default_cxg_table(self, table_name=None):
        """The cell-by-gene table `generate_cxg_table` left in the sdata.

        Prefers `table_name`, then "cxg" -- what `generate_cxg_table` writes by
        default -- then a single "cxg*" table if that is the only candidate. A
        store holding several is ambiguous and says so rather than guessing.
        """
        tables = list(self.sdata.tables)
        if table_name is not None:
            if table_name not in tables:
                raise KeyError(f"No table {table_name!r} in this sdata; have {tables}")
            return self.sdata.tables[table_name]

        if "cxg" in tables:
            return self.sdata.tables["cxg"]

        candidates = [t for t in tables if t.startswith("cxg")]
        if len(candidates) == 1:
            return self.sdata.tables[candidates[0]]
        raise KeyError(
            "Could not tell which table to score against. Run "
            "`generate_cxg_table()` first, or name one with `table_name=`; "
            f"this sdata has {tables}."
            if not candidates
            else "Several cell-by-gene tables are present "
            f"({candidates}); name the one to use with `table_name=`."
        )

    def infer_cats(self, adata, column=None):
        """Read a table's cell type categories, as `cats` for scoring.

        `column` names the obs column holding the types, and its labels are
        taken from what is actually in it. Without one, the column is looked for
        too: a cell type column holds labels rather than numbers or identifiers,
        so it is of categorical or object dtype, is not one of the columns
        `generate_cxg_table` writes itself, and is not unique per cell. Exactly
        one such column is needed to be unambiguous; otherwise the caller is
        asked which to use.
        """
        def labels_of(col):
            values = adata.obs[col]
            return sorted(
                str(v) for v in pd.unique(values.dropna()) if str(v) != "nan"
            )

        if column is not None:
            if column not in adata.obs.columns:
                raise KeyError(
                    f"No obs column {column!r} on this table; have "
                    f"{list(adata.obs.columns)}."
                )
            labels = labels_of(column)
            if not labels:
                raise ValueError(
                    f"The obs column {column!r} holds no cell type labels."
                )
            self.logger.info(f"Scoring against obs column {column!r}: {labels}")
            return {column: labels}

        candidates = {}
        for col in adata.obs.columns:
            if col in self.CXG_OBS_COLUMNS:
                continue
            values = adata.obs[col]
            if not (isinstance(values.dtype, pd.CategoricalDtype)
                    or values.dtype == object):
                continue
            labels = labels_of(col)
            if not labels or len(labels) >= len(adata):
                continue  # empty, or an identifier rather than a type
            candidates[col] = labels

        if len(candidates) == 1:
            col, labels = next(iter(candidates.items()))
            self.logger.info(f"Scoring against obs column {col!r}: {labels}")
            return {col: labels}
        raise ValueError(
            "Could not tell which obs column holds the cell types"
            + (f" -- candidates are {sorted(candidates)}" if candidates else "")
            + ". Name one by passing cats='column name', or give them in full "
            "as cats={'column name': ['cell type', ...]}. The table needs to "
            "have been annotated, e.g. by CellTypeAssigner."
        )

    @staticmethod
    def _per_cell_and_gene(X):
        """Totals a QC filter needs: per cell, then per gene.

        Handles a sparse X as well as a dense one -- both answer `.sum(axis=)`,
        though a sparse one answers with a matrix, hence the ravel.
        """
        counts = np.asarray(X.sum(axis=1)).ravel()
        genes_per_cell = np.asarray((X > 0).sum(axis=1)).ravel()
        cells_per_gene = np.asarray((X > 0).sum(axis=0)).ravel()
        return counts, genes_per_cell, cells_per_gene

    def filter_for_scoring(
        self, adata, min_counts=None, max_counts=None, min_genes=None,
        min_cells=None,
    ):
        """Drop the cells and genes that should not shape the scoring matrix.

        The same QC the pre-package notebooks ran over the cell-by-gene table
        before scoring it, by way of `CellTypeAssigner.filter_cells` and
        `filter_genes` -- a cell carrying too few transcripts has a profile that
        is mostly noise, and a gene seen in too few cells says little about any
        type. Applied here instead so the scoring matrix can be filtered without
        a separate filtered copy of the table.

        min_counts / max_counts (int): keep cells whose total transcript count is
            within these bounds.
        min_genes (int): keep cells expressing at least this many distinct genes.
        min_cells (int): keep genes expressed in at least this many cells, after
            the cell filters have been applied.

        Note what a dropped cell means downstream: it is absent from
        `cell_to_type`, so it is untyped, and an overlapping region containing it
        is skipped unless `use_other_cells=True`. A dropped gene simply stops
        contributing to any score, as a blank does.

        returns: the filtered AnnData, or the original when nothing was asked
            for. Never modifies the table it was given.
        """
        wanted = (min_counts, max_counts, min_genes, min_cells)
        if all(v is None for v in wanted):
            return adata

        counts, genes_per_cell, _ = self._per_cell_and_gene(adata.X)
        keep_cells = np.ones(adata.n_obs, dtype=bool)
        if min_counts is not None:
            keep_cells &= counts >= min_counts
        if max_counts is not None:
            keep_cells &= counts <= max_counts
        if min_genes is not None:
            keep_cells &= genes_per_cell >= min_genes

        if not keep_cells.any():
            raise ValueError(
                f"Every one of the {adata.n_obs} cells was filtered out "
                f"(min_counts={min_counts}, max_counts={max_counts}, "
                f"min_genes={min_genes}). Transcript counts run "
                f"{counts.min():g}-{counts.max():g}."
            )

        out = adata[keep_cells]
        if min_cells is not None:
            # counted after the cell filter, so a gene is judged on the cells
            # that are actually being scored
            _, _, cells_per_gene = self._per_cell_and_gene(out.X)
            keep_genes = cells_per_gene >= min_cells
            if not keep_genes.any():
                raise ValueError(
                    f"Every one of the {adata.n_vars} genes was filtered out "
                    f"(min_cells={min_cells}). A gene appears in at most "
                    f"{cells_per_gene.max():g} of the {int(keep_cells.sum())} "
                    "cells that survived the cell filters."
                )
            out = out[:, keep_genes]

        self.logger.info(
            f"[{datetime.now()}] scoring matrix QC: kept "
            f"{out.n_obs} of {adata.n_obs} cells and "
            f"{out.n_vars} of {adata.n_vars} genes."
        )
        # a view shares the parent's n_obs in places, so hand back a real object
        return out.copy()

    def get_scoring_matrix(
        self, adata=None, cats=None, normed=False, table_name=None, save=True,
        min_counts=UNSET, max_counts=UNSET, min_genes=UNSET, min_cells=UNSET,
    ):
        """
        Need to run this before evaluate_overlapping_regions

        cats is the possible cell types as they are described in adata. Give it
        either way round:
          {"column name": ["cell type", "cell type"], ...}  -- in full
          "column name"                                     -- the column that
              holds the types, whose labels are then read off it
          None                                              -- find that column
              too, when the table has exactly one that could hold cell types

        adata likewise defaults to the cell-by-gene table `generate_cxg_table`
        stored in the sdata (see `default_cxg_table`), so with nothing passed at
        all this scores the table the pipeline just produced. `table_name` names
        which table to read when a store holds more than one.

        The cell-to-type mapping this builds is stored in the sdata as the
        `CELL_TYPE_TABLE` table, so `cell_to_type` can be read back by a later
        assigner over the same store. `save=False` keeps that in memory only.

        min_counts / max_counts / min_genes / min_cells: QC bounds applied to a
        copy of the table before it is scored, as `filter_for_scoring` describes
        -- cells carrying too few (or too many) transcripts or expressing too few
        genes are dropped, then genes seen in too few of the remaining cells. The
        stored table is not touched. A filtered-out cell ends up untyped, so an
        overlapping region containing it is skipped unless `use_other_cells`.
        Left unpassed, each defers to what the sdata records (`qc_bounds`, set at
        init or with `set_qc_bounds`); passing None turns that filter off for
        this call. A store with no recorded bounds filters nothing.
        """
        if adata is None:
            adata = self.default_cxg_table(table_name)
        adata = self.filter_for_scoring(
            adata,
            **self.resolve_qc_bounds(
                min_counts=min_counts, max_counts=max_counts,
                min_genes=min_genes, min_cells=min_cells,
            ),
        )
        if cats is None or isinstance(cats, str):
            cats = self.infer_cats(adata, column=cats)

        all_avg = np.average(adata.X, axis=0)

        # Each type's rows are taken by index off the matrix. Subsetting the
        # AnnData and deep-copying the result, as this used to, duplicates the
        # whole expression matrix once per cell type to compute one mean of it.
        X = adata.X
        avgs = {}
        counts = {}
        for k, v in cats.items():
            column = adata.obs[k].to_numpy()
            for cat in v:
                rows = np.flatnonzero(column == cat)
                if not len(rows):
                    # the QC bounds can empty a type out; averaging nothing
                    # would put nan across its whole row
                    self.logger.info(
                        f"[{datetime.now()}] no cells left of type {cat!r} after "
                        "filtering, so it is left out of the scoring matrix."
                    )
                    continue
                avgs[cat] = np.average(X[rows], axis=0)
                counts[cat] = len(rows)

        avgs["other"] = all_avg
        counts["other"] = len(adata)

        df_avg = pd.DataFrame.from_dict(avgs, orient="index", columns=adata.var.index)
        if normed:
            norm_total = sum(counts.values())
            df_normed = df_avg.multiply(
                [(norm_total - n) / norm_total * 100 for n in counts.values()],
                axis=0,
            )

            df_comp = df_normed
        else:
            df_comp = df_avg

        self.score_mat = df_comp.fillna(0).to_dict()

        # generating cell_to_type dict, while we have our hands on cats
        # Written a type at a time rather than a cell at a time: `iterrows`
        # builds a Series per cell, which on a real table is most of the cost.
        # A cell matching more than one entry still ends up with the last one,
        # since the loops run in the same order.
        self.cell_to_type = {}  # key: cell ID, value: cell type
        self.tr_to_gene = {}  # key: transcript ID, value: gene name
        names = adata.obs_names.to_numpy()
        for key_type, cell_types in cats.items():
            column = adata.obs[key_type].to_numpy()
            for cell_type in cell_types:
                if cell_type not in avgs:
                    continue  # filtered out entirely, so it scores nothing
                for name in names[column == cell_type]:
                    self.cell_to_type[name] = cell_type

        # kept in the sdata so a later assigner over the same store can read the
        # cell types back rather than having to be handed them again
        self.save_cell_to_type(save=save)

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
            min_assigned = parse_cell_ids(
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

    def scan_overlapping_regions(
        self,
        f,
        gene_col_name="gene",
        min_thresh=0.7,
        only_tagged_cells=None,
        use_other_cells=False,
        omit_blanks=True,
        auto_assign_single_target=False,
        disable_tqdm=False,
    ):
        """Sort a FOV's transcripts into confident and ambiguous assignments.

        The first half of `evaluate_overlapping_regions_single_fov`, and the
        expensive one: it reads the transcript table and walks every row. Every
        transcript is either given to a single cell outright or filed under the
        tuple of cells that could claim it, and tuples not worth resolving --
        one cell type between them, or an untyped member -- are marked to keep
        their original assignment.

        Nothing here depends on `default_thresh`: that only enters when a
        candidate assignment is weighed against the original segmentation's
        score. A sweep over `default_thresh` can therefore scan once and resolve
        many times, which is what `SupportFuncs.ParamSweeper` does.

        f (int): current FOV
        gene_col_name (string): the gene/feature column of the transcript table.
        min_thresh (float): threshold to be used for confident transcript identification.
            Defaults to the value the validated vizgen run used; it is a share of
            a transcript's total score, so it does not depend on the panel.
        only_tagged_cells (list): If provided, only cells in this list will be considered for re-evaluation.
        use_other_cells (bool): if True, a tuple whose cells share one type is
            still resolved, against a notional "other" cell; if False it keeps
            its original assignment.
        omit_blanks (bool): if True (the default, as in the validated vizgen run),
            all blanks will be categorically ignored. Note that you
            may not want to ignore blanks in cell assignment if you want to quantify
            any kind of spatial error.
        auto_assign_single_target (bool): if True, unconfident transcripts that have a single
            eligible target cell will automatically be assigned to that cell.
        disable_tqdm (bool): if True, this method will not print output or create
            its own pbar entities.

        returns: the state `resolve_overlapping_regions` consumes -- the
           transcript table, the confident and ambiguous groupings, and the
           tuples already settled -- or None when the FOV has no ambiguous
           region to resolve. Also populates `self.tr_to_gene`, which
           `score_tr_assignment` reads.
        """
        self.require_blurred(f)

        # transcript id is the row label here: the scoring addresses transcripts
        # by it, and it is restored to a column before saving
        tr = self.get_transcripts(f).set_index("index")

        self.logger.info(
            f"[{datetime.now()}] scanning transcripts for fov_{f:0>4}..."
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
        # value: True if using original masks, False if using novel mask.
        # Always a plain bool -- `False` has to read as false, which a
        # single-element list would not.

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

        # Walk the columns as arrays rather than with `iterrows`. Every row
        # `iterrows` yields is a fresh Series, which pandas builds by copying the
        # frame's metadata -- on a transcript table that is most of the loop's
        # cost, and none of it is work this needs.
        tr_index = tr.index.to_numpy()
        tr_cell_ids = tr["cell_ids"].to_numpy()
        tr_genes = tr[gene_col_name].to_numpy()

        for i in range(len(tr_index)):
            if pbar is not None:
                pbar.update(1)

            index = tr_index[i]
            gene = tr_genes[i]

            # omit blanks
            if omit_blanks and "lank" in gene:
                continue

            assigned = parse_cell_ids(tr_cell_ids[i])

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
                self.tr_to_gene[index] = gene
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
                        seg_is_default[unconf_tup] = True
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
                            # float, as everywhere else in assigned_trs: the
                            # ids that come back out of the (id, score) arrays
                            # are floats, so keeping one type here means the
                            # lookup built from this dict has one too
                            if unconf_tup[0] in assigned_trs.keys():
                                assigned_trs[unconf_tup[0]].append(float(index))
                            else:
                                assigned_trs[unconf_tup[0]] = [float(index)]
                            self.logger.info(
                                f"Assigning transcript {index} to unconfident but unambiguous cell assignment {unconf_tup}."
                            )
                            seg_is_default[unconf_tup] = True
                            continue
                        else:
                            skip_cached.append(unconf_tup)
                            seg_is_default[unconf_tup] = True
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

        return {
            "fov": f,
            "tr": tr,
            "conf_trs": conf_trs,
            "unconf_trs": unconf_trs,
            "seg_is_default": seg_is_default,
            "assigned_trs": assigned_trs,
            "max_len": max_len,
        }

    def resolve_overlapping_regions(
        self,
        state,
        default_thresh=5,
        use_conf_trs=False,
        use_other_cells=False,
        use_mse_score=False,
        disable_tqdm=False,
    ):
        """Choose each ambiguous region's best assignment, given a scan.

        The second half of `evaluate_overlapping_regions_single_fov`. For each
        ambiguous region a range of soft-score thresholds is swept to generate
        candidate hard assignments, each scored with `score_tr_assignment`
        against the matrix from `get_scoring_matrix`. Requires that matrix to
        have been built.

        `state` is left untouched, so one scan can be resolved at several
        `default_thresh` values: the candidate scores do not depend on it, only
        the margin a candidate has to clear before it displaces the original
        segmentation.

        state (dict): from `scan_overlapping_regions`.
        default_thresh (float): new segmentation must out-score original segmentation
            by a factor of this much in order to be considered "better".
        use_conf_trs (bool): If true, confident transcripts will be added to each cell
            when scoring transcript assignments. If false, only ambiguous transcripts will
            be used. NOTE: Setting this to True causes a non-trivial slowdown.
        use_other_cells (bool): if True, a one-cell region is compared against a
            notional "other" cell rather than kept as-is.
        use_mse_score (bool): if True, score an assignment by its squared error
            against the cell type's expected profile instead of by summed score.
        disable_tqdm (bool): if True, this method will not print output or create
            its own pbar entities.

        returns: (assigned_trs, seg_is_default)
           assigned_trs: dict of cell id -> list of transcript IDs
           seg_is_default: dict of region tuple -> True where the original
              segmentation was kept
        """
        f = state["fov"]
        tr = state["tr"]
        conf_trs = state["conf_trs"]
        unconf_trs = state["unconf_trs"]
        max_len = state["max_len"]
        # this method owns its copies, so the scan survives being resolved again
        assigned_trs = {k: list(v) for k, v in state["assigned_trs"].items()}
        seg_is_default = dict(state["seg_is_default"])

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
                        # a local copy: the scan's own dict has to survive being
                        # resolved again at another default_thresh
                        tr_by_cell = dict(tr_by_cell)
                        tr_by_cell["other"] = tr_by_cell[unconf_tup[0]]
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

                seg_is_default[unconf_tup] = bool(best_score == -1)

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

        if pbar is not None:
            pbar.close()

        # assign things that were left as default
        if not disable_tqdm:
            print("\tapplying default assignments...")
            pbar = tqdm(total=len(seg_is_default))
        else:
            pbar = None

        for unconf_tup, is_default in seg_is_default.items():
            if pbar is not None:
                pbar.update(1)

            if is_default:
                if unconf_tup in unconf_trs:
                    og_assignment = self.trs_at_default(unconf_trs[unconf_tup])
                    for cell, trs in og_assignment.items():
                        # "other" assignments were for one-way comparisons
                        if cell != "other":
                            if cell in assigned_trs:
                                assigned_trs[cell].extend(float(t) for t in trs)
                            else:
                                assigned_trs[cell] = [float(t) for t in trs]
                else:
                    print(f"{unconf_tup} does not have corresponding trs list")


        if pbar is not None:
            pbar.close()

        return assigned_trs, seg_is_default

    def evaluate_overlapping_regions_single_fov(
        self,
        f,
        gene_col_name="gene",
        min_thresh=0.7,
        default_thresh=5,
        only_tagged_cells=None,
        use_conf_trs=False,
        use_other_cells=False,
        use_mse_score=False,
        assigned_col="assignment",
        omit_blanks=True,
        auto_assign_single_target=False,
        save_delta_tallies=False,
        disable_tqdm=False,
        overwrite=False,
        save=True,
        recompute_og=False,
        scan_state=None,
        defer_region_tables=False,
    ):
        """
        Scores and re-assigns border region transcripts for a single FOV.

        Composes `scan_overlapping_regions` (sort the transcripts) and
        `resolve_overlapping_regions` (pick each ambiguous region's best
        assignment), then writes the result to the FOV's transcript table as
        `assigned_col`, alongside `og_cell`, `og_type` and
        `f"{assigned_col}_type"`. Pass `scan_state` to reuse a scan rather than
        repeating it.

        f (int): current FOV
        gene_col_name (string): the gene/feature column of the transcript table.
        default_thresh (float): new segmentation must out-score original segmentation
            by a factor of this much in order to be considered "better".
        min_thresh (float): threshold to be used for confident transcript identification.
            Defaults to the value the validated vizgen run used; it is a share of
            a transcript's total score, so it does not depend on the panel.
        only_tagged_cells (list): If provided, only cells in this list will be considered for re-evaluation.
        use_conf_trs (bool): If true, confident transcripts will be added to each cell
            when scoring transcript assignments. If false, only ambiguous transcripts will
            be used. NOTE: Setting this to True causes a non-trivial slowdown.
        assigned_col (string): The name of the column for the new assignment to be added to
            for a given FOV's transcript table.
        omit_blanks (bool): if True (the default, as in the validated vizgen run),
            all blanks will be categorically ignored. Note that you
            may not want to ignore blanks in cell assignment if you want to quantify
            any kind of spatial error.
        auto_assign_single_target (bool): if True, unconfident transcripts that have a single
            eligible target cell will automatically be assigned to that cell.
        disable_tqdm (bool): if True, this method will not print output or  create
            its own pbar entities.
        overwrite (bool): if True, this method will overwite assigned_col if it already exists.
        use_other_cells (bool): if True, a region whose cells share one type is
            still resolved, against a notional "other" cell.
        use_mse_score (bool): if True, score an assignment by its squared error
            against the cell type's expected profile instead of by summed score.
        save_delta_tallies (bool): if True, per-cell gained/lost/changed/final
            counts are stored as a `f"{fov}_deltas_{assigned_col}"` table.
        save (bool): if True (default) the transcript table is written back to
            the store. False leaves the edit in memory, for a caller making many
            successive assignments -- see `SupportFuncs.ParamSweeper`.
        recompute_og (bool): og_cell and og_type follow from `cell_ids` and the
            scoring matrix's cell types, not from either threshold, so an
            existing pair is reused. `blur_fov` drops them when it rewrites
            `cell_ids`; set this to rebuild them after changing the cell types
            under an already-used scoring matrix.
        scan_state (dict): a scan from `scan_overlapping_regions` to resolve,
            instead of scanning this FOV again. Must have been taken at the same
            `min_thresh`, which is what a scan depends on.
        defer_region_tables (bool): if True, this FOV's rows for the
            `assigned_trs`/`seg_is_default` tables are returned instead of being
            merged into them here. Those tables cover every FOV, so a runner
            merges them once rather than having each FOV rewrite the element --
            see `evaluate_all_overlapping_regions`.

        writes: `assigned_col`, `og_cell`, `og_type` and `f"{assigned_col}_type"`
           on the FOV's points element, and this FOV's rows of the
           `f"assigned_trs_{assigned_col}"` and
           `f"seg_is_default_{assigned_col}"` tables, which record how those
           assignments were reached. Saved unless `save` is False.
        """
        if self.has_column(f, assigned_col) and not overwrite:
            self.logger.info(
                f"[{datetime.now()}] skipping fov_{f:0>4}, {assigned_col} already present"
            )
            if not disable_tqdm:
                print(f"fov_({f:0>4}) already present, skipping")
            return

        if scan_state is None:
            scan_state = self.scan_overlapping_regions(
                f,
                gene_col_name=gene_col_name,
                min_thresh=min_thresh,
                only_tagged_cells=only_tagged_cells,
                use_other_cells=use_other_cells,
                omit_blanks=omit_blanks,
                auto_assign_single_target=auto_assign_single_target,
                disable_tqdm=disable_tqdm,
            )
        if scan_state is None:
            return

        tr = scan_state["tr"]
        conf_trs = scan_state["conf_trs"]
        assigned_trs, seg_is_default = self.resolve_overlapping_regions(
            scan_state,
            default_thresh=default_thresh,
            use_conf_trs=use_conf_trs,
            use_other_cells=use_other_cells,
            use_mse_score=use_mse_score,
            disable_tqdm=disable_tqdm,
        )

        table_rows = self.region_rows(f, assigned_trs, seg_is_default)
        if not defer_region_tables:
            self.save_region_tables(
                {f: table_rows}, assigned_col, overwrite=overwrite, save=save
            )

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
        # These four columns were row-wise `.apply(axis=1)` passes, which build a
        # Series per transcript. Three of them are pure lookups and vectorise; the
        # fourth has to run python per row, but over the one column it reads
        # rather than over whole rows.
        #
        # All four are `MISSING_DTYPE` ("string"), and a transcript with no cell
        # or no type is `MISSING` (`pd.NA`) in every one of them -- see
        # `as_label_column`.

        # transcript ID -> cell. The ids arrive as a mix of ints and floats
        # (confident assignments keep the index dtype, reassigned ones are cast
        # to float), so both sides are matched as floats.
        if inv_assignment:
            lookup = pd.Series(inv_assignment)
            lookup.index = lookup.index.astype(float)
            values = lookup.reindex(tr.index.to_numpy(dtype=float)).to_numpy()
        else:
            values = None
        tr[assigned_col] = self.as_label_column(values, tr.index)

        # Adding some simple tracking columns to make later analysis easier.
        # `og_cell` and `og_type` describe the segmentation this run started
        # from: they follow from `cell_ids` and the scoring matrix's cell types,
        # and not from either threshold. Re-running with different thresholds
        # therefore recomputes the same answer, so an existing pair is left
        # alone. `blur_fov` drops them when it rewrites `cell_ids`; pass
        # `recompute_og=True` after changing the cell types under a scoring
        # matrix that has already been used.
        if recompute_og or "og_cell" not in tr.columns:
            og_cell = []
            for raw in tr["cell_ids"].to_numpy():
                cell = None if raw is None else self.assign_to_cell(parse_cell_ids(raw))
                og_cell.append(MISSING if cell is None else str(int(cell)))
            tr["og_cell"] = self.as_label_column(og_cell, tr.index)
        else:
            self.logger.debug(
                f"fov_{f:0>4}: reusing the og_cell column already on the table."
            )

        if recompute_og or "og_type" not in tr.columns:
            tr["og_type"] = self.as_label_column(
                tr["og_cell"].map(self.cell_to_type), tr.index
            )

        tr[f"{assigned_col}_type"] = self.as_label_column(
            tr[assigned_col].map(self.cell_to_type), tr.index
        )

        # tally deltas while the table is still addressed by transcript id
        if save_delta_tallies:
            deltas = {}
            final_trs = {}
            # compared as a plain object array: `assigned_col` is nullable, and
            # `== c` on it yields NA wherever a transcript went unassigned, which
            # is not something a row mask accepts
            labels = tr[assigned_col].to_numpy(dtype=object, na_value=None)
            named = labels[pd.notna(labels)]
            for c in pd.unique(named):
                final_trs[str(int(c))] = list(tr.index[labels == c])

            for cell, trs in assigned_trs.items():
                if cell not in deltas.keys():
                    deltas[cell] = [0, 0, 0, 0]

                for trid in trs:
                    original = parse_cell_ids(tr.loc[[trid]]["cell_ids"].values[0])
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
            self.save_table(f"{f}_deltas_{assigned_col}", tally, save=save)

        # last bit of cleanup before we save it:
        tr = tr.loc[:, ~tr.columns.str.contains("^Unnamed")]
        self.set_transcripts(f, tr.reset_index(), save=save)

        # the rows are the compact form of these dicts, so a deferred run hands
        # those back rather than the dicts themselves
        if defer_region_tables:
            return (f, *table_rows)

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
        the process exits. There are three ways to be in that position, and they
        need different fixes.
        """
        if not self.save_to_disk:
            raise ValueError(
                "Cannot run in parallel with save_to_disk=False: a worker hands "
                "its results back by saving its own FOV's element, so with "
                "writing turned off every FOV's output would be lost. Build the "
                "SoftAssigner with save_to_disk=True, or set pool_size=1 to work "
                "in memory."
            )
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

    def _run_over_fovs(self, tasks, total, processes, collect=False):
        """Run `(func, *args)` tasks, in a worker pool or serially, with a pbar.

        `processes <= 1` skips multiprocessing entirely, which keeps single-
        threaded runs usable from a plain script (a pool's workers do not fork,
        so they re-import __main__ and need the usual
        `if __name__ == "__main__":` guard around your script).

        Workers pass their results back by saving their own FOV's element to the
        store, so a parallel run requires one: see `_check_parallel_allowed`.
        Afterwards the store is re-read, since the elements the workers rewrote
        are newer than the ones held here.

        `collect` gathers what each task returned, for the one result a worker
        cannot save itself: rows of a table shared by every FOV. It is off by
        default because most of these tasks hand back the FOV's whole transcript
        table, which there is no reason to keep once it has been saved.
        """
        if processes > 1:
            self._check_parallel_allowed()

        results = []
        with tqdm(total=total) as pbar:
            if processes <= 1:
                for task in tasks:
                    result = task[0](self, *task[1:])
                    if collect:
                        results.append(result)
                    pbar.update(1)
                return results
            with closing(self._pool(processes)) as pool:
                for result in pool.imap_unordered(_run_worker, tasks):
                    if collect:
                        results.append(result)
                    pbar.update(1)
        self._reload_sdata()
        return results

    def evaluate_all_overlapping_regions(
        self,
        sel_fovs=None,
        gene_col_name="gene",
        min_thresh=0.7,
        default_thresh=5,
        only_tagged_cells=None,
        use_conf_trs=False,
        use_other_cells=False,
        use_mse_score=False,
        assigned_col="assignment",
        omit_blanks=True,
        auto_assign_single_target=False,
        save_delta_tallies=False,
        overwrite=False,
        recompute_og=False,
    ):
        """
        Runner for evaluate_overlapping_regions_single_fov.
        Runs said method in parallel on multiple fovs.

        Every argument bar `sel_fovs` is passed straight through; see that
        method for what each does. Each worker scans and resolves its own FOV
        and saves its own element, so this always writes -- the deferred `save`
        that a sweep uses is not available here. The `assigned_trs` and
        `seg_is_default` tables cover every FOV at once, so they are merged and
        written here, after the workers have finished.

        sel_fovs (list): FOVs to run on. Defaults to every FOV whose transcript
           table has been through `blur_fov`.
        recompute_og (bool): rebuild og_cell/og_type rather than reusing an
           existing pair.
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

        results = []
        for subset_fovs in fov_pool:
            self.logger.info(f"[{datetime.now()}] starting sub-pool: {subset_fovs}")
            results.extend(self._run_over_fovs(
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
                    repeat(True),  # save: each worker persists its own element
                    repeat(recompute_og),
                    repeat(None),  # scan_state: each worker scans its own FOV
                    repeat(True),  # defer_region_tables: merged below, once
                ),
                len(subset_fovs),
                min(self.pool_size, len(subset_fovs)),
                collect=True,
            ))

        # the assigned_trs/seg_is_default tables span every FOV, so they are
        # written here rather than by each FOV: a worker holds its own copy of
        # the store, and several of them rewriting one element would leave only
        # whichever finished last. FOVs that were skipped return nothing.
        rows = {f: (trs, seg) for f, trs, seg in (r for r in results if r is not None)}
        if rows:
            self.save_region_tables(rows, assigned_col, overwrite=overwrite)

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
