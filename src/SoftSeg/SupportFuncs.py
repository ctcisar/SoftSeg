import ast
import collections
import fnmatch
import glob
import logging
import random
import re
import warnings
from pathlib import Path
from typing import Any, Mapping, Optional
from typing import Sequence as TypingSequence
from typing import Union

import anndata as ad
import dask.array as da
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import parse as _parse
import scanpy as sc
import skimage
import skimage.io
import spatialdata as sd
import tifffile
from spatialdata import SpatialData, read_zarr
from spatialdata.models import (Image2DModel, Image3DModel, Labels2DModel,
                                Labels3DModel, PointsModel, ShapesModel,
                                TableModel)
from spatialdata.transformations import (Affine, Identity, Sequence,
                                         Translation, get_transformation)
from tqdm.auto import tqdm

from SoftSeg.SoftAssigner import SoftAssigner, parse_cell_ids
from SoftSeg.SpatialDataHelpers import SpatialDataHelpers
from SoftSeg.warp_shapes import smooth_field_warp, warp_shapes


def kl_divergence(p, q, pcount=1):
    """
    Find the kl divergence between two anndata objects containing cell by gene counts, p and q.
    """
    # first make sure that p and q have identical lists of genes
    p = p[:, np.isin(p.var.index, q.var.index)]
    q = q[:, np.isin(q.var.index, p.var.index)]

    # next sort genes to make sure that all genes are in the same order; add pcounts
    p_x = np.nan_to_num(p[:, p.var_names.sort_values()].X) + (
        pcount * np.ones(np.shape(p.X))
    )
    q_x = np.nan_to_num(q[:, q.var_names.sort_values()].X) + (
        pcount * np.ones(np.shape(q.X))
    )

    # take average
    p_sum = np.sum(p_x, axis=0)
    q_sum = np.sum(q_x, axis=0)

    p_m = np.squeeze((p_sum / np.sum(p_sum)).tolist())
    q_m = np.squeeze((q_sum / np.sum(q_sum)).tolist())

    # calculate entropy
    total = 0
    for i in range(len(p_m)):
        total += p_m[i] * np.log(p_m[i] / q_m[i])

    return total


def kl_by_celltype(p, q, cats, pcount=1):
    """
    finds the kl divergence between two anndata objects, subsetting each of them based on cats.
    cats is expected to be dict of format: {label in obs: list of celltypes}
    """
    result = {}
    for label, cell_types in cats.items():
        for cell_type in cell_types:
            p_sub = p[p.obs[label] == cell_type]
            q_sub = q[q.obs[label] == cell_type]
            result[cell_type] = kl_divergence(p_sub, q_sub, pcount)
    return result


def plot_kl_divergence(
    scores, cell_types, legend, title=None, hlines=None, hlines_color=None
):
    for time in legend:
        plt.scatter(cell_types, [scores[time][cell_type] for cell_type in cell_types])
    if hlines is not None:
        for time in legend:
            plt.hlines(hlines[time], 0, len(cell_types), hlines_color[time])

    plt.xlabel("Cell Type")
    plt.tick_params("x", length=10, labelsize="small", labelrotation=90)
    plt.ylabel("KL-divergence")
    if title is not None:
        plt.title(title)

    plt.legend(legend)
    plt.show(block=False)


class NeighborPredictor:

    def __init__(self, adata, type_col="celltypes"):
        self.adata = adata
        self.type_col = type_col
        self.train_losses = []
        self.model = None
        self.model_stats = {}

    def clean_data(self):
        exclude = [name for name in self.adata.var_names if "lank" in name]
        self.adata = self.adata[:, ~self.adata.var_names.isin(exclude)]
        self.full_adata = self.adata.copy()
        self.adata = self.adata[self.adata.obs["fov"].notnull()]

    def find_neighbors(self, n_neighbors=6):
        # torch / scikit-learn are the `neighbors` optional dependency, imported
        # where they are used so the rest of SupportFuncs works without them
        from sklearn.neighbors import NearestNeighbors

        locs = pd.concat(
            [
                self.adata.obs["x_coords"],
                self.adata.obs["y_coords"],
                self.adata.obs["z_coords"],
            ],
            axis=1,
        )
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(
            locs
        )
        dists, indicies = nbrs.kneighbors(locs)

        neighbor_list = []
        for row in indicies:
            this_neighbors = []
            for neigh in row[1:]:
                this_neighbors.append(self.adata.obs.index[neigh])
            neighbor_list.append(this_neighbors)

        self.adata.obs["neighbors"] = neighbor_list

    def normalize_data(self):
        sc.pp.calculate_qc_metrics(self.adata, inplace=True)

        self.adata = self.adata[self.adata.obs["n_genes_by_counts"] > 25]
        sc.pp.normalize_total(self.adata, target_sum=1e-6)
        sc.pp.log1p(self.adata)

    def train_model(self, focal_types, neighbor_types, progress=True):
        import torch
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        # First, restrict the data we're looking at to the focal types
        adata = self.adata[
            self.adata.obs[self.type_col].str.contains(
                "|".join(focal_types), regex=True
            )
        ].copy()

        # Next, come up with our labels, ie whether a neighbor of type neighbor_types
        # is present in the nearest neighbors of a given cell
        has_neighbor = []
        for i, row in adata.obs.iterrows():
            this_neighbor = False
            for index in row["neighbors"]:
                if index in self.full_adata.obs.index:
                    ct = self.full_adata.obs.loc[[index]][self.type_col][0]
                    if ct in neighbor_types:
                        this_neighbor = True
                        break
            has_neighbor.append(float(this_neighbor))

        adata.obs["target_neighbor"] = has_neighbor

        # now, divy up our data into training and validation splits
        total_cells = len(adata)

        # define ratios here
        target = {
            "train": total_cells * 0.75,
            # "test": total_cells * 0.25,
            "valid": total_cells * 0.25,
        }

        cells = {
            "train": [],
            # "test": [],
            "valid": [],
        }

        remaining_fovs = list(adata.obs["fov"].cat.categories)
        for target_type in cells.keys():
            while (
                len(cells[target_type]) < target[target_type]
                and len(remaining_fovs) > 0
            ):
                sel_fov = random.choice(remaining_fovs)
                remaining_fovs.remove(sel_fov)

                new_cells = adata[adata.obs["fov"] == sel_fov].obs.index
                cells[target_type].extend(list(new_cells))

        loader_kwarg = {
            "batch_size": 64,
            "shuffle": True,
        }
        loaders = {}
        for category in cells.keys():
            data = torch.tensor(adata[cells[category]].X)
            labels = torch.tensor(adata[cells[category]].obs["target_neighbor"])
            dataset = TensorDataset(data, labels)
            loaders[category] = DataLoader(dataset, **loader_kwarg)

        # create model
        self.model = torch.nn.Linear(len(adata.var.index), 1)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        num_epochs = 300

        if progress:
            print("Training model...")
            pbar = tqdm(total=num_epochs)

        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (images, labels) in enumerate(loaders["train"], 0):
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()
                    self.model.cuda()
                else:
                    self.model.cpu()

                optimizer.zero_grad()

                outputs = self.model(images)
                loss = criterion(outputs.squeeze(), labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % 200 == 99:
                    running_loss = 0.0
            if progress:
                pbar.update(1)

            self.train_losses.append(running_loss / len(loaders["train"]))

        if progress:
            pbar.close()

        correct = 0
        total = 0

        self.model.eval()

        with torch.no_grad():
            for images, labels in loaders["valid"]:
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.model_stats = {
            "accuracy": correct / total,
            "null_prob": sum(has_neighbor) / len(has_neighbor),
            "focal_cell_tally": len(has_neighbor),
            "neighbor_cell_tally": sum(has_neighbor),
        }

        return self.model_stats

    def plot_loss_curve(self):
        plt.plot(self.train_losses, label="Training Loss")
        plt.title("Training Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def validate_model(
        self, null_prob=None, null_size=None, num_bootstrap=10_000, progress=True
    ):
        if null_prob is None:
            null_prob = self.model_stats["null_prob"]
        if null_size is None:
            null_size = self.model_stats["focal_cell_tally"]

        obs_means = []
        if progress:
            print("Running bootstrap simulation...")
            pbar = tqdm(total=num_bootstrap)

        for _ in range(num_bootstrap):
            success = 0
            for __ in range(null_size):
                if random.random() < null_prob:
                    success += 1
            obs_means.append(success / null_size)
            if progress:
                pbar.update(1)

        if progress:
            pbar.close()

        p_val = (
            sum([1 if x > self.model_stats["accuracy"] else 0 for x in obs_means])
            / num_bootstrap
        )
        valid_stats = {"p_val": p_val, "bootstrap": obs_means}
        self.model_stats.update(valid_stats)
        return valid_stats

    def plot_bootstrap(self):
        if "bootstrap" in self.model_stats:
            plt.hist(self.model_stats["bootstrap"])
            plt.title("Bootstrap Histogram")
            plt.xlabel("Observed Probability")
            plt.ylabel("Count")
            bot, top = plt.gca().get_ylim()
            plt.gca().set_ylim(
                min(bot, self.model_stats["accuracy"] - 0.1),
                max(top, self.model_stats["accuracy"] + 0.1),
            )
            plt.axvline(x=self.model_stats["accuracy"], color="red")
            plt.show()

    def run(self, focal_types, neighbor_types, n_neighbors=6, plot=True, progress=True):
        self.clean_data()
        self.find_neighbors(n_neighbors)
        self.normalize_data()
        self.train_model(
            focal_types=focal_types, neighbor_types=neighbor_types, progress=progress
        )
        self.validate_model(progress=progress)
        if plot:
            self.plot_loss_curve()
            self.plot_bootstrap()
        return self.model_stats


class SegImagePlotter:
    """Draw a FOV's segmentation over its raw image, one z-slice at a time.

    Reads from a SpatialData in the layout `SoftAssigner` uses -- the FOV's mask
    as `labels[f"{fov}_labels"]` and, when present, its microscopy image as
    `images[f"{fov}_image"]`. Rendering is left to `spatialdata_plot`: cell
    outlines are drawn by `render_labels` coloured through a table, rather than
    by contouring each mask by hand.
    """

    # Outline thickness handed to spatialdata_plot, as a `render_shapes` line
    # width. Each cell is its own polygon, so the ring closes at any width, and
    # a thin one keeps two neighbours' rings legible as two rather than running
    # them together into a band.
    CONTOUR_PX = 1

    # the three ways a cell can be drawn, and the colour each gets. Drawing
    # order too: "other" first, so a highlighted cell's ring is never overdrawn
    # by a neighbour's
    OUTLINE_COLORS = {
        "other": "#000000",
        "type": "#ffffff",  # matches highlight_type
        "highlight": "#ff0000",  # named in highlight_cells, or the target cell
    }

    # and the ways a transcript can be drawn, relative to the cells in focus --
    # the target cell and anything in highlight_cells. The microscopy underneath
    # is red/green/blue everywhere, so these are picked to sit off those axes and
    # are drawn over a dark halo besides
    TRANSCRIPT_COLORS = {
        "kept": "#00e5ff",  # in a cell in focus before and after
        "gained": "#00ff87",  # a run moved it into a cell in focus
        "lost": "#ff2bd6",  # a run moved it out of one
        "candidate": "#c8c8c8",  # a cell in focus was in the running, and lost
        "changed": "#ffd400",  # moved, but between cells that are not in focus
    }
    # what each of those means, said in the legend. "in focus" is the target
    # cell plus highlight_cells, which is what the groups are measured against
    TRANSCRIPT_LABELS = {
        "kept": "kept",
        "gained": "gained",
        "lost": "lost",
        "candidate": "considered, not taken",
        "changed": "moved between other cells",
    }
    # marker size, and the dark edge that keeps a marker legible over a bright
    # image. Sized for a cell's worth of transcripts -- a few dozen on a slice --
    # rather than for the handful a run moved, which would run together into
    # blobs at that density
    TRANSCRIPT_PX = 20
    TRANSCRIPT_EDGE_PX = 0.8
    TRANSCRIPT_EDGE_COLOR = "#000000"

    def __init__(self, sdata, fov=None, exposure_eq=False):
        """
        sdata: the dataset, or a path to a zarr store holding one.
        fov: the FOV to draw; may also be chosen later with `load_fov`.
        """
        SpatialDataHelpers.quiet_ome_zarr()
        self.sdata = sd.read_zarr(sdata) if isinstance(sdata, (str, Path)) else sdata
        self.fov = None
        self.exposure_eq = exposure_eq
        if fov is not None:
            self.load_fov(fov, exposure_eq=exposure_eq)

    # ----------------------------------------------------------------- #
    # the FOV's elements                                                 #
    # ----------------------------------------------------------------- #

    def load_fov(self, fov, exposure_eq=False):
        """Choose the FOV to draw.

        exposure_eq: if True, the image is equalised with skimage.exposure
           rather than being min-max scaled per z-slice.
        """
        if f"{fov}_labels" not in self.sdata.labels:
            raise KeyError(
                f"No '{fov}_labels' in this SpatialData; available: "
                f"{list(self.sdata.labels)}"
            )
        self.fov = fov
        self.exposure_eq = exposure_eq

    def _require_fov(self):
        if self.fov is None:
            raise Exception("load_fov must be called before performing this operation.")

    @property
    def labels_key(self):
        self._require_fov()
        return f"{self.fov}_labels"

    @property
    def image_key(self):
        self._require_fov()
        key = f"{self.fov}_image"
        return key if key in self.sdata.images else None

    @property
    def points_key(self):
        self._require_fov()
        key = f"{self.fov}_points"
        return key if key in self.sdata.points else None

    def get_transcripts(self):
        """The FOV's transcript table, indexed by transcript id, or None."""
        if self.points_key is None:
            return None
        el = self.sdata.points[self.points_key]
        df = el.compute() if hasattr(el, "compute") else pd.DataFrame(el)
        return df.set_index("index") if "index" in df.columns else df

    def changed_transcripts(self, assigned_col, og_col="og_cell"):
        """The transcripts a prior `evaluate_overlapping_regions` run moved.

        Every transcript whose `assigned_col` ended up somewhere other than the
        segmentation the run started from (`og_col`) -- which is exactly what
        that step changes, including a transcript it picked up out of no cell or
        dropped out of one.

        The two columns are compared as plain objects rather than as the
        nullable columns they are: `!=` between those propagates NA, which would
        drop a transcript that only one of them placed.

        returns: a dataframe indexed by transcript id, with the transcript's
            `x`, `y` and `z`, and the `from`/`to` cells.
        """
        df = self.get_transcripts()
        if df is None:
            raise KeyError(f"No '{self.fov}_points' in this SpatialData.")
        for column in (assigned_col, og_col):
            if column not in df.columns:
                raise KeyError(
                    f"No {column!r} column on {self.fov}_points; it has "
                    f"{[c for c in df.columns]}."
                )

        pair = df[[og_col, assigned_col]].copy()
        for column in (og_col, assigned_col):
            SoftAssigner.normalize_labels(pair, column)
        was = pair[og_col].to_numpy(dtype=object, na_value=None)
        now = pair[assigned_col].to_numpy(dtype=object, na_value=None)

        moved = df[was != now].copy()
        out = pd.DataFrame(index=moved.index)
        out["x"] = moved["x"].to_numpy(dtype=float)
        out["y"] = moved["y"].to_numpy(dtype=float)
        out["z"] = self._slice_of(moved)
        out["from"] = was[was != now]
        out["to"] = now[was != now]
        return out

    @staticmethod
    def _considered_by(df, wanted, cell_ids_col="cell_ids"):
        """Which transcripts had one of `wanted` among their candidate cells.

        The `cell_ids` maps are stored one JSON string per transcript, and there
        are hundreds of thousands of them, so parsing every one to answer this
        would dominate the plot. A substring test over the raw column narrows it
        first -- an id appears in the text of any map that holds it, whether the
        keys were written quoted or bare -- and only the survivors are parsed,
        which is what settles the over-matches the substring test lets through.
        """
        if cell_ids_col not in df.columns:
            return np.zeros(len(df), dtype=bool)

        raw = df[cell_ids_col]
        maybe = np.zeros(len(df), dtype=bool)
        for cell in wanted:
            maybe |= raw.str.contains(cell, regex=False, na=False).to_numpy(dtype=bool)

        out = np.zeros(len(df), dtype=bool)
        for i in np.flatnonzero(maybe):
            try:
                scores = parse_cell_ids(raw.iat[i])
            except (SyntaxError, ValueError):
                continue
            out[i] = any(str(k) in wanted for k in scores)
        return out

    def cell_transcripts(self, cells, assigned_col=None, og_col="og_cell",
                         candidates=True, cell_ids_col="cell_ids"):
        """Every transcript `cells` hold, plus every one they were in the running for.

        `changed_transcripts` answers "what did this run move?", which for a
        single cell is usually one or two transcripts -- the run only moves what
        was ambiguous. This answers "what was this cell's claim on the data?":
        the transcripts either assignment places in it, and the ones where it was
        a candidate and lost. Those two together are what a plot of a cell is
        for, since the ones it did not get are the reason it has the ones it did.

        A cell is a candidate for a transcript when it appears in that
        transcript's `cell_ids` map, which is the set of cells `blur_fov` scored
        it against and the set `evaluate_overlapping_regions` chose from.

        cells: one cell id or a list of them.
        assigned_col: the assignment to read as the "after" state. Left None,
           only `og_col` is read and every transcript is shown as staying put.
        og_col: the assignment to read as the "before" state.
        candidates: include the transcripts `cells` were only in the running for.
        cell_ids_col: the column holding the `{cell_id: score}` maps. Absent from
           the table, there are no candidates to find and only the assignment
           columns are read.

        returns: a frame in the shape `changed_transcripts` returns, plus a
            `considered` column flagging the ones a focus cell was in the running
            for, so it can be handed to `plot_subset(transcripts=...)` directly.
            The colours then separate them into the ones the cell kept, gained,
            lost, and was passed over for.
        """
        df = self.get_transcripts()
        if df is None:
            raise KeyError(f"No '{self.fov}_points' in this SpatialData.")
        if isinstance(cells, (str, int, np.integer)):
            cells = [cells]
        wanted = {str(c) for c in cells}

        if assigned_col is not None and assigned_col not in df.columns:
            raise KeyError(
                f"No {assigned_col!r} column on {self.fov}_points; it has "
                f"{[c for c in df.columns]}."
            )
        # og_col is a default rather than something the caller asked for, so a
        # store that never recorded it reads the one assignment it does have
        columns = [c for c in (og_col, assigned_col)
                   if c is not None and c in df.columns]
        if not columns:
            raise KeyError(
                f"Neither {og_col!r} nor an assignment column to read is on "
                f"{self.fov}_points; it has {[c for c in df.columns]}."
            )

        pair = df[columns].copy()
        for column in columns:
            SoftAssigner.normalize_labels(pair, column)
        # as plain objects, for the same reason changed_transcripts does it: a
        # comparison against the nullable columns propagates NA
        sides = {
            column: pair[column].to_numpy(dtype=object, na_value=None)
            for column in columns
        }

        keep = np.zeros(len(df), dtype=bool)
        for values in sides.values():
            keep |= np.array(
                [v is not None and str(v) in wanted for v in values], dtype=bool
            )

        considered = self._considered_by(df, wanted, cell_ids_col) if candidates \
            else np.zeros(len(df), dtype=bool)
        keep |= considered

        held = df[keep]
        out = pd.DataFrame(index=held.index)
        out["x"] = held["x"].to_numpy(dtype=float)
        out["y"] = held["y"].to_numpy(dtype=float)
        out["z"] = self._slice_of(held)
        out["considered"] = considered[keep]
        before = og_col if og_col in sides else columns[0]
        after = assigned_col if assigned_col in sides else before
        out["from"] = sides[before][keep]
        out["to"] = sides[after][keep]
        return out

    def valid_slices(self):
        """The z-slice indices this FOV's segmentation covers.

        The same list `SoftAssigner.valid_slices` works from: the `valid_z`
        policy recorded in `sdata.attrs` when there is one, else every plane of
        the labels array.
        """
        recorded = SpatialDataHelpers._recorded_valid_z(self.sdata, self.fov)
        if recorded:
            return sorted(int(v) for v in recorded)
        el = self.sdata.labels[self.labels_key]
        if "z" in getattr(el, "dims", ()):
            return list(range(int(el.sizes["z"])))
        return [0]

    def _slice_of(self, df):
        """Which z-slice each transcript sits on, as a float (NaN for none).

        The same rule that put the transcript there in the first place --
        `SoftAssigner.transcript_slices`, and through it the `valid_z`/`snap_z`
        policy in `sdata.attrs`: a 2D FOV puts everything on slice 0, and in 3D
        a transcript belongs to the valid slice its z rounds to, snapped to the
        nearest one or belonging to no slice at all when it rounds to none.

        Rounding the raw z instead would disagree with the run being drawn
        wherever the two differ -- every transcript of a 2D FOV whose z is not 0,
        and every one `snap_z` moved onto a slice -- and a transcript landing on
        a slice that is not drawn is simply left off the plot, so the
        disagreement shows up as missing markers rather than as an error.
        """
        allowed = np.asarray(self.valid_slices(), dtype=float)
        column = "z" if "z" in df.columns else "global_z"
        if len(allowed) == 1 or column not in df.columns:
            return np.full(len(df), allowed[0], dtype=float)
        nearest, _ = SpatialDataHelpers._assign_points_to_slices(
            df[column].to_numpy(dtype=float),
            allowed,
            SpatialDataHelpers._resolve_snap_z(self.sdata, None),
        )
        return np.asarray(nearest, dtype=float)

    def resolve_transcripts(self, transcripts, focus_cells=None):
        """Whatever `plot_subset` was handed, as a frame it can draw.

        transcripts: the name of an assignment column, a frame from
            `cell_transcripts`/`changed_transcripts`, or a list of transcript ids
            to look up in the FOV's table.
        focus_cells: the cells the plot is about, if any. A column name is read
            against them -- every transcript either assignment puts in one of
            them, which is what "show me this cell's transcripts" means and what
            a plot zoomed to a cell is for. With nothing in focus there is no
            cell to scope to, so a column name means the whole FOV's edits.

        A column name deserves that scoping because the two differ by orders of
        magnitude: for one real cell, 162 transcripts belong to it and 1 of them
        was moved by the run, so the unscoped reading draws a near-empty plot
        that looks like a bug. Pass a `changed_transcripts` frame to get the
        edits of a cell you are zoomed to.
        """
        if transcripts is None:
            return None
        if isinstance(transcripts, str):
            if focus_cells:
                return self.cell_transcripts(focus_cells, transcripts)
            return self.changed_transcripts(transcripts)
        if isinstance(transcripts, pd.DataFrame):
            return transcripts

        df = self.get_transcripts()
        if df is None:
            raise KeyError(f"No '{self.fov}_points' in this SpatialData.")
        wanted = df.reindex(pd.Index(list(transcripts)))
        if wanted["x"].isna().any():
            raise KeyError(
                f"{int(wanted['x'].isna().sum())} of the transcript ids given "
                f"are not in {self.fov}_points."
            )
        out = pd.DataFrame(index=wanted.index)
        out["x"] = wanted["x"].to_numpy(dtype=float)
        out["y"] = wanted["y"].to_numpy(dtype=float)
        out["z"] = self._slice_of(wanted)
        out["from"] = None
        out["to"] = None
        return out

    def transcript_groups(self, frame, focus_cells):
        """Which colour each transcript should get, relative to the focus cells.

        A transcript in one of them both before and after is "kept", one a run
        moved into one is "gained", and one it moved out of is "lost". One that
        ended up in neither is "candidate" when a focus cell was in the running
        for it -- the frame's `considered` column, which `cell_transcripts` sets
        from `cell_ids` -- and "changed" otherwise, which is also what everything
        gets when there are no focus cells, or when the ids came without a
        from/to.
        """
        focus = {str(c) for c in focus_cells}
        if not focus or "from" not in frame.columns:
            return ["changed"] * len(frame)

        considered = (
            frame["considered"].to_numpy(dtype=bool)
            if "considered" in frame.columns
            else np.zeros(len(frame), dtype=bool)
        )
        groups = []
        for was, now, in_running in zip(frame["from"], frame["to"], considered):
            was_in = was is not None and str(was) in focus
            now_in = now is not None and str(now) in focus
            if now_in and was_in:
                groups.append("kept")
            elif now_in:
                groups.append("gained")
            elif was_in:
                groups.append("lost")
            else:
                groups.append("candidate" if in_running else "changed")
        return groups

    def get_masks(self):
        """The FOV's mask as a `(z, y, x)` numpy array (2D masks get one slice)."""
        el = self.sdata.labels[self.labels_key]
        arr = np.asarray(
            el.transpose("z", "y", "x") if "z" in el.dims else el.transpose("y", "x")
        )
        return arr if arr.ndim == 3 else arr[None, ...]

    def get_image(self):
        """The FOV's image as a `(c, z, y, x)` float array, normalised, or None.

        Normalisation matches what the plot used to do for itself: each z-slice
        scaled into 0-1, or histogram-equalised when `exposure_eq` is set.
        """
        if self.image_key is None:
            return None
        el = self.sdata.images[self.image_key]
        dims = el.dims
        arr = np.asarray(
            el.transpose("c", "z", "y", "x") if "z" in dims else el.transpose("c", "y", "x")
        ).astype(float)
        if arr.ndim == 3:
            arr = arr[:, None, ...]

        if self.exposure_eq:
            return skimage.exposure.equalize_hist(arr)
        out = np.zeros_like(arr)
        for c in range(arr.shape[0]):
            for z in range(arr.shape[1]):
                layer = arr[c, z]
                span = np.max(layer)
                out[c, z] = (layer - np.min(layer)) / span if span else layer
        return out

    # ----------------------------------------------------------------- #
    # what to draw                                                       #
    # ----------------------------------------------------------------- #

    def cell_extent(self, cell_id, border=100):
        """Where a cell sits: `(z indices, row slice, col slice)`, grown by border.

        The mask value for cell `c` is `c + 1`, the SoftSeg convention.
        """
        masks = self.get_masks()
        zs, rows, cols = np.nonzero(masks == cell_id + 1)
        if len(zs) == 0:
            return None
        n_z, n_rows, n_cols = masks.shape
        return (
            list(range(int(zs.min()), int(zs.max()) + 1)),
            slice(max(int(rows.min()) - border, 0), min(int(rows.max()) + border, n_rows)),
            slice(max(int(cols.min()) - border, 0), min(int(cols.max()) + border, n_cols)),
        )

    def outline_groups(self, mask_values, highlight_cells=None, adata=None,
                       highlight_type=None, type_col="celltypes"):
        """Which outline colour each mask value should get.

        A cell named in `highlight_cells` is drawn red; one whose `adata` row
        matches `highlight_type` is drawn white; everything else black.
        """
        highlight = set(highlight_cells or [])
        groups = []
        for m in mask_values:
            cell = int(m) - 1
            if cell in highlight:
                groups.append("highlight")
            elif (
                adata is not None
                and highlight_type is not None
                and str(cell) in adata.obs.index
                and adata.obs.loc[str(cell)][type_col] == highlight_type
            ):
                groups.append("type")
            else:
                groups.append("other")
        return groups

    @staticmethod
    def _in_crop(frame, rows, cols):
        """Where each transcript falls in a crop, and which ones are inside it.

        A transcript at `(x, y)` sits at mask array position `[y - 1, x - 1]`,
        the SoftSeg convention, and spatialdata_plot puts array cell `i` across
        `[i, i + 1]` -- so its centre lands half a pixel in. Cropping then shifts
        both axes by where the window starts.
        """
        x = frame["x"].to_numpy(dtype=float) - cols.start - 0.5
        y = frame["y"].to_numpy(dtype=float) - rows.start - 0.5
        inside = (
            (x >= 0) & (x <= cols.stop - cols.start)
            & (y >= 0) & (y <= rows.stop - rows.start)
        )
        return x, y, inside

    def _slice_points(self, z, rows, cols, frame, groups):
        """The transcripts on one z-slice, in the crop's display space.

        returns: `(frame, n)` -- `x`, `y` and `group` for the ones that land on
            this slice inside the crop, or None when none do, and how many.
        """
        if frame is None or not len(frame):
            return None, 0
        # the slice column is a float, and NaN for a transcript that belongs to
        # no slice at all; comparing it as one keeps those out rather than
        # raising on the cast to int
        on_slice = frame["z"].to_numpy(dtype=float) == float(z)
        x, y, in_crop = self._in_crop(frame, rows, cols)
        inside = on_slice & in_crop
        if not inside.any():
            return None, 0

        return pd.DataFrame({
            "x": x[inside],
            "y": y[inside],
            "group": [g for g, keep in zip(groups, inside) if keep],
        }), int(inside.sum())

    def _draw_transcripts(self, ax, points):
        """Put one slice's transcript markers on the axes.

        Drawn with `ax.scatter` rather than through `render_points`, which is
        otherwise the natural fit: that path builds an AnnData per call to reach
        scanpy's colouring, and doing so warns twice over (a deprecated `dtype=`,
        and an integer index it rewrites as strings) before matplotlib warns
        again that the `cmap`/`norm` it always passes have nothing to colour-map.
        None of the three mean anything here -- the colours are already decided,
        one per group -- and a scatter takes them directly.

        The markers are in the crop's display space, which is the space
        `pl.show` left the axes in, so they need no transform. The dark edge is
        the marker's own contrast, in place of the second overdrawn layer a
        halo needed when this went through `render_points`.
        """
        if points is None or not len(points):
            return
        ax.scatter(
            points["x"].to_numpy(dtype=float),
            points["y"].to_numpy(dtype=float),
            s=self.TRANSCRIPT_PX,
            c=[self.TRANSCRIPT_COLORS[g] for g in points["group"]],
            edgecolors=self.TRANSCRIPT_EDGE_COLOR,
            linewidths=self.TRANSCRIPT_EDGE_PX,
            # over the image and the outlines, which pl.show has already drawn
            zorder=100,
        )

    def _undrawn_note(self, frame, drawn, zs, rows, cols):
        """Why some of the transcripts asked for did not reach a plot, or None.

        A transcript is left off when it belongs to no z-slice, when its slice
        is not among the ones being drawn -- zooming to a cell draws only the
        slices that cell appears on -- or when it falls outside the crop. Each
        of those is a silent omission on the plot, so it is worth saying which
        happened, especially when it happened to all of them.
        """
        missing = len(frame) - drawn
        if missing <= 0:
            return None
        z = frame["z"].to_numpy(dtype=float)
        _, _, in_crop = self._in_crop(frame, rows, cols)
        on_slice = np.isin(z, np.asarray(zs, dtype=float))
        reasons = []
        no_slice = int(np.isnan(z).sum())
        if no_slice:
            reasons.append(f"{no_slice} on no segmented z-slice")
        other_slice = int((~on_slice & ~np.isnan(z)).sum())
        if other_slice:
            reasons.append(f"{other_slice} on a z-slice not being drawn")
        outside = int((on_slice & ~in_crop).sum())
        if outside:
            reasons.append(f"{outside} outside the crop")
        return (
            f"fov_{self.fov:0>4}: {missing} of {len(frame)} transcripts are not "
            f"drawn (" + ", ".join(reasons) + ")."
        )

    def _cell_polygons(self, labels2d):
        """One polygon per cell in a cropped slice, in the crop's display space.

        `render_labels` cannot draw these: it colours by category, so every cell
        sharing a colour is one region to it, and two neighbours that touch come
        out as a single ring around the pair with no border between them. A
        polygon per cell is a polygon per cell however they are coloured, so each
        one keeps its own closed outline.

        The contours are the ones `SpatialDataHelpers` traces for the pipeline,
        so what is drawn is the boundary transcripts were actually measured
        against. Each cell is contoured inside its own bounding box rather than
        over the whole crop, which is what makes this affordable on a full FOV.

        returns: `{mask value: geometry}`, the geometries placed so that mask
            array cell `i` spans `[i, i + 1]` -- the convention spatialdata_plot
            draws in, and the one `_in_crop` puts the transcripts in.
        """
        from scipy import ndimage
        from shapely.affinity import translate

        # compact the mask values to 1..n so find_objects can give each cell a
        # bounding box in one pass; the values themselves are global cell ids and
        # far too large to index by. The +1 matters: find_objects treats label 0
        # as background and returns nothing for it, which would drop a cell in a
        # crop that has no background pixels of its own
        values, compact = np.unique(labels2d, return_inverse=True)
        compact = (compact + 1).reshape(labels2d.shape)
        boxes = ndimage.find_objects(compact)

        geoms = {}
        for k, value in enumerate(values, start=1):
            if value == 0:  # background
                continue
            box = boxes[k - 1]
            if box is None:
                continue
            row_slice, col_slice = box
            binimg = (compact[box] == k).astype(np.uint8)
            # contour the single cell: _slice_to_polygons keys on mask value - 1,
            # so a 1/0 image comes back under cell id 0
            traced = SpatialDataHelpers._slice_to_polygons(binimg)
            geom = traced.get(0)
            if geom is None:
                continue
            # _slice_to_polygons offsets vertices by 1 into transcript space, and
            # a transcript at x sits at array index x - 1 whose display centre is
            # x - 0.5 -- the same half-pixel shift `_in_crop` applies
            geoms[int(value)] = translate(
                geom, xoff=col_slice.start - 0.5, yoff=row_slice.start - 0.5
            )
        return geoms

    def _slice_sdata(self, z, rows, cols, image, masks, groups_for):
        """A flat 2D SpatialData of one z-slice, ready for spatialdata_plot.

        Holds the matching image if there is one, and the cells as one shapes
        element per outline group, so each group can be drawn in its own colour
        while every cell keeps its own ring. The transcripts are not in it --
        they go on the axes afterwards, see `_draw_transcripts`.
        """
        labels2d = masks[z][rows, cols]
        images = {}
        if image is not None:
            images["microscopy"] = Image2DModel.parse(
                image[:, z, rows, cols],
                dims=("c", "y", "x"),
                transformations={"plot": Identity()},
            )

        present = [m for m in np.unique(labels2d) if m != 0]
        if not present:
            # no cells on this slice, so there is nothing to outline; the caller
            # skips the slice entirely
            return None, present

        geoms = self._cell_polygons(labels2d)
        groups = dict(zip(present, groups_for(present)))
        shapes = {}
        for group in self.OUTLINE_COLORS:
            members = [m for m in present if groups.get(m) == group and m in geoms]
            if not members:
                continue
            shapes[f"outline_{group}"] = ShapesModel.parse(
                gpd.GeoDataFrame(
                    {"geometry": [geoms[m] for m in members]},
                    index=[str(m) for m in members],
                ),
                transformations={"plot": Identity()},
            )

        return SpatialData(images=images, shapes=shapes), present

    def _transcript_legend(self, ax, groups):
        """Key the transcript colours on one slice's axes.

        `groups` is what this slice actually drew, so the legend only ever names
        colours that are on the plot in front of it -- a slice holding nothing
        but the cell's own transcripts is not keyed for moves that happened two
        slices up. It follows that the legend changes between slices of one call,
        and that a slice with no transcripts on it gets none at all.

        Drawn below the axes: the crops are tight around a cell, so a legend
        inside them would cover the cells being looked at. The swatch is the
        marker as it appears on the plot -- coloured face, dark edge -- rather
        than a plain patch.
        """
        wanted = [g for g in self.TRANSCRIPT_COLORS if g in set(groups)]
        if not wanted:
            return
        handles = [
            Line2D(
                [], [], linestyle="none", marker="o",
                markerfacecolor=self.TRANSCRIPT_COLORS[g],
                markeredgecolor=self.TRANSCRIPT_EDGE_COLOR,
                markeredgewidth=self.TRANSCRIPT_EDGE_PX, markersize=8,
                label=self.TRANSCRIPT_LABELS[g],
            )
            for g in wanted
        ]
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=min(len(handles), 4),
            frameon=False,
            fontsize="small",
            handletextpad=0.4,
            columnspacing=1.4,
        )

    # ----------------------------------------------------------------- #
    # the plot                                                           #
    # ----------------------------------------------------------------- #

    def plot_subset(
        self,
        target_cell=None,
        highlight_cells=None,
        adata=None,
        highlight_type=None,
        single_channel=None,
        transcripts=None,
    ):
        """
        target_cell: if provided, will zoom in on that cell specifically, and
           highlight it in red alongside `highlight_cells`
        highlight_cells: cell ids listed will be highlighted in red
        adata: anndata with cell ids and other information. Only needed for highlight_types
        highlight_type: cells that match this type will be highlighted in white
        single_channel: if provided, only this channel of the segmentation image will be usd
        transcripts: transcripts to draw over the segmentation. Give it the name
           of an assignment column -- "first_try", say -- and what it draws
           depends on whether anything is in focus: with a `target_cell` or
           `highlight_cells` it draws **those cells' claim on the data** --
           every transcript either that column or `og_cell` places in them, plus
           every one they were a candidate for and lost (`cell_transcripts`) --
           and with nothing in focus it draws the whole FOV's edits, the ones the
           run moved (`changed_transcripts`). To draw a zoomed-to cell's edits
           rather than its claim, pass a `changed_transcripts(...)` frame -- any
           frame from either method is drawn as given. A plain list of transcript
           ids works too. Each marker is coloured by where it stands relative to
           the cells in focus: cyan for one that stayed in, green for one moved
           in, magenta for one moved out, grey for one a focus cell was in the
           running for and did not get, yellow for one that moved elsewhere.
           Only the transcripts on the slice being drawn, and inside the crop,
           appear on it; the rest are counted up in a warning at the end, since a
           plot missing them looks no different from one with nothing to draw.
        """
        # imported for its side effect -- it registers the `.pl` accessor used
        # below -- and done here so the rest of SupportFuncs does not depend on
        # the plotting stack
        import spatialdata_plot

        assert spatialdata_plot is not None

        self._require_fov()
        masks = self.get_masks()
        image = self.get_image()

        zs = list(range(masks.shape[0]))
        rows = slice(0, masks.shape[1])
        cols = slice(0, masks.shape[2])
        if target_cell is not None:
            extent = self.cell_extent(target_cell, border=100)
            if extent is None:
                raise ValueError(
                    f"Cell {target_cell} does not appear in fov_{self.fov:0>4}."
                )
            zs, rows, cols = extent

        # the cells in focus -- outlined red, and what a transcript's colour is
        # measured against. The target cell is one of them: it is the cell being
        # looked at, so leaving it black would make the zoom unreadable.
        focus = list(highlight_cells or [])
        if target_cell is not None and target_cell not in focus:
            focus.append(target_cell)

        def groups_for(mask_values):
            return self.outline_groups(
                mask_values,
                highlight_cells=focus,
                adata=adata,
                highlight_type=highlight_type,
            )

        frame = self.resolve_transcripts(transcripts, focus_cells=focus)
        tr_groups = None if frame is None else self.transcript_groups(frame, focus)

        drawn = 0
        for z in zs:
            points, n_points = self._slice_points(z, rows, cols, frame, tr_groups)
            sub, present = self._slice_sdata(z, rows, cols, image, masks, groups_for)
            if len(present) == 0:
                # no cells on this slice, so it is not drawn at all -- and any
                # transcripts that fell on it were not drawn either
                continue
            drawn += n_points

            rendered = sub
            if "microscopy" in sub.images:
                rendered = rendered.pl.render_images(
                    element="microscopy",
                    channel=single_channel if single_channel is not None else None,
                )
            # outlines only, so the image underneath stays visible. One call per
            # group, in OUTLINE_COLORS order, so the highlighted cells go on top
            for group, color in self.OUTLINE_COLORS.items():
                if f"outline_{group}" not in sub.shapes:
                    continue
                rendered = rendered.pl.render_shapes(
                    element=f"outline_{group}",
                    fill_alpha=0.0,
                    outline_alpha=1.0,
                    outline_color=color,
                    outline_width=self.CONTOUR_PX,
                    colorbar=False,
                )
            # the axes come back rather than being shown straight away, so the
            # transcript legend can go on before the figure is drawn
            axes = rendered.pl.show(
                coordinate_systems="plot",
                title=f"fov_{self.fov:0>4}, z-slice {z}",
                colorbar=False,
                legend_loc=None,
                return_ax=True,
                show=False,
            )
            ax = axes if not isinstance(axes, np.ndarray) else axes.flat[0]
            self._draw_transcripts(ax, points)
            # frame the crop, not whatever happens to be drawn on this slice:
            # with no image the shapes alone would set the extent, so every
            # slice would be framed differently and a crop with cells only in
            # one corner would lose the rest of itself. Array cell `i` spans
            # `[i, i + 1]`, so the crop runs from 0 to its width.
            ax.set_xlim(0, cols.stop - cols.start)
            ax.set_ylim(rows.stop - rows.start, 0)
            # keyed on this slice's markers, not the call's -- `points` is what
            # was actually drawn above
            if points is not None:
                self._transcript_legend(ax, points["group"])
            plt.show()

        if frame is not None:
            note = self._undrawn_note(frame, drawn, zs, rows, cols)
            if note is not None:
                warnings.warn(note, stacklevel=2)


class UnitTestWarper:
    """Deforms a simulated dataset and hands it back in SoftAssigner's layout.

    The input is the st-qc simulation benchmark object: shapes ``cells`` and
    ``cell_borders``, points ``transcripts``, table ``counts``. Its cell
    boundaries are warped to stand in for imperfect segmentation, and the result
    is rasterized into the per-FOV segmentation/transcript pair
    :class:`~SoftSeg.SoftAssigner.SoftAssigner` reads:

    * ``labels[f"{fov}_labels"]``  -- the warped cells as an integer mask, using
      the SoftSeg convention that mask value ``m`` is cell id ``m - 1``.
    * ``points[f"{fov}_points"]``  -- the transcripts in that mask's pixel space,
      1-based as SoftSeg expects, with the untouched simulation coordinates kept
      alongside as ``global_x``/``global_y``.

    Both sit in a per-FOV coordinate system plus ``"global"``, and the
    z-correction policy is recorded in ``attrs``, so the object can be handed
    straight to ``SoftAssigner(sdata)``.

    The ground truth stays in the object next to them: the warped shapes
    (``f"{labels}_warped"``, plus ``cell_borders_warped`` when the object has a
    borders element) and the ``counts_warped`` table.
    """

    def __init__(self, sdata, fov="0000", labels="cells"):
        """
        sdata: the simulation object, or a path to a zarr store holding one.
        fov: the FOV token to name the emitted elements after.
        labels: name of the shapes element holding the cell segmentation, passed
           through to `warp_shapes`. Defaults to "cells"; the warped copy of it
           is what gets rasterized into `{fov}_labels`.

        Everything this produces is written back into `sdata`, element by
        element, and saved to the store it came from -- an in-memory object is
        modified in place and saved nowhere.
        """

        SpatialDataHelpers.quiet_ome_zarr()
        self.sdata = sd.read_zarr(sdata) if isinstance(sdata, (str, Path)) else sdata
        self.fov = fov
        self.labels = labels
        warnings.filterwarnings(
            "ignore",
            message="zarr v3 autosharding will be the default in the next minor release",
        )

    @property
    def labels_key(self):
        return f"{self.fov}_labels"

    @property
    def points_key(self):
        return f"{self.fov}_points"

    def _source_transcripts(self):
        """The transcripts in the simulation's own coordinates.

        A run consumes ``transcripts``, leaving them under the per-FOV name in
        the mask's pixel space with the original coordinates preserved as
        ``global_x``/``global_y``. Rebuilding from those lets the warper be run
        again -- over a range of amplitudes, say -- rather than being single-use.
        """
        if "transcripts" in self.sdata.points:
            return self.sdata.points["transcripts"]

        if self.points_key in self.sdata.points:
            df = self.sdata.points[self.points_key].compute()
            if {"global_x", "global_y"}.issubset(df.columns):
                df = df.drop(columns=["x", "y"]).rename(
                    columns={"global_x": "x", "global_y": "y"}
                )
                # Any assignment a previous pass left behind was made against
                # the old segmentation, so it cannot travel to a new warp.
                stale = {"cell_ids", "og_cell", "og_type"}
                stale |= {c for c in df.columns if f"{c}_type" in df.columns}
                stale |= {c for c in df.columns if c.endswith("_type")}
                df = df.drop(columns=[c for c in stale if c in df.columns])
                kwargs = {"feature_key": "gene"} if "gene" in df.columns else {}
                rebuilt = PointsModel.parse(
                    df,
                    coordinates={"x": "x", "y": "y"},
                    transformations={"global": Identity()},
                    **kwargs,
                )
                self.sdata.points["transcripts"] = rebuilt
                return rebuilt

        raise KeyError(
            "No 'transcripts' points element, and no "
            f"'{self.points_key}' carrying global_x/global_y to rebuild them "
            "from. This object does not look like a simulation dataset."
        )

    def run(
        self, amplitude=10, length_scale=200, seed=0, gene_col="feature_name", scale=1
    ):
        warped = warp_shapes(
            self.sdata,
            smooth_field_warp(
                amplitude=amplitude, length_scale=length_scale, seed=seed
            ),
            labels=self.labels,
        )
        # whatever warp_shapes deformed -- the cell segmentation, plus a matching
        # borders element when the object has one
        for name, gdf in warped.shapes.items():
            self.sdata.shapes[name] = gdf
            SpatialDataHelpers.save_element(self.sdata, name)
        warped_labels = f"{self.labels}_warped"

        # Count against the warped cells while the transcripts are still in the
        # simulation's own coordinates. A previous run will have renamed the gene
        # column to "gene", so take whichever of the two this object carries.
        src = self._source_transcripts()
        if gene_col not in src.columns and "gene" in src.columns:
            gene_col = "gene"

        new_table = self.sdata.aggregate(
            values="transcripts", by=warped_labels, value_key=gene_col, agg_func="count"
        )
        index_diff = pd.Index(
            set(self.sdata.tables["counts"].var.index)
            - set(new_table["table"].var.index)
        )
        new_table.tables["table"].var.index.append(index_diff)
        # note that the following might break if we have warped cells with no transcripts assigned to them
        new_table.tables["table"].obs["celltype"] = self.sdata.tables["counts"].obs[
            "celltype"
        ]
        self.sdata.tables["counts_warped"] = new_table.tables["table"]
        SpatialDataHelpers.save_element(self.sdata, "counts_warped")

        points = self._source_transcripts().compute()
        min_x = points["x"].min()
        min_y = points["y"].min()
        max_x = points["x"].max()
        max_y = points["y"].max()
        rasterized = sd.rasterize(
            self.sdata[warped_labels],
            ["x", "y"],
            min_coordinate=[min_x * scale, min_y * scale],
            max_coordinate=[max_x * scale, max_y * scale],
            target_coordinate_system="global",
            target_unit_to_pixels=scale,
        )

        # The raster's own grid is the authority on where a transcript falls in
        # it: `rasterize` snaps the grid to whole pixels, so its origin is not
        # exactly min_coordinate. Composing the transcripts' transform with the
        # inverse of the raster's gives the exact mapping into pixel space.
        raster_to_global = get_transformation(rasterized, "global")
        trs_to_global = get_transformation(self.sdata.points["transcripts"], "global")
        to_pixels = np.asarray(
            Sequence([trs_to_global, raster_to_global.inverse()]).to_affine_matrix(
                input_axes=("x", "y"), output_axes=("x", "y")
            )
        )

        fov_cs = f"fov_{self.fov}"
        transformations = {fov_cs: Identity(), "global": raster_to_global}

        # The rasterizer numbers its labels by its own internal index, so remap
        # them onto the cell numbers the simulation uses. Those are 1-based
        # (`cell_1` ...), and SoftSeg reads mask value `m` as cell id `m - 1`, so
        # storing the number as-is makes cell id = number - 1. Two consumers
        # depend on exactly that: `ParamSweeper.process_unit_test` re-indexes the
        # reference table positionally (row 0 is `cell_1`), and
        # `ParamSweeper.rate_unit_test` names a cell back as `f"cell_{id + 1}"`.
        mapping = {
            k: int(x[5:]) for k, x in rasterized.label_index_to_category.items()
        }
        if 0 in mapping.values():
            raise ValueError(
                "A cell is numbered 0, which would be stored as mask value 0 and "
                "so be indistinguishable from background. The simulation's cells "
                "are expected to be numbered from 1."
            )
        mapping[0] = 0
        f = np.vectorize(lambda x: mapping[x])
        im = f(rasterized.to_numpy()).squeeze().astype(np.uint32)
        self.sdata.labels[self.labels_key] = Labels2DModel.parse(
            im, dims=("y", "x"), transformations=dict(transformations)
        )
        SpatialDataHelpers.save_element(self.sdata, self.labels_key)

        # The transcripts, moved into that mask's pixel space. SoftSeg's x/y are
        # 1-based against the mask array -- `masks_to_shapes` shifts the cell
        # contours by the matching +1 -- so the pixel coordinates are offset to
        # suit. The simulation's own coordinates are kept as global_x/global_y,
        # which is where the canonical layout puts them too.
        df = points.reset_index(drop=True)
        gx = df["x"].to_numpy(dtype=float)
        gy = df["y"].to_numpy(dtype=float)
        # applied directly rather than through `spatialdata.transform`, which
        # builds a whole transformed element and is far more machinery than two
        # columns of arithmetic need
        px = np.column_stack([gx, gy, np.ones(len(df))]) @ to_pixels.T
        df["global_x"] = gx
        df["global_y"] = gy
        df["x"] = px[:, 0] + 1
        df["y"] = px[:, 1] + 1
        if gene_col in df.columns and gene_col != "gene":
            df = df.rename(columns={gene_col: "gene"})
        # a re-run starts from a table that already carries one
        df = df.drop(columns=["index"], errors="ignore")
        df.insert(0, "index", np.arange(len(df)))

        kwargs = {"feature_key": "gene"} if "gene" in df.columns else {}
        self.sdata.points[self.points_key] = PointsModel.parse(
            df,
            coordinates={"x": "x", "y": "y"},
            transformations=dict(transformations),
            **kwargs,
        )
        SpatialDataHelpers.save_element(self.sdata, self.points_key)
        # the transcripts now live under the per-FOV name, in the mask's space
        SpatialDataHelpers.remove_element(self.sdata, "transcripts")

        # 2D, so there is a single z-slice and nothing to snap
        attrs = dict(getattr(self.sdata, "attrs", None) or {})
        attrs[SpatialDataHelpers.SOFTSEG_ATTRS] = {
            "z_offset": 0,
            "snap_z": False,
            "valid_z": {str(self.fov): [0]},
        }
        self.sdata.attrs = attrs
        if self.sdata.is_backed():
            self.sdata.write_attrs()

        return self.sdata


class ParamSweeper:
    """Sweeps `SoftAssigner` parameters over one SpatialData.

    Every run works on the same object, so the sweep's bookkeeping is done with
    column names rather than with separate output directories: blurring
    overwrites the `cell_ids` column each time, and each assignment is written to
    a column named after the **full** parameter set that produced it
    (`assigned_size25_dist3.5_default10_min0.8`). The sdata therefore keeps one
    distinguishable column per combination tried, alongside the transcript table
    it came from.

    The FOVs to sweep default to every FOV in the object, which for the unit-test
    datasets this is aimed at is a single one.
    """

    # what `rate_unit_test` collapses every flavour of "no cell here" to
    MISSING = "none"

    def __init__(self, sdata, pool_size=5, fovs=None):
        self.asgn = SoftAssigner(sdata=sdata, pool_size=pool_size)
        self.fovs = list(fovs) if fovs is not None else self.asgn.get_all_fovs()
        if not self.fovs:
            raise ValueError(
                "No FOVs found in this SpatialData: expected paired "
                "'{fov}_labels' and '{fov}_points' elements."
            )
        # set by blur_fovs, so an assignment column can name the blur that fed it
        self.blur_params = None

    @property
    def sdata(self):
        return self.asgn.sdata

    def param_name(self, default_thresh, min_thresh):
        """Column name for one point in the sweep, blur parameters included."""
        name = ""
        if self.blur_params is not None:
            min_size, max_dist = self.blur_params
            name += f"size{min_size}_dist{max_dist}_"
        return f"assigned_{name}default{default_thresh}_min{min_thresh}"

    def blur_fovs(self, min_size=25, max_dist=3.5, save=True):
        """Re-blur every swept FOV.

        Goes through `blur_all_fovs` rather than calling `blur_fov` per FOV, so
        the decision to replace the previous point's `cell_ids` follows from the
        parameters having changed -- which is exactly what a sweep does -- rather
        than from this method asserting it.
        """
        self.blur_params = (min_size, max_dist)
        self.asgn.blur_all_fovs(
            min_size, max_dist, sel_fovs=self.fovs, save=save
        )

    def process_unit_test(self, table_name, cats=None):
        """Build the scoring matrix from a cell-typed reference table.

        `table_name` names a table in the sdata -- where `generate_cxg_table` and
        `CellTypeAssigner` leave their results.
        """
        adata = self.sdata.tables[table_name]

        if cats is None:
            cats = {"celltype": ["ct_0", "ct_1"]}

        if hasattr(adata.X, "todense"):
            adata.X = adata.X.todense()
        adata.obs.index = [str(x) for x in range(len(adata))]
        adata.obs["instance_id"] = [f"cell_{x}" for x in range(len(adata))]
        # QC bounds, if this dataset records any, come from its own attrs
        self.asgn.get_scoring_matrix(adata, cats, normed=False)

    def scan(self, gene_col_name, min_thresh, **kwargs):
        """Scan every swept FOV once, for reuse across `default_thresh` values.

        Sorting transcripts into confident and ambiguous is the expensive part of
        an assignment and depends on `min_thresh`, not on `default_thresh`, so a
        sweep over the latter scans once and resolves repeatedly. Returns
        `{fov: scan_state}` to hand back to `assign`.
        """
        scans = {}
        for fov in self.fovs:
            state = self.asgn.scan_overlapping_regions(
                fov,
                gene_col_name=gene_col_name,
                min_thresh=min_thresh,
                disable_tqdm=True,
                **kwargs,
            )
            if state is not None:
                scans[fov] = state
        return scans

    def assign(self, gene_col_name, default_thresh=10, min_thresh=0.8, save=True,
               scans=None):
        """Evaluate overlapping regions for this parameter set.

        Returns the name of the column written, which is also where the result
        now lives in the sdata. `save=False` keeps the edit in memory; see
        `save_transcripts`. Pass `scans` from `scan` to reuse a scan taken at the
        same `min_thresh` rather than repeating it.
        """
        name = self.param_name(default_thresh, min_thresh)
        for fov in self.fovs:
            if scans is not None and fov not in scans:
                continue  # nothing ambiguous in this FOV
            self.asgn.evaluate_overlapping_regions_single_fov(
                fov,
                assigned_col=name,
                gene_col_name=gene_col_name,
                default_thresh=default_thresh,
                min_thresh=min_thresh,
                disable_tqdm=True,
                overwrite=True,
                save=save,
                scan_state=None if scans is None else scans[fov],
            )

        return name

    def _normalize_missing(self, series):
        """One marker for "no cell here", whichever way it was written.

        All four assignment columns now spell it `SoftAssigner.MISSING`
        (`pd.NA`), but a store written before that has the string "None" in
        og_cell/og_type/{col}_type and a null in the assignment column. Both have
        to collapse, or a column using the other spelling reaches `int()` below
        -- which is how rating og_cell, as the unmodified segmentation's
        baseline, used to raise "invalid literal for int(): 'None'".
        """
        values = series.astype(object)
        values = values.where(~pd.isna(values), self.MISSING)
        return values.replace({"None": self.MISSING, "": self.MISSING})

    def save_transcripts(self):
        """Persist the swept FOVs' points elements to the store.

        A sweep leaves its columns in memory -- writing the whole element after
        every parameter point would cost more than the scoring does, and the
        table grows by three columns each time -- so the whole run is flushed
        once, here.
        """
        for fov in self.fovs:
            SpatialDataHelpers.save_element(self.sdata, self.asgn._points_key(fov))

    def rate_unit_test(self, cell_col, type_col, save=True, frames=None):
        """Accuracy of one assignment against the simulated ground truth.

        Expects the transcript table to carry the truth columns `celltype_sim`
        and `cell_id`, as the simulated datasets do.

        The cell each transcript was given, in the ground truth's own naming, is
        written back to the FOV's points element as `f"{cell_col}_transformed"`
        and saved, so the comparison behind a score stays inspectable in the
        object rather than living on the sweeper.

        `frames` optionally supplies `{fov: dataframe}` to score in place of
        reading the element. A sweep passes the frames its scans are already
        holding, so every parameter point's columns land on one table instead of
        each point re-reading and overwriting the last one's.
        """
        type_hits = cell_hits = total = 0

        for fov in self.fovs:
            tr = (
                self.asgn.get_transcripts(fov)
                if frames is None or fov not in frames
                else frames[fov]
            )

            # a FOV with nothing ambiguous to resolve is skipped without the
            # column being written; that is "no transcript reassigned", not
            # missing data
            for col in [cell_col, type_col, "og_cell", "og_type"]:
                if col not in tr.columns:
                    tr[col] = np.nan

            cells = self._normalize_missing(tr[cell_col])
            types = self._normalize_missing(tr[type_col])
            transformed = cells.apply(
                lambda y: f"cell_{int(y) + 1}" if y != self.MISSING else self.MISSING
            )

            tr[f"{cell_col}_transformed"] = transformed
            # a frame handed in by a sweep is still addressed by transcript id;
            # the element wants that back as a column
            out = tr.reset_index() if "index" not in tr.columns else tr
            self.asgn.set_transcripts(fov, out, save=save)

            type_hits += int((tr["celltype_sim"] == types).sum())
            cell_hits += int((tr["cell_id"] == transformed).sum())
            total += len(tr)

        return type_hits / total, cell_hits / total

    def run_unit_sweep(
        self,
        gene_col_name,
        unit_table_name,
        min_size=25,
        max_dist=3.5,
        default_thresh=10,
        min_thresh=0.8,
    ):
        if isinstance(min_size, list):
            sizes = min_size
        else:
            sizes = [min_size]

        if isinstance(max_dist, list):
            dists = max_dist
        else:
            dists = [max_dist]

        if isinstance(default_thresh, list):
            threshs = default_thresh
        else:
            threshs = [default_thresh]

        if isinstance(min_thresh, list):
            mins = min_thresh
        else:
            mins = [min_thresh]
        results = []

        total_len = len(sizes) * len(dists) * len(threshs) * len(mins)
        pbar = tqdm(total=total_len)

        for size in sizes:
            for dist in dists:
                self.blur_fovs(size, dist, save=False)
                self.process_unit_test(unit_table_name)

                # min_thresh is the outer of the two: it decides which
                # transcripts are ambiguous, so it is what a scan depends on.
                # default_thresh only sets the margin a candidate assignment
                # must clear, and every value of it reuses the same scan.
                for min_t in mins:
                    scans = self.scan(gene_col_name, min_t)
                    frames = {fov: st["tr"] for fov, st in scans.items()}

                    for thresh in threshs:
                        pbar.update(1)
                        entry = {
                            "min_thresh": min_t,
                            "default_thresh": thresh,
                            "max_dist": dist,
                            "min_size": size,
                        }
                        name = self.assign(
                            gene_col_name, thresh, min_t, save=False, scans=scans
                        )
                        type_acc, cell_acc = self.rate_unit_test(
                            name, f"{name}_type", save=False, frames=frames
                        )
                        entry["type_acc"] = type_acc
                        entry["cell_acc"] = cell_acc
                        results.append(entry)

        pbar.close()
        # every parameter point's columns are in memory; write them out once
        self.save_transcripts()
        return results


# --------------------------------------------------------------------------- #
# Formatting a SoftSeg loose-file dataset into a SpatialData                    #
# --------------------------------------------------------------------------- #


class DatasetFormatter:
    """Builds a :class:`spatialdata.SpatialData` out of the SoftSeg on-disk layout.

    This is for datasets that exist as the older per-FOV file set -- a transcript
    CSV, an integer-labelled mask TIFF and optionally a raw image TIFF, addressed
    by ``.format(fov)`` path patterns. Data captured on a supported platform
    should go through the standard readers (``spatialdata_io.cosmx``,
    ``xenium``, and friends) instead; this exists to put example and legacy
    datasets into the same shape.

    The SoftSeg layout, and what the conversion produces:

    * ``csv_loc``   -- per-FOV transcript CSV. Columns: ``x``, ``y`` (local, within
      the FOV), optional ``global_z``, a gene column, and a leading index column.
    * ``mask_loc``  -- per-FOV integer-labelled segmentation mask TIFF, 2D ``(y, x)``
      or 3D ``(z, y, x)`` (z is axis 0). Mask value ``m`` denotes cell id ``m - 1``;
      value 0 is background.
    * ``image_loc`` -- (optional) per-FOV raw microscopy image TIFF.

    Per-FOV global placement is given by ``fov_locs``:
    ``{fov: {"x": [x0, ...], "y": [y0, ...]}}`` -- the FOV's global origin is
    ``(x0, y0)`` and a transcript's global position is its local position plus
    that offset.

    Design (mirrors :func:`spatialdata_io.cosmx`)
    ---------------------------------------------
    * Every element is **split by FOV**: elements are named ``f"{fov}_image"``,
      ``f"{fov}_labels"``, ``f"{fov}_points"``.
    * There is **one coordinate system per FOV** (named ``f"fov_{fov}"``) plus a
      shared ``"global"`` coordinate system, and two type-restricted global systems:
      ``"global_only_image"`` (images only) and ``"global_only_labels"`` (labels
      only), each carrying the same transform as ``"global"``, so images or labels
      can be rendered/operated on in isolation. Points stay on ``fov`` +
      ``"global"`` only.
    * Each element carries an ``Identity`` transform into its own FOV system and a
      ``Translation`` into ``"global"`` (the FOV offset). So the transcripts have
      **two sets of coordinates**: their raw ``x, y`` are their location *within the
      FOV*, and mapping to ``"global"`` gives their location *in the whole
      experiment*. For convenience the global position is also written explicitly
      into the points frame as ``global_x``/``global_y``.
    * All raster elements are validated to xarray ``DataArray``\\s with **correctly
      labelled axes**, including ``z`` for 3D stacks (``("z", "y", "x")`` for masks,
      ``("c", "z", "y", "x")`` for images).

    Pixel convention
    ----------------
    The transcript CSVs' ``x``/``y`` are **1-based** with respect to the mask array:
    a transcript ``(x, y)`` belongs to mask pixel ``[y - 1, x - 1]``. Points are
    stored here **verbatim** from the CSV (so they still match the files on disk,
    and ``global_x``/``global_y`` stay comparable across the dataset); the
    correction is carried by the *polygons* instead --
    :meth:`SpatialDataHelpers.masks_to_shapes` shifts every contour vertex by
    ``+1``. Aggregating points into those shapes therefore reproduces SoftSeg's
    own transcript-to-cell assignment.

    Z alignment
    -----------
    Transcript ``global_z`` values must index the mask's z-slices. Because the two
    come from different files they can disagree (e.g. a 7-plane transcript table
    against a 6-slice mask), which would silently misassign every transcript on the
    extra plane. :meth:`softseg_to_spatialdata` therefore validates the points' z
    against the labels' z axis and **warns** when they do not line up -- but keeps
    every transcript, so the points element stays a faithful copy of the transcript
    table. What to do about the strays is recorded as a policy in
    ``sdata.attrs[SpatialDataHelpers.SOFTSEG_ATTRS]`` (``z_offset``, ``valid_z``, ``snap_z``) and acted
    on by :meth:`SpatialDataHelpers.aggregate_zslice_shapes`, which is where they
    are dropped or snapped. Images and labels are expected to have the same number
    of z-slices, and a mismatch is warned about.
    """

    @staticmethod
    def _fov_coordinate_system(fov) -> str:
        """Name of the per-FOV coordinate system this converter creates.

        Only the conversion needs to build this name: it is what stamps a FOV's
        system onto each new element. Anything reading an existing object takes
        the name from the element instead, via
        :meth:`SpatialDataHelpers._element_fov_cs`, so nothing downstream depends
        on this spelling.
        """
        return f"fov_{fov}"

    @staticmethod
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
        raise ValueError(
            f"pixel_size must be scalar, (sx, sy[, sz]), or dict; got {pixel_size!r}"
        )

    @staticmethod
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
            return Affine(
                matrix, input_axes=("x", "y", "z"), output_axes=("x", "y", "z")
            )
        matrix = np.array(
            [
                [sx, 0.0, sx * x0],
                [0.0, sy, sy * y0],
                [0.0, 0.0, 1.0],
            ]
        )
        return Affine(matrix, input_axes=("x", "y"), output_axes=("x", "y"))

    @staticmethod
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
                raise KeyError(
                    f"FOV {fov!r} not found in fov_locs keys {list(fov_locs)}"
                )
        if isinstance(entry, Mapping):
            x0 = entry["x"][0] if hasattr(entry["x"], "__len__") else entry["x"]
            y0 = entry["y"][0] if hasattr(entry["y"], "__len__") else entry["y"]
        else:  # (x0, y0) pair
            x0, y0 = entry[0], entry[1]
        return float(x0), float(y0)

    @staticmethod
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

    @staticmethod
    def _prep_image(arr, image_dims: Optional[TypingSequence[str]] = None):
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
            raise ValueError(
                f"Unsupported image ndim={ndim}; pass image_dims explicitly."
            )

        # spatialdata requires 4D images in (c, z, y, x) order: physically rearrange
        # the axes so the stored array matches, rather than relying on parse to do it.
        canonical = ("c", "z", "y", "x")
        if len(dims) == 4 and tuple(dims) != canonical:
            order = [dims.index(ax) for ax in canonical]
            arr = da.transpose(arr, axes=order)
            dims = canonical

        return arr, dims

    @staticmethod
    def _parse_image(
        arr, fov, offset, image_dims=None, pixel_size=None, **image_models_kwargs
    ):
        """Parse a raw image array into a spatialdata image element.

        Placed in the FOV system (``Identity``), the shared ``"global"`` system, and
        an image-only ``"global_only_image"`` system (same transform as ``"global"``)
        so images can be operated on/rendered in isolation.
        """
        fov_cs = DatasetFormatter._fov_coordinate_system(fov)
        arr, dims = DatasetFormatter._prep_image(arr, image_dims)
        is_3d = "z" in dims
        transformations = {
            fov_cs: Identity(),
            "global": DatasetFormatter._global_transform(offset, pixel_size, is_3d),
            "global_only_image": DatasetFormatter._global_transform(
                offset, pixel_size, is_3d
            ),
        }
        model = Image3DModel if is_3d else Image2DModel
        return model.parse(
            arr, dims=dims, transformations=transformations, **image_models_kwargs
        )

    @staticmethod
    def _parse_mask(arr, fov, offset, pixel_size=None, **image_models_kwargs):
        """Parse an integer mask array into a spatialdata labels element with
        correctly labelled axes.

        Placed in the FOV system (``Identity``), the shared ``"global"`` system, and
        a labels-only ``"global_only_labels"`` system (same transform as ``"global"``)
        so labels can be operated on/rendered in isolation.
        """
        fov_cs = DatasetFormatter._fov_coordinate_system(fov)
        arr = da.asarray(arr)
        if arr.ndim == 2:  # (y, x)
            transformations = {
                fov_cs: Identity(),
                "global": DatasetFormatter._global_transform(offset, pixel_size, False),
                "global_only_labels": DatasetFormatter._global_transform(
                    offset, pixel_size, False
                ),
            }
            return Labels2DModel.parse(
                arr,
                dims=("y", "x"),
                transformations=transformations,
                **image_models_kwargs,
            )
        if arr.ndim == 3:  # (z, y, x)
            transformations = {
                fov_cs: Identity(),
                "global": DatasetFormatter._global_transform(offset, pixel_size, True),
                "global_only_labels": DatasetFormatter._global_transform(
                    offset, pixel_size, True
                ),
            }
            return Labels3DModel.parse(
                arr,
                dims=("z", "y", "x"),
                transformations=transformations,
                **image_models_kwargs,
            )
        raise ValueError(
            f"Unsupported mask ndim={arr.ndim}; expected 2 (y,x) or 3 (z,y,x)."
        )

    # Key under which this module's own bookkeeping lives in ``SpatialData.attrs``.
    @staticmethod
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
        :data:`SpatialDataHelpers.SOFTSEG_ATTRS`. ``snap_z`` is passed in here only so the warning can
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
        gap = np.abs(
            z - slices[np.argmin(np.abs(z[:, None] - slices[None, :]), axis=1)]
        )
        off = gap > 0.5  # does not round to any valid slice

        if off.any():
            bad = sorted({float(v) for v in z_raw[off]})
            shown = ", ".join(f"{v:g}" for v in bad[:10]) + (
                "..." if len(bad) > 10 else ""
            )
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
                + f" (recorded as snap_z={snap_z} in sdata.attrs[{SpatialDataHelpers.SOFTSEG_ATTRS!r}])."
            )
            warnings.warn(msg, stacklevel=3)

        return df

    @staticmethod
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
        fov_cs = DatasetFormatter._fov_coordinate_system(fov)
        x0, y0 = offset
        df = df.copy()
        df["global_x"] = df["x"] + x0
        df["global_y"] = df["y"] + y0

        coordinates = {"x": "x", "y": "y"}
        z_col = (
            "z"
            if "z" in df.columns
            else ("global_z" if "global_z" in df.columns else None)
        )
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
                "global": DatasetFormatter._global_transform(offset, pixel_size, is_3d),
            },
            **kwargs,
        )

    @staticmethod
    def _discover_fovs(loc: str) -> list:
        """Infer FOV identifiers from a ``.format(fov)`` path pattern on disk."""
        hits = glob.glob(loc.format("****"))
        fovs = []
        for h in hits:
            parsed = _parse.parse(loc, h)
            if parsed is not None:
                fovs.append(parsed[0])
        return fovs

    @staticmethod
    def softseg_to_spatialdata(
        csv_loc: str,
        mask_loc: str,
        fov_locs: Optional[Mapping] = None,
        *,
        image_loc: Optional[str] = None,
        fovs: Optional[TypingSequence] = None,
        gene_col: str = "gene",
        image_dims: Optional[Union[TypingSequence[str], Mapping]] = None,
        pixel_size=None,
        valid_z: Optional[TypingSequence[int]] = None,
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

            ``attrs[SpatialDataHelpers.SOFTSEG_ATTRS]`` carries the z-correction policy:

            * ``"z_offset"`` — the offset already folded into the points' ``z``.
            * ``"snap_z"``   — whether transcripts that miss a slice should be
              snapped (``True``) or dropped (``False``) when aggregating.
            * ``"valid_z"``  — ``{fov: [slice indices]}``, the slices each FOV's
              segmentation actually covers.
        """
        if fovs is None:
            csv_fovs = set(DatasetFormatter._discover_fovs(csv_loc))
            mask_fovs = set(DatasetFormatter._discover_fovs(mask_loc))
            fovs = sorted(csv_fovs & mask_fovs, key=lambda v: (str(type(v)), v))
            if not fovs:
                raise FileNotFoundError(
                    f"No FOVs found matching both {csv_loc!r} and {mask_loc!r}."
                )

        pbar = tqdm(total=len(fovs))

        pixel_size = DatasetFormatter._normalize_pixel_size(pixel_size)
        SpatialDataHelpers.quiet_ome_zarr()

        valid_z_by_fov: dict = {}
        our_sd = SpatialData()
        if save_loc is not None:
            our_sd.write(Path(save_loc), overwrite=True)


        for fov in fovs:
            offset = DatasetFormatter._fov_offset(fov_locs, fov)
            if save_loc is not None:
                del our_sd
                our_sd = read_zarr(Path(save_loc))

            # --- segmentation mask -> labels (per FOV) ---
            n_label_z = None
            mask_path = mask_loc.format(fov)
            if Path(mask_path).is_file():
                mask = tifffile.imread(mask_path, **imread_kwargs)
                element = DatasetFormatter._parse_mask(
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
                    element = DatasetFormatter._parse_image(
                        im,
                        fov,
                        offset,
                        image_dims=DatasetFormatter._resolve_image_dims(
                            image_dims, fov
                        ),
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
                df = DatasetFormatter._align_points_z(
                    df, fov, valid_z=fov_valid_z, z_offset=z_offset, snap_z=snap_z
                )
                if len(df) > 0:
                    our_sd.points[f"{fov}_points"] = DatasetFormatter._parse_points(
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
            SpatialDataHelpers.SOFTSEG_ATTRS: {
                "z_offset": z_offset,
                "snap_z": snap_z,
                "valid_z": valid_z_by_fov,
            }
        }
        if save_loc is not None:
            our_sd.write_attrs()
        return our_sd
        # return SpatialData(images=images, labels=labels, points=points, attrs=attrs)


def adjusted_rand_index(left, right):
    """Agreement between two ways of grouping the same items, chance-corrected.

    Both arguments label the same items -- here, the cell each transcript was
    given by each of two assignments. 1.0 is identical grouping, 0.0 is what two
    unrelated groupings of the same shape would score, and negative is worse than
    that. Cell *names* are not compared, only which transcripts share a cell, so
    this is meaningful even between runs that numbered their cells differently.

    Computed from the contingency table rather than through scikit-learn, which
    is an optional dependency here.
    """
    n = len(left)
    if n < 2:
        return float("nan")

    table = pd.crosstab(pd.Series(list(left)), pd.Series(list(right)))
    counts = table.to_numpy(dtype=float)
    return _ari_from_contingency(
        counts.ravel(), counts.sum(axis=1), counts.sum(axis=0), n
    )


def adjusted_rand_index_from_pairs(pairs_counted):
    """`adjusted_rand_index` from a counted contingency table.

    `pairs_counted` maps `(left cell, right cell)` to how many transcripts both
    sides placed that way -- the same table `crosstab` builds, but countable a
    FOV at a time and bounded by the number of distinct cell pairings rather than
    by the number of transcripts. That is what lets a streamed comparison give
    the same index as one that held every transcript at once.
    """
    by_left = collections.Counter()
    by_right = collections.Counter()
    n = 0
    for (left, right), count in pairs_counted.items():
        by_left[left] += count
        by_right[right] += count
        n += count
    if n < 2:
        return float("nan")
    return _ari_from_contingency(
        np.fromiter(pairs_counted.values(), dtype=float, count=len(pairs_counted)),
        np.fromiter(by_left.values(), dtype=float, count=len(by_left)),
        np.fromiter(by_right.values(), dtype=float, count=len(by_right)),
        n,
    )


def _ari_from_contingency(cells, row_sums, col_sums, n):
    """The index itself, from a contingency table's cells and margins."""

    def pairs(x):
        return x * (x - 1) / 2

    agree = pairs(np.asarray(cells, dtype=float)).sum()
    by_left = pairs(np.asarray(row_sums, dtype=float)).sum()
    by_right = pairs(np.asarray(col_sums, dtype=float)).sum()
    total = pairs(float(n))

    expected = by_left * by_right / total
    largest = (by_left + by_right) / 2
    if largest == expected:
        return 1.0
    return float((agree - expected) / (largest - expected))


class _CompareTally:
    """`AssignmentComparer.summary`'s numbers, foldable one FOV at a time.

    Every metric summary reports is either a count, which adds, or is derived
    from a table whose size follows the *result* rather than the data: the
    contingency of (left cell, right cell) pairings behind the Rand index, the
    set of cells each side used, and the two cell-by-gene tallies. So a whole
    run can be folded in a FOV at a time and still give exactly the numbers a
    single pass over every transcript at once would -- which is what the
    equivalence is worth checking, and what `AssignmentComparer` tests by
    running both ways over the same FOVs.

    The one thing that does not add is the median per-cell Jaccard, and it does
    not have to: it is computed at the end from the accumulated per-cell counts.
    """

    def __init__(self, gene_col_name):
        self.gene_col_name = gene_col_name
        self.fovs = set()
        self.n_transcripts = 0
        self.n_placed_left = 0
        self.n_placed_right = 0
        self.n_placed_both = 0
        self.n_placed_neither = 0
        self.n_agree = 0
        self.n_either = 0
        self.pairs = collections.Counter()
        self.cells_left = set()
        self.cells_right = set()
        # cell -> gene -> count, one per side, plus how many transcripts both
        # sides gave each cell
        self.cxg_left = collections.defaultdict(collections.Counter)
        self.cxg_right = collections.defaultdict(collections.Counter)
        self.shared = collections.Counter()
        self.moved = None

    def add(self, df):
        """Fold one FOV's side-by-side frame in."""
        left = AssignmentComparer._cells(df["left"])
        right = AssignmentComparer._cells(df["right"])
        placed_l, placed_r = left != None, right != None  # noqa: E711
        both = placed_l & placed_r
        agree = both & (left == right)

        self.fovs.update(df["fov"].unique())
        self.n_transcripts += len(df)
        self.n_placed_left += int(placed_l.sum())
        self.n_placed_right += int(placed_r.sum())
        self.n_placed_both += int(both.sum())
        self.n_placed_neither += int((~placed_l & ~placed_r).sum())
        self.n_agree += int(agree.sum())
        self.n_either += int((placed_l | placed_r).sum())

        self.pairs.update(zip(left[both], right[both]))
        self.cells_left.update(left[placed_l])
        self.cells_right.update(right[placed_r])
        self.shared.update(left[agree])

        genes = df[self.gene_col_name].to_numpy(dtype=object)
        for cells, placed, into in (
            (left, placed_l, self.cxg_left),
            (right, placed_r, self.cxg_right),
        ):
            for cell, gene in zip(cells[placed], genes[placed]):
                into[cell][gene] += 1

        if "og" in df.columns:
            self._add_movement(df, left, right)
        return self

    def _add_movement(self, df, left, right):
        """Fold in how far each side moved off the original segmentation."""
        og_l = AssignmentComparer._cells(df["og"])
        og_r = (
            AssignmentComparer._cells(df["og_right"])
            if "og_right" in df.columns
            else og_l
        )
        moved_l = left != og_l
        moved_r = right != og_r
        both = moved_l & moved_r
        if self.moved is None:
            self.moved = dict.fromkeys(
                ("og_differs", "left", "right", "both", "either", "agree"), 0
            )
        self.moved["og_differs"] += int((og_l != og_r).sum())
        self.moved["left"] += int(moved_l.sum())
        self.moved["right"] += int(moved_r.sum())
        self.moved["both"] += int(both.sum())
        self.moved["either"] += int((moved_l | moved_r).sum())
        self.moved["agree"] += int((left[both] == right[both]).sum())

    # ----------------------------------------------------------------- #
    # what the accumulators add up to                                    #
    # ----------------------------------------------------------------- #

    def per_cell(self):
        """The per-cell comparison, from the accumulated tallies."""
        cells = sorted(self.cells_left | self.cells_right, key=str)
        if not cells:
            return pd.DataFrame(
                columns=["n_left", "n_right", "n_shared", "jaccard_trs", "jaccard_cxg"]
            )
        rows = []
        for cell in cells:
            left = self.cxg_left.get(cell, {})
            right = self.cxg_right.get(cell, {})
            n_left = sum(left.values())
            n_right = sum(right.values())
            n_shared = self.shared.get(cell, 0)
            union = n_left + n_right - n_shared
            lo = hi = 0
            for gene in set(left) | set(right):
                a, b = left.get(gene, 0), right.get(gene, 0)
                lo += min(a, b)
                hi += max(a, b)
            rows.append(
                (
                    n_left,
                    n_right,
                    n_shared,
                    n_shared / union if union > 0 else np.nan,
                    lo / hi if hi > 0 else np.nan,
                )
            )
        out = pd.DataFrame(
            rows,
            index=pd.Index(cells, name="cell_id"),
            columns=["n_left", "n_right", "n_shared", "jaccard_trs", "jaccard_cxg"],
        )
        return out.sort_values("jaccard_trs")

    def cxg_jaccard(self):
        """The two cell-by-gene tallies compared, entry by entry."""
        if not self.cxg_left or not self.cxg_right:
            return {"binary": float("nan"), "weighted": float("nan")}
        cells = self.cells_left | self.cells_right
        genes = set()
        shared = either = 0
        lo = hi = 0
        for cell in cells:
            left = self.cxg_left.get(cell, {})
            right = self.cxg_right.get(cell, {})
            genes.update(left)
            genes.update(right)
            for gene in set(left) | set(right):
                a, b = left.get(gene, 0), right.get(gene, 0)
                either += 1
                shared += 1 if a > 0 and b > 0 else 0
                lo += min(a, b)
                hi += max(a, b)
        return {
            "binary": float(shared / either) if either else float("nan"),
            "weighted": float(lo / hi) if hi else float("nan"),
            "n_cells": len(cells),
            "n_genes": len(genes),
        }

    def result(self):
        """The dict `AssignmentComparer.summary` returns."""
        both = self.n_placed_both
        out = {
            "n_fovs": len(self.fovs),
            "n_transcripts": self.n_transcripts,
            "n_placed_left": self.n_placed_left,
            "n_placed_right": self.n_placed_right,
            "n_placed_both": both,
            "n_placed_neither": self.n_placed_neither,
            "n_agree": self.n_agree,
            "n_disagree": both - self.n_agree,
            "agreement": float(self.n_agree / both) if both else float("nan"),
            "jaccard_transcripts": (
                float(self.n_agree / self.n_either) if self.n_either else float("nan")
            ),
            "adjusted_rand_index": (
                adjusted_rand_index_from_pairs(self.pairs)
                if both > 1
                else float("nan")
            ),
        }

        union = self.cells_left | self.cells_right
        out["n_cells_left"] = len(self.cells_left)
        out["n_cells_right"] = len(self.cells_right)
        out["jaccard_cells"] = (
            len(self.cells_left & self.cells_right) / len(union)
            if union
            else float("nan")
        )
        out.update({f"cxg_jaccard_{k}": v for k, v in self.cxg_jaccard().items()})

        cells = self.per_cell()
        out["per_cell_jaccard_mean"] = float(cells["jaccard_trs"].mean())
        out["per_cell_jaccard_median"] = float(cells["jaccard_trs"].median())
        out["n_cells_unchanged"] = int((cells["jaccard_trs"] == 1).sum())

        if self.moved is not None:
            m = self.moved
            out.update(
                {
                    "n_og_differs": m["og_differs"],
                    "n_moved_left": m["left"],
                    "n_moved_right": m["right"],
                    "n_moved_both": m["both"],
                    "jaccard_moved": (
                        float(m["both"] / m["either"]) if m["either"] else float("nan")
                    ),
                    "agreement_on_moved": (
                        float(m["agree"] / m["both"]) if m["both"] else float("nan")
                    ),
                }
            )
        return out


class AssignmentComparer:
    """Compare two assignments of the same transcripts, transcript by transcript.

    The two are named by their columns on the transcript tables -- the output of
    two `evaluate_all_overlapping_regions` runs under different parameters, say,
    or a run against the segmentation it started from (`og_cell`). By default
    both columns are read from the same sdata; pass `right_sdata` to read the
    second from a different object, which is how a pre-refactor run loaded by
    `LegacyRunLoader` is compared against a current one.

    sdata: the SpatialData holding the left column, or a `SoftAssigner` already
        built over it. Given an object or path, a read-only assigner
        (`save_to_disk=False`) is built over it -- nothing here writes.
    left, right: the assignment columns to compare.
    right_sdata: where to read `right` from, if not the same object.
    sel_fovs: which FOVs to compare. Defaults to every FOV both sides carry the
        relevant column for.
    gene_col_name: the transcript table's gene column, for the cell-by-gene
        matrices.
    og_col: the column holding the segmentation both runs started from, used for
        the "what moved" metrics. Skipped when it is not on the table.

    The metrics come in three groups, all available per FOV as well as overall:

    - agreement, from `summary`: how many transcripts each side placed, how often
      they placed them in the same cell, and `adjusted_rand_index`, which asks
      whether the two agree on *which transcripts belong together* regardless of
      what the cells are called.
    - movement, also from `summary`: of the transcripts each side moved away from
      the original segmentation, how much those two sets overlap, and how often
      the two agree on where a moved transcript went. Each side is measured
      against the `og_col` of the object it came from, since two runs that
      blurred differently do not share a baseline; `n_og_differs` counts where
      the two baselines disagree, and `report` says so when they do.
    - composition, from `cxg_jaccard` and `per_cell`: the two cell-by-gene
      matrices built from the assignments, compared entry by entry.

    A low `jaccard_moved` next to a high `agreement` is the usual surprise, and
    it is normally real rather than a quirk: only a fraction of a percent of
    transcripts move at all, so the movement metrics are measured over a tiny
    population while `agreement` is diluted across everything both runs placed.
    `candidate_agreement()` says whether the two runs were even offered the same
    cells per transcript, which is the usual underlying cause.
    """

    def __init__(
        self,
        sdata,
        left,
        right,
        right_sdata=None,
        sel_fovs=None,
        gene_col_name="gene",
        og_col="og_cell",
        stream=None,
    ):
        self.left_asgn = self._as_assigner(sdata)
        self.right_asgn = (
            self.left_asgn if right_sdata is None else self._as_assigner(right_sdata)
        )
        self.left = left
        self.right = right
        self.gene_col_name = gene_col_name
        self.og_col = og_col
        self.stream = self._resolve_stream(stream)
        self.fovs = self._resolve_fovs(sel_fovs)
        self._transcripts = None

    @staticmethod
    def _as_assigner(source):
        """A read-only assigner over `source`, unless it is already one.

        A `LegacyFovView` is passed through untouched: it already answers the
        questions an assigner does, one FOV at a time, which is the whole point
        of handing one in.
        """
        if isinstance(source, (SoftAssigner, LegacyFovView)):
            return source
        if isinstance(source, LegacyRunLoader):
            # the loader itself means "read this run", and reading it a FOV at a
            # time is the only way that scales, so take the streaming view of it
            return source.streaming()
        return SoftAssigner(source, save_to_disk=False)

    def _resolve_stream(self, stream):
        """Whether to visit the FOVs or gather them.

        Streaming is automatic when either side holds one FOV at a time, since
        gathering is exactly what such a side cannot do -- `transcripts()` over a
        whole legacy run would read every CSV in the directory and keep them all.
        """
        if stream is not None:
            return bool(stream)
        return any(
            isinstance(side, LegacyFovView)
            for side in (self.left_asgn, self.right_asgn)
        )

    def _resolve_fovs(self, sel_fovs):
        """The FOVs to compare: those both sides can answer for."""
        if sel_fovs is None:
            candidates = [
                f
                for f in self.left_asgn.get_complete_fovs(column=self.left)
                if self.right_asgn.has_column(f, self.right)
            ]
            if not candidates:
                raise ValueError(
                    f"No FOV carries both {self.left!r} and {self.right!r}. "
                    "Check the column names, or name the FOVs with sel_fovs."
                )
            return candidates

        missing = []
        for f in sel_fovs:
            for side, asgn, column in (
                ("left", self.left_asgn, self.left),
                ("right", self.right_asgn, self.right),
            ):
                if not asgn.has_column(str(f), column):
                    missing.append((str(f), side, asgn, column))
        if missing:
            raise ValueError(self._missing_message(missing))
        return [str(f) for f in sel_fovs]

    # columns every transcript table carries, which are never an assignment
    NOT_ASSIGNMENTS = frozenset(
        {
            "index", "x", "y", "z", "global_x", "global_y", "global_z",
            "gene", "cell_ids", "fov", "barcode_id", "transcript_id",
        }
    )

    @classmethod
    def assignment_columns(cls, asgn, fov):
        """The columns of a FOV's transcript table that name a cell per transcript.

        Everything that is not a coordinate, an id, the gene or the `cell_ids`
        scores, and not one of the `{column}_type` companions -- so what is left
        is what can be compared. A streaming side answers from the CSV header,
        without loading the FOV to be told.
        """
        fov = str(fov)
        if isinstance(asgn, LegacyFovView):
            names = asgn.loader.columns(fov)
        else:
            key = asgn._points_key(fov)
            if key not in asgn.sdata.points:
                return []
            names = list(asgn.sdata.points[key].columns)
        return [
            c
            for c in names
            if c not in cls.NOT_ASSIGNMENTS and not str(c).endswith("_type")
        ]

    @classmethod
    def _missing_message(cls, missing):
        """Say which side is missing which column, and what it does have."""
        lines = []
        for fov, side, asgn, column in missing:
            lines.append(f"fov {fov}: the {side} side has no {column!r} column.")

            # the usual mix-up: naming a table that a column's results were
            # stored in, rather than the column itself
            tables = getattr(getattr(asgn, "_asgn", asgn), "sdata", None)
            tables = getattr(tables, "tables", {}) if tables is not None else {}
            for prefix in ("assigned_trs_", "seg_is_default_"):
                if column.startswith(prefix) and column in tables:
                    lines.append(
                        f"  {column!r} is a table in that object, not a column. "
                        f"The column it records is {column[len(prefix):]!r}."
                    )
                    break

            available = cls.assignment_columns(asgn, fov)
            lines.append(
                f"  columns that could be compared there: {available}"
                if available
                else "  that FOV's table has no assignment columns at all."
            )
        return "\n".join(lines)

    # ----------------------------------------------------------------- #
    # the transcript-level table everything else is derived from         #
    # ----------------------------------------------------------------- #

    def iter_frames(self, desc="comparing"):
        """Yield `(fov, frame)` one FOV at a time, keeping none of them.

        The spine of every metric here. Each frame is built, handed over, and
        dropped before the next FOV is read, so what a comparison costs follows
        the largest single FOV and the size of the answer, not the number of
        FOVs -- which is what makes a 294-FOV legacy directory comparable at all.
        """
        for fov in tqdm(self.fovs, desc=desc):
            yield fov, self._fov_frame(fov)

    def transcripts(self):
        """One row per transcript: where each side put it, and where it started.

        Columns: `fov`, the gene, `left`, `right` and (when available) `og`,
        holding cell ids with `SoftAssigner.MISSING` for "not assigned".

        **This is the one thing here that holds every FOV at once.** The metrics
        do not use it -- they fold the FOVs in one at a time -- so reach for it
        to look at the rows themselves, over a few FOVs. In streaming mode the
        result is not cached, since keeping it is what streaming is avoiding;
        `disagreements()` is the bounded way to get at the interesting rows.
        """
        if self._transcripts is not None:
            return self._transcripts

        frames = [frame for _, frame in self.iter_frames("reading transcripts")]
        out = pd.concat(frames)
        if not self.stream:
            self._transcripts = out
        return out

    def _fov_frame(self, fov):
        """One FOV's side-by-side table, indexed by transcript id."""
        left = self.left_asgn.get_transcripts(fov).set_index("index")
        if self.right_asgn is self.left_asgn:
            right = left
        else:
            right = self.right_asgn.get_transcripts(fov).set_index("index")

        out = pd.DataFrame(index=left.index)
        out.insert(0, "fov", str(fov))
        out[self.gene_col_name] = left[self.gene_col_name].astype(object)
        out["left"] = self._labels(left, self.left)
        out["right"] = self._labels(right, self.right).reindex(out.index)
        if self.og_col in left.columns:
            out["og"] = self._labels(left, self.og_col)
        # each object records the segmentation its own run started from, and two
        # runs blurred with different parameters do not share one. Measuring both
        # sides' movement against the left object's baseline would credit the
        # right side with every place the two baselines simply disagree.
        if self.right_asgn is not self.left_asgn and self.og_col in right.columns:
            out["og_right"] = self._labels(right, self.og_col).reindex(out.index)
        return out

    @staticmethod
    def _cells(series):
        """One assignment column as a plain object array, absent as None.

        Comparisons are done on these rather than on the columns themselves:
        `!=` between nullable columns propagates NA, so a transcript that one
        side placed and the other did not would drop out of the comparison
        instead of counting as a difference.
        """
        return series.to_numpy(dtype=object, na_value=None)

    def _labels(self, df, column):
        """One assignment column, with every spelling of absent collapsed."""
        single = df[[column]].copy()
        SoftAssigner.normalize_labels(single, column)
        return single[column]

    # ----------------------------------------------------------------- #
    # cell-by-gene                                                       #
    # ----------------------------------------------------------------- #

    def cell_by_gene(self, side, df=None):
        """The cell-by-gene counts one side's assignment implies.

        Tallied straight from the transcripts, so it needs no segmentation and
        works against a faked sdata that has none. `SoftAssigner.generate_cxg_table`
        is the equivalent that also measures each cell.

        Given no frame, the FOVs are read one at a time and tallied as they go:
        the matrix is the size of the result, not of the transcripts behind it.
        """
        if df is not None:
            placed = df[df[side].notna()]
            if not len(placed):
                return pd.DataFrame()
            return pd.crosstab(placed[side], placed[self.gene_col_name])

        counted = getattr(self.tally(), f"cxg_{side}")
        if not counted:
            return pd.DataFrame()
        out = pd.DataFrame.from_dict(counted, orient="index").fillna(0).astype(int)
        return out.sort_index(key=lambda i: i.map(str)).sort_index(axis=1)

    @staticmethod
    def _align(left, right):
        """Both matrices over the union of their cells and genes, missing as 0."""
        cells = left.index.union(right.index)
        genes = left.columns.union(right.columns)
        a = left.reindex(index=cells, columns=genes).fillna(0).to_numpy(dtype=float)
        b = right.reindex(index=cells, columns=genes).fillna(0).to_numpy(dtype=float)
        return a, b, cells, genes

    def cxg_jaccard(self, df=None):
        """Jaccard index between the two cell-by-gene matrices.

        Two readings of the same question, both returned:

        - `binary`: over which (cell, gene) pairs have any count at all --
          |shared| / |either|. Says whether the same genes turn up in the same
          cells, ignoring how many.
        - `weighted`: the count-aware form, sum(min) / sum(max) over every entry.
          Says how much of the total signal is placed identically, so a cell
          whose count moved from 9 to 10 barely registers while one that moved
          from 9 to 0 does.

        A matrix is compared over the union of both sides' cells and genes, so a
        cell only one side produced counts fully against the score.
        """
        return self.tally(df).cxg_jaccard()

    def per_cell(self, df=None):
        """Per-cell comparison: how many transcripts each side gave it, and how
        much of its profile the two agree on.

        Columns:
          `n_left`/`n_right`  transcripts the cell was given by each side
          `n_shared`          transcripts both sides gave it
          `jaccard_trs`       n_shared / (n_left + n_right - n_shared), so 1.0 is
                              a cell whose contents did not change at all
          `jaccard_cxg`       the weighted cell-by-gene Jaccard for this cell's
                              row, which differs from `jaccard_trs` when the
                              transcripts that moved were interchangeable with
                              the ones that arrived
        """
        return self.tally(df).per_cell()

    # ----------------------------------------------------------------- #
    # the headline numbers                                               #
    # ----------------------------------------------------------------- #

    def tally(self, df=None):
        """The folded-up `_CompareTally` behind every metric.

        Given a frame, folds just that one. Given nothing, walks the FOVs and
        folds each in turn, which is the same arithmetic over one FOV of memory.
        """
        acc = _CompareTally(self.gene_col_name)
        if df is not None:
            return acc.add(df)
        for _, frame in self.iter_frames("comparing"):
            acc.add(frame)
        return acc

    def summary(self, df=None):
        """Every metric over the compared FOVs at once, as a dict.

        Pass a frame to summarise just it; pass nothing and the FOVs are read one
        at a time and folded together, which gives the same numbers without ever
        holding more than one.
        """
        return self.tally(df).result()

    def per_fov(self):
        """`summary` for each compared FOV, as a dataframe indexed by FOV.

        One FOV resident at a time, whatever the comparison covers -- and each
        row is tallied on its own, so nothing accumulates across them either.
        """
        rows = {}
        for fov, frame in self.iter_frames("comparing per fov"):
            rows[fov] = _CompareTally(self.gene_col_name).add(frame).result()
        return pd.DataFrame(rows).T

    def candidate_agreement(self):
        """How often the two sides' blurs offered a transcript the same cells.

        The first thing to check when `jaccard_moved` is low while `agreement` is
        high. A transcript's ambiguous region is its set of candidate cells, so
        two runs whose `cell_ids` differ are not solving the same problem for it
        -- and the difference shows up almost entirely in the movement metrics,
        because a transcript only one side has candidates for can only ever be
        moved by that side. `agreement` cannot see it at all: that is measured
        over the transcripts both sides placed, which excludes these.

        Blurring with different `max_dist` is the usual reason. Note that
        `n_og_differs` will not catch it: `og_cell` is the cell whose polygon
        contains the transcript, which a wider search radius does not change.

        Parses `cell_ids` on both sides, so it is not part of `summary`.

        Counted a FOV at a time, like the rest: only the running totals are
        kept, never the parsed candidate sets.

        returns: a dict of counts, or None when a side has no `cell_ids`.
        """
        left = self.left_asgn
        right = self.right_asgn
        totals = dict.fromkeys(
            ("n", "same", "sum_left", "sum_right", "only_left", "only_right"), 0
        )
        for fov in tqdm(self.fovs, desc="comparing candidates"):
            if not (
                left.has_column(fov, "cell_ids")
                and right.has_column(fov, "cell_ids")
            ):
                return None
            l_df = left.get_transcripts(fov).set_index("index")
            r_df = right.get_transcripts(fov).set_index("index").reindex(l_df.index)
            l_regions = np.asarray(self._regions(l_df["cell_ids"]), dtype=object)
            r_regions = np.asarray(self._regions(r_df["cell_ids"]), dtype=object)
            n_left = np.fromiter(map(len, l_regions), dtype=int, count=len(l_regions))
            n_right = np.fromiter(map(len, r_regions), dtype=int, count=len(r_regions))

            totals["n"] += len(l_df)
            totals["same"] += int((l_regions == r_regions).sum())
            totals["sum_left"] += int(n_left.sum())
            totals["sum_right"] += int(n_right.sum())
            totals["only_left"] += int(((n_left > 0) & (n_right == 0)).sum())
            totals["only_right"] += int(((n_right > 0) & (n_left == 0)).sum())

        n = totals["n"]
        if not n:
            return None
        return {
            "n_transcripts": n,
            "n_same_candidates": totals["same"],
            "frac_same_candidates": float(totals["same"] / n),
            "mean_candidates_left": float(totals["sum_left"] / n),
            "mean_candidates_right": float(totals["sum_right"] / n),
            "n_only_left_has_candidates": totals["only_left"],
            "n_only_right_has_candidates": totals["only_right"],
        }

    @staticmethod
    def _regions(series):
        """Each transcript's candidate cells, as a sorted tuple."""
        out = []
        for raw in series.to_numpy(dtype=object):
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                out.append(())
            else:
                out.append(tuple(sorted(parse_cell_ids(raw))))
        return out

    def disagreements(self):
        """Just the transcripts the two sides placed differently.

        The rows behind `n_disagree`, for looking at what actually moved. Built a
        FOV at a time and only the differing rows kept, so this stays small even
        where the full table would not fit: two runs of the same pipeline
        disagree about a handful of transcripts in a million.
        """
        kept = []
        for _, df in self.iter_frames("finding disagreements"):
            both = df["left"].notna() & df["right"].notna()
            kept.append(df[both & (df["left"] != df["right"])])
        return pd.concat(kept) if kept else pd.DataFrame()

    def report(self):
        """`summary` as a few lines of text, for printing."""
        s = self.summary()
        lines = [
            f"{self.left!r} vs {self.right!r} over {s['n_fovs']} fov(s), "
            f"{s['n_transcripts']} transcripts",
            f"  placed: {s['n_placed_left']} left, {s['n_placed_right']} right, "
            f"{s['n_placed_both']} both",
            f"  agreement (of those placed by both): {s['agreement']:.4f} "
            f"({s['n_disagree']} differ)",
            f"  adjusted rand index: {s['adjusted_rand_index']:.4f}",
            f"  cell-by-gene jaccard: {s['cxg_jaccard_weighted']:.4f} weighted, "
            f"{s['cxg_jaccard_binary']:.4f} binary",
            f"  per-cell jaccard: {s['per_cell_jaccard_mean']:.4f} mean, "
            f"{s['n_cells_unchanged']} of {s['cxg_jaccard_n_cells']} cells unchanged",
        ]
        if "n_moved_left" in s:
            lines.append(
                f"  moved off the original segmentation: {s['n_moved_left']} left, "
                f"{s['n_moved_right']} right, jaccard {s['jaccard_moved']:.4f}, "
                f"agreeing on {s['agreement_on_moved']:.4f} of the shared ones"
            )
            if s["n_og_differs"]:
                lines.append(
                    f"  note: the two started from different segmentations "
                    f"({s['n_og_differs']} of {s['n_transcripts']} transcripts sit "
                    "in a different cell before either run), so the two movement "
                    "counts are not measured against the same baseline"
                )
        return "\n".join(lines)


class LegacyRunLoader:
    """A SpatialData faked from a pre-refactor run's output directory.

    Before the restructure a run's results were loose files in a `complete_loc`
    directory rather than elements in a store:

      `fov_{fov}_cellids.csv`                    the transcript table, carrying
                                                 `cell_ids` and any assignment
                                                 columns a run added
      `cxg_adata_{stamp}[_resegmented_{col}].h5ad`  the cell-by-gene result
      `overlap_eval_{col}_fov_{fov}.pydict`      `{cell id: [transcript ids]}`,
                                                 what the new code keeps as the
                                                 `assigned_trs_{col}` table
      `delta_tallies_{col}_fov_{fov}.pydict`     per-cell tallies

    `to_sdata()` reads those into an in-memory `SpatialData` in the layout the
    current code expects, so an old run can be handed to `AssignmentComparer`,
    or to a `SoftAssigner` for anything that only needs transcripts. The object
    is not backed by a store and is never written to; there are no labels or
    images in it, so the steps that read a segmentation -- `blur_fov`,
    `generate_cxg_table`'s cell measurements -- are not available on it. Call
    `sdata.write(path)` yourself if you want it on disk.

    Note that `delta_tallies_*.pydict` files hold the assignments rather than the
    tallies: the code that wrote them passed the wrong variable. They are read
    back as what they contain, not as what their name says.
    """

    CSV_GLOB = "fov_*_cellids.csv"
    CSV_RE = re.compile(r"fov_(?P<fov>\w+)_cellids\.csv$")
    PYDICT_RE = re.compile(r"^(?P<kind>overlap_eval|delta_tallies)_(?P<col>.+)_fov_(?P<fov>\w+)\.pydict$")

    def __init__(self, complete_loc, gene_col_name="gene"):
        """complete_loc: the output directory of a pre-refactor run."""
        self.complete_loc = Path(complete_loc)
        if not self.complete_loc.is_dir():
            raise NotADirectoryError(f"No such output directory: {complete_loc}")
        self.gene_col_name = gene_col_name
        self._headers = {}

    def fovs(self):
        """Every FOV the directory has a transcript table for, sorted."""
        found = []
        for path in self.complete_loc.glob(self.CSV_GLOB):
            match = self.CSV_RE.search(path.name)
            if match:
                found.append(match.group("fov"))
        return sorted(found)

    def csv_path(self, fov):
        return self.complete_loc / f"fov_{fov}_cellids.csv"

    def read_fov(self, fov):
        """One FOV's transcript table, as a dataframe with an `index` column."""
        df = pd.read_csv(self.csv_path(fov))
        unnamed = [c for c in df.columns if re.match(r"^Unnamed: \d+$", str(c))]
        if "index" not in df.columns and unnamed:
            # written with the transcript id as the frame's index, so it comes
            # back as the first, unnamed column
            df = df.rename(columns={unnamed[0]: "index"})
            unnamed = unnamed[1:]
        return df.drop(columns=unnamed)

    def columns(self, fov):
        """One FOV's CSV header, without reading a single row of it.

        Cached, since resolving which FOVs can answer for a column asks this of
        every FOV in the directory and the answer cannot change under us.
        """
        fov = str(fov)
        if fov not in self._headers:
            header = pd.read_csv(self.csv_path(fov), nrows=0).columns
            names = [str(c) for c in header]
            if "index" not in names:
                # the transcript id was written as the frame's index, so it
                # comes back as the first unnamed column -- read_fov renames it
                unnamed = [c for c in names if re.match(r"^Unnamed: \d+$", c)]
                if unnamed:
                    names = ["index"] + [c for c in names if c != unnamed[0]]
            self._headers[fov] = names
        return self._headers[fov]

    def has_column(self, fov, column):
        """True if a FOV's CSV carries `column`, from its header alone.

        The counterpart of `SoftAssigner.has_column`, which reads a parquet
        schema; here it keeps `AssignmentComparer` from having to open a 200MB
        CSV just to find out whether it is worth opening.
        """
        try:
            return column in self.columns(fov)
        except FileNotFoundError:
            return False

    def assignment_columns(self, fov=None):
        """The assignment columns a FOV's legacy CSV carries.

        These are the names to hand to `AssignmentComparer`. Note that they are
        not the names of the tables `to_sdata` builds: the pydict for a column is
        stored as the table `assigned_trs_{column}`, so the table for
        "thresh5_min0.7" is "assigned_trs_thresh5_min0.7" while the column to
        compare stays "thresh5_min0.7".

        Reads the CSV header only.
        """
        fov = self.fovs()[0] if fov is None else str(fov)
        return [
            c
            for c in self.columns(fov)
            if not re.match(r"^Unnamed: \d+$", str(c))
            and c not in AssignmentComparer.NOT_ASSIGNMENTS
            and not str(c).endswith("_type")
        ]

    def h5ad_files(self):
        """`{file stem: path}` for the cell-by-gene files in the directory."""
        return {path.stem: path for path in sorted(self.complete_loc.glob("*.h5ad"))}

    def _select_tables(self, load_tables):
        """Which h5ad files `to_sdata` was asked for."""
        files = self.h5ad_files()
        if not load_tables:
            return {}
        if load_tables is True:
            return files

        patterns = [load_tables] if isinstance(load_tables, str) else load_tables
        wanted = {}
        for pattern in patterns:
            matched = {
                stem: path
                for stem, path in files.items()
                if stem == pattern or fnmatch.fnmatch(stem, pattern)
            }
            if not matched:
                raise FileNotFoundError(
                    f"No .h5ad matching {pattern!r} in {self.complete_loc}. "
                    f"Available: {sorted(files)}"
                )
            wanted.update(matched)
        return wanted

    def pydicts(self):
        """`{(kind, column, fov): path}` for every pydict in the directory."""
        found = {}
        for path in self.complete_loc.glob("*.pydict"):
            match = self.PYDICT_RE.match(path.name)
            if match:
                found[
                    (match.group("kind"), match.group("col"), match.group("fov"))
                ] = path
        return found

    @staticmethod
    def read_pydict(path):
        """One pydict file, which holds a python repr rather than JSON."""
        with open(path) as handle:
            return ast.literal_eval(handle.read())

    def to_sdata(self, sel_fovs=None, load_tables=False, load_pydicts=True):
        """Build the SpatialData.

        sel_fovs: which FOVs to read. Defaults to every one the directory has.
        load_tables: which `.h5ad` files to read in as tables, named by file
            stem. False by default, because each one is a whole cell-by-gene
            matrix -- 26 files and 19GB in one real run's output directory --
            and a comparison does not need them: `AssignmentComparer` tallies
            its own from the transcripts. True reads every one; a name or list
            of names reads those, matched against the file stem or as a glob
            over it. `h5ad_files()` lists what is there.
        load_pydicts: read the `.pydict` files in as `assigned_trs_{col}` tables,
            in the shape `evaluate_overlapping_regions_single_fov` writes now.
            These are named after the column they record, not instead of it --
            to compare an assignment, pass `AssignmentComparer` the column
            (`"thresh5_min0.7"`), which `assignment_columns` lists, rather than
            the table (`"assigned_trs_thresh5_min0.7"`).
        """
        fovs = self.fovs() if sel_fovs is None else [str(f) for f in sel_fovs]
        if not fovs:
            raise FileNotFoundError(
                f"No {self.CSV_GLOB} files in {self.complete_loc}; this does not "
                "look like a pre-refactor output directory."
            )

        points = {}
        for fov in tqdm(
            fovs, desc="reading legacy transcript tables", disable=len(fovs) < 2
        ):
            points[f"{fov}_points"] = self._points_element(fov)

        selected = self._select_tables(load_tables)
        tables = {
            stem: ad.read_h5ad(path)
            for stem, path in tqdm(
                selected.items(), desc="reading legacy tables", disable=not selected
            )
        }

        sdata = SpatialData(points=points, tables=tables)
        if load_pydicts:
            self._add_pydict_tables(sdata, fovs)
        return sdata

    def fov_sdata(self, fov, load_pydicts=False):
        """One FOV on its own, as a SpatialData.

        What `to_sdata` builds, for a single FOV and nothing else. The whole
        point of it is that the next one can replace it: a legacy directory's
        transcript CSVs run to tens of gigabytes, so the FOVs have to be visited
        rather than gathered.
        """
        return self.to_sdata(sel_fovs=[str(fov)], load_pydicts=load_pydicts)

    def streaming(self, load_pydicts=False):
        """A view of this run that holds one FOV at a time.

        Hand it to `AssignmentComparer` in place of an sdata and the comparison
        reads each FOV as it reaches it and lets go of the one before, instead of
        loading the directory up front. See `LegacyFovView`.
        """
        return LegacyFovView(self, load_pydicts=load_pydicts)

    def _points_element(self, fov):
        """One FOV's transcript table as a points element."""
        df = self.read_fov(fov)
        if "z" not in df.columns and "global_z" in df.columns:
            # the old code indexed mask slices with global_z directly, which is
            # what the `z` column means now
            df["z"] = df["global_z"]

        coordinates = {"x": "x", "y": "y"}
        if "z" in df.columns:
            coordinates["z"] = "z"
        kwargs = {}
        if self.gene_col_name in df.columns:
            kwargs["feature_key"] = self.gene_col_name

        return PointsModel.parse(
            df,
            coordinates=coordinates,
            transformations={
                DatasetFormatter._fov_coordinate_system(fov): Identity(),
                "global": Identity(),
            },
            **kwargs,
        )

    def _add_pydict_tables(self, sdata, fovs):
        """Turn the per-FOV pydicts into the shared `assigned_trs_{col}` tables."""
        wanted = set(fovs)
        found = {k: v for k, v in self.pydicts().items() if k[2] in wanted}
        if not found:
            return

        asgn = SoftAssigner(sdata, save_to_disk=False)
        by_column = {}
        for (kind, column, fov), path in sorted(found.items()):
            name = column if kind == "overlap_eval" else f"{column}_{kind}"
            assigned_trs = self.read_pydict(path)
            by_column.setdefault(name, {})[fov] = asgn.region_rows(
                fov, assigned_trs, {}
            )

        for name, rows in by_column.items():
            asgn.save_region_tables(rows, name, overwrite=True, save=False)


class LegacyFovView:
    """A legacy run seen one FOV at a time, standing in for a `SoftAssigner`.

    `LegacyRunLoader.to_sdata()` reads every FOV's transcript CSV into one
    object, which is fine for a handful and impossible for a whole run -- one
    real output directory holds 294 of them and 57GB of CSV. This offers
    `AssignmentComparer` the same surface an assigner does, but builds the
    assigner for a FOV when it is asked for and drops the one before, so the
    directory is visited rather than gathered and only one FOV is resident.

    The questions the comparison asks *about* the data rather than of it --
    which FOVs exist, which carry a column -- are answered from the CSV headers
    alone, so choosing what to compare reads no rows at all.

    Build one with `LegacyRunLoader.streaming()`. It is read-only, like the
    assigner it stands in for.
    """

    def __init__(self, loader, load_pydicts=False):
        """
        loader: the `LegacyRunLoader` to read through.
        load_pydicts: build each FOV's `assigned_trs_{col}` tables as it is
           loaded. Off by default -- the comparison reads columns, not tables,
           and the pydicts are per-FOV files that would be re-read every visit.
        """
        self.loader = loader
        self.load_pydicts = load_pydicts
        self._fov = None
        self._asgn = None

    def __repr__(self):
        held = "nothing" if self._fov is None else f"fov {self._fov}"
        return (
            f"{type(self).__name__}({self.loader.complete_loc}, "
            f"{len(self.loader.fovs())} fovs, holding {held})"
        )

    # ----------------------------------------------------------------- #
    # answered from the headers, without reading a FOV                   #
    # ----------------------------------------------------------------- #

    def get_all_fovs(self):
        return self.loader.fovs()

    def has_column(self, fov, column):
        return self.loader.has_column(fov, column)

    def get_complete_fovs(self, column="cell_ids"):
        return [f for f in self.get_all_fovs() if self.has_column(f, column)]

    def get_incomplete_fovs(self, column="cell_ids"):
        return [f for f in self.get_all_fovs() if not self.has_column(f, column)]

    def _points_key(self, fov):
        return f"{fov}_points"

    # ----------------------------------------------------------------- #
    # the one FOV currently in hand                                      #
    # ----------------------------------------------------------------- #

    def assigner(self, fov):
        """A `SoftAssigner` over `fov` alone, replacing whichever was held.

        Asking for the FOV already in hand hands it straight back, so a step
        that reads the same FOV twice does not pay for it twice; asking for any
        other drops it first, so two FOVs are never resident at once.
        """
        fov = str(fov)
        if fov != self._fov:
            self.release()
            self._asgn = SoftAssigner(
                self.loader.fov_sdata(fov, load_pydicts=self.load_pydicts),
                save_to_disk=False,
            )
            self._fov = fov
        return self._asgn

    def release(self):
        """Drop the FOV in hand, if any."""
        self._asgn = None
        self._fov = None

    @property
    def sdata(self):
        """The FOV in hand, as a SpatialData -- not the whole run.

        `AssignmentComparer` reaches for this only to say what a FOV's table
        does carry when the column it wanted is missing, which is about the FOV
        it just asked for.
        """
        if self._asgn is None:
            raise RuntimeError(
                "No FOV loaded yet; call assigner(fov) or get_transcripts(fov) "
                "first. This view holds one FOV at a time rather than the whole "
                "run -- LegacyRunLoader.to_sdata() is the one that holds it all."
            )
        return self._asgn.sdata

    def get_transcripts(self, fov):
        return self.assigner(fov).get_transcripts(fov)

    def get_cell_ids(self, fov):
        return self.assigner(fov).get_cell_ids(fov)
