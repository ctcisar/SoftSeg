import glob
import logging
import random
import warnings
from pathlib import Path
from typing import Any, Mapping, Optional
from typing import Sequence as TypingSequence
from typing import Union

import cv2
import dask.array as da
import matplotlib.pyplot as plt
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
                                Labels3DModel, PointsModel)
from spatialdata.transformations import (Affine, Identity, Sequence,
                                         Translation, get_transformation)
from tqdm.auto import tqdm

from SoftSeg.SoftAssigner import SoftAssigner
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

    def __init__(self, mask_str, seg_str):
        self.mask_str = mask_str
        self.seg_str = seg_str
        self.masks = None
        self.image = None

    def get_range_dict(mask, border=10):
        i = {}
        inds = np.nonzero(mask)
        if len(inds[0]) == 0:
            return None

        if len(inds) == 3:
            i["zs"] = [x for x in range(np.min(inds[0]), np.max(inds[0]) + 1)]
            offset = 1
        else:
            offset = 0

        i["x0"] = np.min(inds[offset]) - border
        i["x1"] = np.max(inds[offset]) + border
        i["y0"] = np.min(inds[offset + 1]) - border
        i["y1"] = np.max(inds[offset + 1]) + border

        return i

    def load_fov(self, fov, exposure_eq=False):
        """
        exposure_eq: if True, will use skimage.exposure method for normaliztion
        """
        self.masks = skimage.io.imread(self.mask_str.format(fov))
        if exposure_eq:
            self.image = skimage.exposure.equalize_hist(
                skimage.io.imread(self.seg_str.format(fov))
            )
        else:
            temp = skimage.io.imread(self.seg_str.format(fov))
            self.image = np.zeros(np.shape(temp))
            for z in range(np.shape(temp)[0]):
                layer = temp[z]
                layer = (layer - np.min(layer)) / np.max(layer)
                self.image[z] = layer

    def mask_from_img(self, ind):
        return (self.masks == ind + 1).astype(np.uint8)

    def cross_to_slice(self, cross):
        # TODO: calculate xmax and ymax from self and then use those for the upper bounds!!
        xmax = np.shape(self.masks)[-2] - 1
        ymax = np.shape(self.masks)[-1] - 1
        zmin = min(cross["zs"])
        zmax = max(cross["zs"]) + 1
        return (
            slice(zmin, zmax),
            slice(max(cross["x0"], 0), min(cross["x1"], xmax)),
            slice(max(cross["y0"], 0), min(cross["y1"], ymax)),
        )

    def plot_subset(
        self,
        target_cell=None,
        highlight_cells=None,
        adata=None,
        highlight_type=None,
        single_channel=None,
    ):
        """
        target_cell: if provided, will zoom in on that cell specifically
        highlight_cells: cell ids listed will be highlighted in red
        adata: anndata with cell ids and other information. Only needed for highlight_types
        highlight_type: cells that match this type will be highlighted in white
        single_channel: if provided, only this channel of the segmentation image will be usd
        """
        if self.masks is None:
            raise Exception("load_fov must be called before performing this operation.")
        if target_cell is not None:
            mask = self.mask_from_img(target_cell)
            cross = SegImagePlotter.get_range_dict(mask, border=100)
            crossection = self.cross_to_slice(cross)
            image_sub = self.image[crossection].copy()
            mask_sub = self.masks[crossection].copy()
        else:
            image_sub = self.image.copy()
            mask_sub = self.masks.copy()

        for z in range(np.shape(mask_sub)[0]):
            contours = []
            contour_color = []
            for c in np.unique(mask_sub[z]):
                if c == 0:
                    continue
                mask = (mask_sub[z] == c).astype(np.uint8)

                contours.append(
                    cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
                )

                if highlight_cells is not None and c - 1 in highlight_cells:
                    color = (1, 0, 0)
                elif (
                    adata is not None
                    and highlight_type is not None
                    and str(c - 1) in adata.obs.index
                    and adata.obs.loc[str(c - 1)]["celltypes"] == highlight_type
                ):
                    color = (1, 1, 1)
                else:
                    color = (0, 0, 0)
                contour_color.append(color)

            this_img = image_sub[z]
            for i in range(len(contours)):
                this_img = cv2.drawContours(
                    this_img, contours[i], 0, contour_color[i], 1
                )
            if single_channel is None:
                plt.imshow(this_img)
            else:
                plt.imshow(this_img[:, :, single_channel])
            plt.show()


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

    def blur_fovs(self, min_size=25, max_dist=3.5):
        """Re-blur every swept FOV, overwriting the `cell_ids` column."""
        self.blur_params = (min_size, max_dist)
        for fov in self.fovs:
            self.asgn.blur_fov(fov, min_size, max_dist, disable_tqdm=True)

    def process_unit_test(self, table_name, cats=None):
        """Build the scoring matrix from a cell-typed reference table.

        `table_name` names a table in the sdata -- where `convert_to_adata` and
        `CellTypeAssigner` leave their results.
        """
        adata = self.sdata.tables[table_name]

        if cats is None:
            cats = {"celltype": ["ct_0", "ct_1"]}

        if hasattr(adata.X, "todense"):
            adata.X = adata.X.todense()
        adata.obs.index = [str(x) for x in range(len(adata))]
        adata.obs["instance_id"] = [f"cell_{x}" for x in range(len(adata))]
        self.asgn.get_scoring_matrix(adata, cats, normed=False)

    def assign(self, gene_col_name, default_thresh=10, min_thresh=0.8):
        """Evaluate overlapping regions for this parameter set.

        Returns the name of the column written, which is also where the result
        now lives in the sdata.
        """
        name = self.param_name(default_thresh, min_thresh)
        for fov in self.fovs:
            self.asgn.evaluate_overlapping_regions_single_fov(
                fov,
                assigned_col=name,
                gene_col_name=gene_col_name,
                default_thresh=default_thresh,
                min_thresh=min_thresh,
                disable_tqdm=True,
                overwrite=True,
            )

        return name

    def _normalize_missing(self, series):
        """One marker for "no cell here", whichever way it was written.

        It reaches the table two ways: an unassigned transcript is null in the
        assignment column, while `evaluate_overlapping_regions_single_fov` writes
        the string "None" into og_cell/og_type/{col}_type. Both have to collapse,
        or a column using the other spelling reaches `int()` in
        `rate_unit_test` -- which is how rating og_cell, as the unmodified
        segmentation's baseline, used to raise
        "invalid literal for int(): 'None'".
        """
        values = series.astype(object)
        values = values.where(~pd.isna(values), self.MISSING)
        return values.replace({"None": self.MISSING, "": self.MISSING})

    def rate_unit_test(self, cell_col, type_col):
        """Accuracy of one assignment against the simulated ground truth.

        Expects the transcript table to carry the truth columns `celltype_sim`
        and `cell_id`, as the simulated datasets do.

        The cell each transcript was given, in the ground truth's own naming, is
        written back to the FOV's points element as `f"{cell_col}_transformed"`
        and saved, so the comparison behind a score stays inspectable in the
        object rather than living on the sweeper.
        """
        type_hits = cell_hits = total = 0

        for fov in self.fovs:
            tr = self.asgn.get_transcripts(fov)

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
            self.asgn.set_transcripts(fov, tr)

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
                self.blur_fovs(size, dist)
                self.process_unit_test(unit_table_name)

                for thresh in threshs:
                    for min_t in mins:
                        pbar.update(1)
                        entry = {
                            "min_thresh": min_t,
                            "default_thresh": thresh,
                            "max_dist": dist,
                            "min_size": size,
                        }
                        name = self.assign(gene_col_name, thresh, min_t)
                        type_acc, cell_acc = self.rate_unit_test(name, f"{name}_type")
                        entry["type_acc"] = type_acc
                        entry["cell_acc"] = cell_acc
                        results.append(entry)

        pbar.close()
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

        valid_z_by_fov: dict = {}
        our_sd = SpatialData()
        if save_loc is not None:
            our_sd.write(Path(save_loc), overwrite=True)

            # ignore uneeded warning when converting dataset
            logging.getLogger("ome_zarr.reader").addFilter(
                lambda r: not r.getMessage().startswith("no parent found for")
            )

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
