import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
from spatialdata.transformations import (
    Affine,
    MapAxis,
    Scale,
    Sequence,
    Translation,
    get_transformation,
    set_transformation,
)
import skimage
import skimage.io
import torch
import warnings
import torch.optim as optim
import anndata as ad
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset
from SoftSeg.warp_shapes import warp_shapes, smooth_field_warp, plot_warp_overlay, rasterize_shapes
from SoftSeg.SoftAssigner import SoftAssigner
from tqdm import tqdm


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
                max(top, self.model_stats["accuracy"] + 0.1)
            )
            plt.axvline(x=self.model_stats["accuracy"], color="red")
            plt.show()

    def run(self, focal_types, neighbor_types, n_neighbors=6, plot=True, progress=True):
        self.clean_data()
        self.find_neighbors(n_neighbors)
        self.normalize_data()
        self.train_model(focal_types=focal_types, neighbor_types=neighbor_types, progress=progress)
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
    def __init__(self, in_loc, out_loc):
        self.in_loc = in_loc
        self.out_loc = out_loc
        self.sdata = sd.read_zarr(in_loc)
        warnings.filterwarnings("ignore", message="zarr v3 autosharding will be the default in the next minor release")

    def run(self, amplitude=10, length_scale=200, seed=0):
        warped = warp_shapes(self.sdata, smooth_field_warp(amplitude=amplitude, length_scale=length_scale, seed=seed))
        self.sdata.shapes["cell_borders_warped"] = warped.shapes["cell_borders_warped"]
        self.sdata.shapes["cells_warped"] = warped.shapes["cells_warped"]

        points = self.sdata.points["transcripts"].compute()
        min_x = points["x"].min()
        min_y = points["y"].min()
        max_x = points["x"].max()
        max_y = points["y"].max()
        scale = 1
        self.sdata["rasterized"] = sd.rasterize(
            self.sdata["cells_warped"],
            ["x", "y"],
            min_coordinate=[min_x*scale, min_y*scale],
            max_coordinate=[max_x*scale, max_y*scale],
            target_coordinate_system="global",
            target_unit_to_pixels=scale,
        )
        points["x"] = (points["x"] + (-1 * min_x)) * scale**2  # for some reason the scale is squared??
        points["y"] = (points["y"] + (-1 * min_y)) * scale**2

        mapping = {k: int(x[5:]) for k,x in self.sdata["rasterized"].label_index_to_category.items()}
        mapping[0] = 0
        f = np.vectorize(lambda x: mapping[x])
        im = f(self.sdata["rasterized"].to_numpy()).squeeze()
        new_table = self.sdata.aggregate(
            values="transcripts",
            by="cells_warped",
            value_key="feature_name",
            agg_func="count"
        )

        index_diff = pd.Index(set(self.sdata.tables["counts"].var.index) - set(new_table["table"].var.index))
        new_table.tables["table"].var.index.append(index_diff)
        # note that the following might break if we have warped cells with no transcripts assigned to them
        new_table.tables["table"].obs["celltype"] = self.sdata.tables["counts"].obs["celltype"]
        self.sdata.tables["counts_warped"] = new_table.tables["table"]

        skimage.io.imsave(f"{self.out_loc}cell_borders_warped_labels.tif", im)
        points.to_csv(f"{self.out_loc}/transcripts.csv")
        self.sdata.tables["counts"].write(f"{self.out_loc}truth_cxg.h5ad")
        self.sdata.tables["counts_warped"].write(f"{self.out_loc}warped_cxg.h5ad")
        self.sdata.write(f"{self.out_loc}warped_sdata.zarr", overwrite=True)


class ParamSweeper:
    def __init__(self, csv_loc, im_loc, complete_base_loc, pool_size=5):
        self.complete_base_loc = complete_base_loc
        self.csv_loc = csv_loc
        self.im_loc = im_loc
        self.complete_base_loc = complete_base_loc
        self.pool_size = pool_size

    def blur_fovs(self, min_size=25, max_dist=3.5):
        name = f"size{min_size}_dist{max_dist}"
        self.asgn = SoftAssigner(
            csv_loc=self.csv_loc,
            im_loc=self.im_loc,
            complete_loc=self.complete_base_loc.format(name),
            pool_size=self.pool_size
        )
        self.asgn.blur_fov(0, min_size, max_dist, disable_tqdm=True)

    def process_unit_test(self, adata_loc):
        adata = ad.read_h5ad(adata_loc)
        cats = {
            "celltype": [
                "ct_0",
                "ct_1"
            ]
        }
        adata.X = adata.X.todense()
        adata.obs.index = [str(x) for x in list(range(len(adata)))]
        adata.obs["instance_id"] = [f"cell_{x}" for x in list(range(len(adata)))]
        self.asgn.get_scoring_matrix(adata, cats, normed=False)

    def assign(self, gene_col_name, default_thresh=10, min_thresh=0.8):
        name = f"default{default_thresh}_min{min_thresh}"
        self.asgn.evaluate_overlapping_regions_single_fov(
            "0000",
            assigned_col=f"assigned_{name}",
            gene_col_name=gene_col_name,
            default_thresh=default_thresh,
            min_thresh=min_thresh,
            disable_tqdm=True,
            overwrite=True
        )
        self.tr = pd.read_csv(self.asgn.complete_csv_name.format("0"), index_col=0)

    def rate_unit_test(self, cell_col, type_col):
        for col in ["og_type", "og_cell", cell_col, type_col]:
            self.tr[col] = self.tr[col].fillna("none")

        type_acc = sum(self.tr["celltype_sim"] == self.tr[type_col])/len(self.tr)
        self.tr["transformed"] = self.tr[cell_col].apply(lambda y: f"cell_{int(y) + 1}" if y != "none" else "none")
        cell_acc = sum(self.tr["cell_id"] == self.tr["transformed"])/len(self.tr)
        return type_acc, cell_acc

    def run_unit_sweep(self, gene_col_name, unit_adata_loc, min_size=25, max_dist=3.5, default_thresh=10, min_thresh=0.8):
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

        total_len = len(sizes)*len(dists)*len(threshs)*len(mins)
        pbar = tqdm(total=total_len)

        for size in sizes:
            for dist in dists:
                self.blur_fovs(size, dist)
                self.process_unit_test(unit_adata_loc)

                for thresh in threshs:
                    for min_t in mins:
                        pbar.update(1)
                        entry = {
                            "min_thresh": min_t,
                            "default_thresh": thresh,
                            "max_dist": dist,
                            "min_size": size,
                        }
                        name = f"default{thresh}_min{min_t}"
                        self.assign(gene_col_name, thresh, min_t)
                        type_acc, cell_acc = self.rate_unit_test(f"assigned_{name}", f"assigned_{name}_type")
                        entry["type_acc"] = type_acc
                        entry["cell_acc"] = cell_acc
                        results.append(entry)

        pbar.close()
        return results
