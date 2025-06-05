from collections import Counter
from datetime import datetime

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
import scanpy as sc
from tqdm import tqdm


def download_lung_dataset(filepath):
    url = "https://datasets.cellxgene.cziscience.com/b351804c-293e-4aeb-9c4c-043db67f4540.h5ad"
    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True)

    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(filepath, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")


class CellTypeAssigner:
    def __init__(
        self, ref_levels, ann_levels, adata_loc=None, adata=None, verbose=True
    ):
        self.ref_levels = ref_levels  # list of columns to look at for celltype labels in ref data
        self.ann_levels = ann_levels  # list of columns to assign celltypes target adata
        if adata is not None:
            self.adata = adata
        elif adata_loc is not None:
            self.adata = ad.read(adata_loc)
            self.adata_loc = adata_loc
        else:
            Exception("Either adata object or file location must be provided.")
        self.verbose = verbose

    def save_annotated_adata(self):
        if self.adata_loc is None:
            print("Requires `adata_loc` to be set.")
        else:
            dat = datetime.today().strftime("%Y%m%d")
            if ".h5ad" in self.adata_loc:
                newloc = "/".join(self.adata_loc.split("/")[:-1])
                filename = f"{newloc}/cxg_adata_celltypes_{dat}.h5ad"
            else:
                filename = f"{self.adata_loc}/cxg_adata_highlevel_wpca_{dat}.h5ad"
            self.adata.write_h5ad(filename)
            print(f"Saved {filename}")
            return filename

    def load_annotated_adata(self, loc):
        self.adata = ad.read(loc)

    def filter_cells(self, adata=None, **kwargs):
        """
        Run cell filtering.

        If adata is provided, result will be returned.
        Otherwise, operation will be run on internal adata.

        Any valid parameters for scanpy's filter_cells may be provided,
        and they will be executed in order.
        https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.filter_cells.html
        """
        internal = False
        if adata is None:
            internal = True
            adata = self.adata

        for k, v in kwargs.items():
            if self.verbose:
                print(f"#cells before {k} filter: {adata.n_obs}")
            sc.pp.filter_cells(adata, **{k: v})
            if self.verbose:
                print(f"#cells after {k} filter: {adata.n_obs}")

        if internal:
            self.adata = adata
        else:
            return adata

    def filter_genes(self, adata=None, **kwargs):
        """
        Run gene filtering.

        If adata is provided, result will be returned.
        Otherwise, operation will be run on internal adata.

        Any valid parameters for scanpy's filter_genes may be provided,
        and they will be executed in order.
        https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.filter_genes.html
        """
        internal = False
        if adata is None:
            internal = True
            adata = self.adata

        for k, v in kwargs.items():
            if self.verbose:
                print(f"#genes before {k} filter: {adata.n_vars}")
            sc.pp.filter_genes(adata, **{k: v})
            if self.verbose:
                print(f"#genes after {k} filter: {adata.n_vars}")

        if internal:
            self.adata = adata
        else:
            return adata

    def filter_blanks(self, adata=None):
        internal = False
        if adata is None:
            internal = True
            adata = self.adata

        valid_genes = [name for name in adata.var_names if "lank" not in name]
        indicator = np.in1d(adata.var_names, valid_genes)
        if internal:
            self.adata = adata[:, indicator]
        else:
            return adata[:, indicator]

    def apply_qc(self, adata=None):
        internal = False
        if adata is None:
            internal = True
            adata = self.adata

        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mt"], inplace=True, percent_top=None
        )

        if internal:
            self.adata = adata
        else:
            return adata

    def get_celltypes(self, ref_adata, ref_level, target_type=None, ignored=[]):
        # First, subset the reference data if target_type is provided
        radata = ref_adata.copy()
        if target_type is not None:
            radata = radata[
                np.isin(radata.obs[self.ref_levels[ref_level - 1]], [target_type])
            ]
            radata = radata[
                ~np.isin(radata.obs[self.ref_levels[ref_level]], ignored)
            ]

        # Remove unknown celltypes
        radata = radata[radata.obs[self.ref_levels[ref_level]] != "Unknown"]

        # Find marker genes for celltypes (using only genes shared with local data)
        radata = radata[:, np.isin(radata.var.index, self.adata.var.index)]
        sc.tl.rank_genes_groups(radata, self.ref_levels[ref_level], method="wilcoxon")
        # sc.tl.filter_rank_genes_groups(radata, annotation_level)
        sc.tl.filter_rank_genes_groups(radata)
        marker_df = sc.get.rank_genes_groups_df(radata, group=None)
        celltypes = radata.uns["rank_genes_groups"]["names"].dtype.names
        pos_markers = {}
        neg_markers = {}

        pval_thresh = 0.01
        for celltype in celltypes:
            pos_markers[celltype] = list(
                marker_df[
                    (marker_df["group"] == celltype)
                    & (marker_df["scores"] > 0)
                    & (marker_df["pvals_adj"] < pval_thresh)
                ]["names"]
            )
            neg_markers[celltype] = list(
                marker_df[
                    (marker_df["group"] == celltype)
                    & (marker_df["scores"] < 0)
                    & (marker_df["pvals_adj"] < pval_thresh)
                ].sort_values("scores")["names"]
            )

        if self.verbose:
            print(
                f"celltypes present in single-cell reference: {Counter(radata.obs[self.ref_levels[ref_level]])}"
            )

        self.pos_markers = pos_markers
        self.neg_markers = neg_markers
        self.celltypes = celltypes
        self.level = ref_level
        self.target = target_type
        if self.target is not None:
            adata = self.adata.copy()
            adata = adata[adata.obs[self.ann_levels[self.level-1]] == self.target]
            self.sub_adata = adata
        else:
            self.sub_adata = None

    def get_clusters(self, n_pcs=50, n_neighbors=25, res=0.5):
        # if target is defined, only run this on a subset of the data
        if self.target is not None:
            adata = self.sub_adata
        else:
            adata = self.adata

        # Unsupervised clustering (nothing new here, just verifying that we're getting decent looking clusters)
        sc.pp.pca(adata, svd_solver="arpack", n_comps=n_pcs, copy=False)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=res)

    def plot_raw_clusters(self, only_dotplot=True):
        # if target is defined, only run this on a subset of the data
        if self.target is not None:
            adata = self.sub_adata
        else:
            adata = self.adata

        sc.tl.dendrogram(adata, "leiden")
        if not only_dotplot:
            sc.pl.umap(adata, color=["leiden"], cmap="tab20")

            sc.pl.dendrogram(adata, "leiden")

            plt.figure(figsize=(15, 9), constrained_layout=True, dpi=200)

            for cluster in sorted(set(adata.obs["leiden"].astype(int))):
                cluster_adata = adata[adata.obs["leiden"] == str(cluster)]
                plt.scatter(
                    cluster_adata.obs["x_coords"],
                    cluster_adata.obs["y_coords"],
                    label=int(cluster),
                    cmap=matplotlib.colors.ListedColormap(
                        plt.get_cmap("tab20").colors[::2]
                    ),
                    s=1,
                    alpha=1,
                )

            lgnd = plt.legend(fontsize=14)
            for handle in lgnd.legendHandles:
                handle.set_sizes([100])
            # plt.ylim(-1000, 81000)
            # plt.xlim(-2000, 70000)
            plt.show(block=False)

        for celltype in sorted(self.celltypes):
            sc.pl.dotplot(
                adata,
                self.pos_markers[celltype][:20],
                groupby="leiden",
                dendrogram=True,
                title=f"{celltype}: Positive Expression",
            )
            plt.show(block=False)

            sc.pl.dotplot(
                adata,
                self.neg_markers[celltype][:20],
                groupby="leiden",
                dendrogram=True,
                title=f"{celltype}: Negative Expression",
            )
            plt.show(block=False)

        plt.show()

    def assign_cluster_celltypes(self, assigned_clusters):
        """
        obs_level will be the name of the cell typing obs column in the adata.

        assigned_clusters is expected to be of format
            {"cell_type": [cluster int, cluster int, ...],...}
        """
        if self.target is not None:
            adata = self.sub_adata
        else:
            adata = self.adata

        num_leiden = len(set(adata.obs["leiden"]))
        clusters2types = {k: "Unassigned" for k in range(0, num_leiden)}

        for k1, v1 in clusters2types.items():
            for k2, v2 in assigned_clusters.items():
                for i in v2:
                    clusters2types[int(i)] = k2

        obs_level = self.ann_levels[self.level]

        adata.obs[self.ann_levels[self.level]] = [
            clusters2types[int(cluster)] for cluster in adata.obs["leiden"]
        ]
        types = list(set(adata.obs[obs_level]))
        type2int = {cell_type: x for x, cell_type in enumerate(types)}
        adata.obs["int_type"] = [
            type2int[cell_type] for cell_type in adata.obs[obs_level]
        ]

    def plot_assigned_clusters(self, level=None):
        if self.target is not None:
            adata = self.sub_adata
        else:
            adata = self.adata

        plt.figure(figsize=(15, 9), constrained_layout=True, dpi=200)

        if level is None:
            obs_level = self.ann_levels[self.level]
        else:
            obs_level = self.ann_levels[level]
        for celltype in sorted(set(adata.obs[obs_level])):
            celltype_adata = adata[adata.obs[obs_level] == celltype]
            plt.scatter(
                celltype_adata.obs["x_coords"],
                celltype_adata.obs["y_coords"],
                label=celltype,
                cmap="tab20",
                s=0.1,
                alpha=1,
            )

        lgnd = plt.legend(fontsize=14)
        for handle in lgnd.legendHandles:
            handle.set_sizes([100])
        # plt.ylim(-1000, 81000)
        # plt.xlim(-2000, 70000)
        plt.show(block=False)

        # UMAP
        sc.pl.umap(adata, color=[obs_level], cmap="tab20")
        plt.show()

    def merge_target_adata(self):
        # TODO: make this smarter so it modifies the cluster columns?'
        if self.target is not None:
            not_adata = self.adata[self.adata.obs[self.ann_levels[self.level-1]] != self.target]
            self.adata = ad.concat([self.sub_adata, not_adata], join="outer", merge="same")
            self.sub_adata = None
