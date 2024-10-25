from tqdm import tqdm
from collections import Counter

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scanpy as sc


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
    def __init__(self, adata_loc=None, adata=None, verbose=True):
        if adata is not None:
            self.adata = adata
        elif adata_loc is not None:
            self.adata = ad.read(adata_loc)
        else:
            Exception("Either adata object or file location must be provided.")
        self.verbose = verbose

    def filter_cells(self, adata=None, **kwargs):
        """
        Run cell filtering.

        If adata is provided, result will be returned.
        Otherwise, operation will be run on internal adata.

        Any valid parameters for scanpy's filter_cells may be provided,
        and they will be executed in order.
        https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.filter_cells.html
        """
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

    def apply_qc(self, adata=None):
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

    def get_celltypes(self, ref_adata, annotation_level):
        # Remove unknown celltypes
        radata = ref_adata[ref_adata.obs[annotation_level] != "Unknown"]

        # Find marker genes for celltypes (using only genes shared with local data)
        radata = radata[:, np.isin(radata.var.index, self.adata.var.index)]
        sc.tl.rank_genes_groups(radata, annotation_level, method="wilcoxon")
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
                f"celltypes present in single-cell reference: {Counter(radata.obs[annotation_level])}"
            )

        self.pos_markers = pos_markers
        self.neg_markers = neg_markers
        self.celltypes = celltypes

    def get_clusters(self, n_pcs=50, n_neighbors=25, res=0.5):
        # Unsupervised clustering (nothing new here, just verifying that we're getting decent looking clusters)
        sc.pp.pca(self.adata, svd_solver="arpack", n_comps=n_pcs, copy=False)
        sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        sc.tl.umap(self.adata)
        sc.tl.leiden(self.adata, resolution=res)

    def plot_raw_clusters(self, only_dotplot=True):
        sc.tl.dendrogram(self.adata, "leiden")
        if not only_dotplot:
            sc.pl.umap(self.adata, color=["leiden"], cmap="tab20")

            sc.pl.dendrogram(self.adata, "leiden")

            plt.figure(figsize=(15, 9), constrained_layout=True, dpi=200)

            for cluster in sorted(set(self.adata.obs["leiden"].astype(int))):
                cluster_adata = self.adata[self.adata.obs["leiden"] == str(cluster)]
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
            fig, ax = sc.pl.dotplot(
                self.adata,
                self.pos_markers[celltype][:20],
                groupby="leiden",
                dendrogram=True,
                title=f"{celltype}: Positive Expression",
            )
            plt.show(block=False)

            fig, ax = sc.pl.dotplot(
                self.adata,
                self.neg_markers[celltype][:20],
                groupby="leiden",
                dendrogram=True,
                title=f"{celltype}: Negative Expression",
            )
            plt.show(block=False)

        plt.show()

    def assign_cluster_celltypes(self, obs_level, assigned_clusters):
        """
        obs_level will be the name of the cell typing obs column in the adata.

        assigned_clusters is expected to be of format
            {"cell_type": [cluster int, cluster int, ...],...}
        """
        num_leiden = len(set(self.adata.obs["leiden"]))
        clusters2types = {k: "Unassigned" for k in range(0, num_leiden)}

        for k1, v1 in clusters2types.items():
            for k2, v2 in assigned_clusters.items():
                for i in v2:
                    clusters2types[int(i)] = k2

        self.adata.obs[obs_level] = [
            clusters2types[int(cluster)] for cluster in self.adata.obs["leiden"]
        ]
        types = list(set(self.adata.obs[obs_level]))
        type2int = {cell_type: x for x, cell_type in enumerate(types)}
        self.adata.obs["int_type"] = [
            type2int[cell_type] for cell_type in self.adata.obs[obs_level]
        ]

    def plot_assigned_clusters(self, obs_level):
        plt.figure(figsize=(15, 9), constrained_layout=True, dpi=200)

        for celltype in sorted(set(self.adata.obs[obs_level])):
            celltype_adata = self.adata[self.adata.obs[obs_level] == celltype]
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
        sc.pl.umap(self.adata, color=[obs_level], cmap="tab20")
        plt.show()

    def pretty_joiner(self, adatas):
        """
        Combines anndata frames while retaining .var
        adatas is expected to be a list of anndata structures
        to be concatenated.
        """
        result = pd.concat(adatas, join="outer")
        all_vars = [x.var for x in adatas[0]]
        all_vars = pd.concat(all_vars)
        all_vars = all_vars[~all_vars.index.duplicated()]
        result.var = all_vars.loc[result.var_names]
        return result
