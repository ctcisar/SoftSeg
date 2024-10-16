import math
import ast
import random
import collections
import itertools
import time
from collections import Counter
from copy import deepcopy
from datetime import datetime
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import anndata as ad
import cv2
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import psutil
import scanpy as sc
import skimage
import skimage.io
from matplotlib.patches import Circle
from skimage.measure import regionprops
from skimage.segmentation import clear_border

loc = "/media/cecilia/Data/vizgen/LungCancerPatient1/"
loc_compl = "/media/cecilia/Data/vizgen/LungCancerPatient1/nickcomp_3/"
tr_st = f"{loc}fov_{{:04}}_selected_transcripts.csv"
tr_compl = f"{loc}nickcomp_3/fov_{{:04}}_cellids.csv"
im_st = f"{loc}nick_masks_240930/fov_{{:04}}_labels.tif"

# single cell dataset stuff
sc_loc = "/media/cecilia/Data/single_cell/"
version = "hlca_healthy"
annotation_level = "ann_level_1"

fov_count = 384
sigma = 15  # spread on gaussian filter
dilat = 10  # size of dilation
min_size = 30  # masks smaller than this will be removed
thresh = 0.2  # minimum value for a transcript to be considered, non-inclusive
pool_size = 10


def filtSize(im, minSize=None, maxSize=None):
    props = regionprops(im)
    to_del = []
    for region in props:
        if minSize is not None and region.area < minSize:
            to_del.append(region.label)
        if maxSize is not None and region.area > maxSize:
            to_del.append(region.label)

    def will_del(y):
        return 0 if y in to_del else y

    return np.vectorize(will_del)(im)


def filtImage(loc, f, im_st, tr_st, min_size, dilate, sigma, thresh, tr_compl):
    t1 = time.time()
    im = skimage.io.imread(im_st.format(f))
    # filter this before we do anything else to save time...
    im = filtSize(im, minSize=min_size)

    tr = pd.read_csv(tr_st.format(f), index_col=0)
    tr = tr.reset_index()

    if len(tr) > 0:
        print(f"starting fov_{f:04}...")

        if "cell_ids" in tr.keys():
            tr.drop(["cell_ids"], axis=1, inplace=True)

        tr["cell_ids"] = [{} for _ in range(len(tr))]

        for m in np.unique(im):
            if m == 0:  # not a cell
                continue

            # print(f"cell {m}")
            temp_im = im == m

            """
            for i in range(len(temp_im)):
                print(i)
                print(sum(sum(temp_im[i,:,:])))
                plt.imshow(temp_im[i,:,:])
                plt.show()
            """

            def fn(y): return skimage.morphology.binary_dilation(
                y, skimage.morphology.disk(dilate)
            )

            if len(np.shape(temp_im)) == 2:  # two dimensional
                filt = fn(temp_im)
                filt = skimage.filters.gaussian(filt, sigma=sigma)
            else:  # three dimensional (presumably)
                filt = np.array([fn(sl) for sl in temp_im])
                filt = np.array(
                    [skimage.filters.gaussian(sl, sigma=sigma) for sl in filt]
                )

            # print(f"cell {m}, {sum(sum(sum(filt > thresh)))} total eligible pixels")

            """
            for i in range(len(temp_im)):
                print(i)
                #plt.imshow(temp_im[i,:,:])
                #plt.show()
                plt.imshow(filt[i,:,:])
                plt.show()
            """

            if len(np.shape(temp_im)) == 2:
                tr["cell_ids"] = tr.apply(
                    lambda row: (
                        row["cell_ids"]
                        if not filt[row["y"] - 1, row["x"] - 1] > thresh
                        else dict(
                            row["cell_ids"],
                            **{str(m): filt[row["y"] - 1, row["x"] - 1]},
                        )
                    ),
                    axis=1,
                )
            else:
                """
                tr["cell_ids"] = tr.apply(lambda row:
                                          row["cell_ids"]
                                              if not filt[row['global_z'],row['y']-1,row['x']-1] > thresh
                                              else dict(row["cell_ids"], **{str(m): filt[row['global_z'],row['y']-1,row['x']-1]}),
                                          axis=1)
                """
                # can't use zslice 6 for now
                tr["cell_ids"] = tr.apply(
                    lambda row: (
                        row["cell_ids"]
                        if row["global_z"] == 6
                        or (
                            not filt[row["global_z"], row["y"] - 1, row["x"] - 1]
                            > thresh
                        )
                        else dict(
                            row["cell_ids"],
                            **{
                                str(m): filt[
                                    row["global_z"], row["y"] - 1, row["x"] - 1
                                ]
                            },
                        )
                    ),
                    axis=1,
                )
            del filt
            del temp_im
        del im
        tr.to_csv(tr_compl.format(f), index=False)
        print(
            f"saved fov_{f:04}\n\t{len(tr)} transcripts, now unpacking intensities..."
        )

        tr = tr.set_index("index")

        intensities = []
        for i, row in tr.iterrows():
            for k, v in row["cell_ids"].items():
                intensities.append(v)

        print(f"\t{len(intensities)} assigned to cells")
        print(f"\ttime taken:{(time.time() - t1)/60} minutes.")

        return (tr, intensities)

    else:
        print(f"skipping fov_{f:04}, no transcripts found.")
        return None


t0 = time.time()

sel_fovs = []
for f in range(fov_count):
    if not Path(tr_compl.format(f)).is_file() and (
        Path(tr_st.format(f)).is_file() and Path(im_st.format(f)).is_file()
    ):
        sel_fovs.append(f)

print(f"FOVs without files currently: {sel_fovs}")

# going to manually override this to just have a partial run

# sel_fovs = [0,1,2,4,5,6,19,20,21,22]
# sel_fovs = [23, 24, 25, 26, 27, 38, 39]
# sel_fovs = [40, 41, 42, 44, 45, 46, 47, 48, 49, 57, 58, 59, 60, 61, 62, 63]
# sel_fovs = [64, 65, 66, 67, 68, 69, 70, 71, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120]
sel_fovs = [
    121,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
]
# sel_fovs = [151, 152, 153, 154, 155, 156, 157, 158]
# sel_fovs = [159, 160, 161, 162, 163, 164, 165, 166]

with Pool(pool_size) as pool:
    results = pool.starmap(
        filtImage,
        zip(
            repeat(loc),
            sel_fovs,
            repeat(im_st),
            repeat(tr_st),
            repeat(min_size),
            repeat(dilat),
            repeat(sigma),
            repeat(thresh),
            repeat(tr_compl),
        ),
    )

print(f"Total time to complete: {(time.time() - t0)/60} minutes.")

n = 1
rad = 1
chk_fovs = [0]

# testing new params..
sigma = 35  # spread on gaussian filter
dilat = 20  # size of dilation

for f in random.sample(chk_fovs, n):
    print(f"img {f}")
    im = skimage.io.imread(im_st.format(f))
    tr = pd.read_csv(tr_compl.format(f))

    # go through a given cell and find all transcripts assigned to it
    # as well as their relative probabilites
    for m in set(im.flatten()):
        if m == 0:
            continue

        tr_df = pd.DataFrame(columns=["x", "y", "z", "p"])
        for r, row in tr.iterrows():
            assigned = ast.literal_eval(row["cell_ids"])
            if str(m) in assigned.keys():
                tr_df.loc[-1] = [row["x"], row["y"], row["global_z"], assigned[str(m)]]
                tr_df.index = tr_df.index + 1
                tr_df = tr_df.sort_index()

        if len(tr_df) == 0:
            print(f"No transcripts in mask ID {m}")
            continue

        # print(f"cell {m}\n{tr_df}")

        # plot the transcripts using the intensity as a color scale
        # on both the raw segmask and the blurred one
        # (will have to replicate blurring code here)

        mask = im == m

        def fn(y): return skimage.morphology.binary_dilation(
            y, skimage.morphology.disk(dilat)
        )
        filt = np.array([fn(sl) for sl in mask])
        filt = np.array([skimage.filters.gaussian(sl, sigma=sigma) for sl in filt])

        x_0 = tr_df["x"].min()
        x_1 = tr_df["x"].max()
        y_0 = tr_df["y"].min()
        y_1 = tr_df["y"].max()

        for z in range(np.shape(im)[0]):
            fig = plt.figure()
            ax = fig.add_subplot(111)

            # ax.imshow((mask + filt)[z,:,:], cmap="winter")
            ax.imshow((filt)[z, :, :], cmap="winter")

            ax.set_xlim([x_0 - 10, x_1 + 10])
            ax.set_ylim([y_0 - 10, y_1 + 10])

            tally = 0
            for r, row in tr_df.iterrows():
                if row["z"] != z:
                    continue
                circle = Circle(
                    (int(row["x"]), int(row["y"])),
                    radius=rad,
                    facecolor=matplotlib.cm.autumn(row["p"]),
                )
                ax.add_patch(circle)
                tally += 1
                # print(f"\tadded {r} @ ({int(row['x'])},{int(row['y'])})")
            print(f"plotting {m}:{z}\n{tally} total transcripts")
            fig.colorbar(
                matplotlib.cm.ScalarMappable(
                    norm=matplotlib.colors.Normalize(vmin=0, vmax=1),
                    cmap=matplotlib.cm.autumn,
                ),
                ax=ax,
            )
            plt.show()


# manually resetting fovs for the ones we ran
sel_fovs = []
for f in range(fov_count):
    if Path(tr_compl.format(f)).is_file():
        sel_fovs.append(f)
sel_fovs

del results

full_tr = None

print(f"current memory usage: {psutil.virtual_memory()[3]/1000000000}GB")
print("Unpacking results...")

for f in range(fov_count):
    if Path(tr_compl.format(f)).is_file():
        tr = pd.read_csv(f"{loc}fov_{f:04}_cellids.csv")
        if full_tr is None:
            full_tr = tr
        else:
            full_tr = pd.concat([full_tr, tr])

full_tr.to_csv(f"{loc_compl}all_soft_cellids.csv", index=False)

tr_tally = len(full_tr)
max_ass_est = 5
cutoffs = np.empty((tr_tally, max_ass_est))
c_row = 0

for f in sel_fovs:
    tr = pd.read_csv(tr_compl.format(f))
    for r, row in tr.iterrows():
        assigned = ast.literal_eval(row["cell_ids"])
        ind = 0

        # make the mat wider if needed
        if np.shape(cutoffs)[1] < len(assigned):
            n_wid = len(assigned) - np.shape(cutoffs)[1]
            cutoffs = np.append(cutoffs, np.empty((tr_tally, n_wid)), axis=1)

        # transfer in our floats...
        for k, v in assigned.items():
            cutoffs[c_row, ind] = v
            ind += 1
        c_row += 1
    print(f"Just completed fov_{f:04}, current c_row {c_row}")


assigned_tr = []
dupe_tr = []
num = 100
rang = np.linspace(0, 1, num=num)

for thresh in rang:
    assd = cutoffs > thresh
    assigned_tr.append(np.nansum(assd))
    dupe_tr.append(np.nansum(assd[np.nansum(assd, axis=1) > 1]))

fig, ax = plt.subplots()
im = ax.scatter(assigned_tr, dupe_tr, c=rang)
# ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlabel("Total transcripts assigned")
ax.set_ylabel("Number of transcripts assigned to more than one cell")
fig.colorbar(im, ax=ax)
plt.show()

dif = np.diff(np.diff(dupe_tr))

fig, ax = plt.subplots()
im = ax.scatter(assigned_tr[2:], dif, c=rang[2:])
# ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlabel("Total transcripts assigned")
ax.set_ylabel("Number of transcripts assigned to more than one cell, 2nd derivative")
fig.colorbar(im, ax=ax)
plt.show()

percentile = np.argmax(dif)
print(
    f"Elbow point at thresh of {rang[percentile+2]}, \n\t{assigned_tr[percentile+2]} transcripts assigned\n\t{dupe_tr[percentile+2]} assigned to more than one cell\n\t{dupe_tr[percentile+2]/assigned_tr[percentile+2]*100:.5f}% assigned to more than one cell."
)

ass_per = [dupe_tr[i] / assigned_tr[i] * 100 for i in range(num)]
fir = ass_per.index(0)
print(
    f"{fir}\nFirst overlap happens at {rang[fir]}, \n\t{assigned_tr[fir]} transcripts assigned\n\t{dupe_tr[fir]} assigned to more than one cell\n\t{dupe_tr[fir]/assigned_tr[fir]*100:.5f}% assigned to more than one cell."
)


thresh = 0.90
# transcripts with assigned values above this will be retained

cxg_dict = {}
for f in range(fov_count):
    if Path(tr_compl.format(f)).is_file():
        tr = pd.read_csv(tr_compl.format(f))
        for i, row in tr.iterrows():
            assigned = ast.literal_eval(row["cell_ids"])
            for k, v in assigned.items():
                if float(v) > thresh:
                    if k in cxg_dict.keys():
                        if row["gene"] in cxg_dict[k].keys():
                            cxg_dict[k][row["gene"]] += 1
                        else:
                            cxg_dict[k][row["gene"]] = 1
                    else:
                        cxg_dict[k] = {row["gene"]: 1}
        # print(cxg_dict)
        del tr
        print(
            f"completed fov_{f:04}\ncurrent memory usage: {psutil.virtual_memory()[3]/1000000000}GB"
        )

cxg_df = pd.DataFrame.from_dict(cxg_dict, orient="index")
cxg_df

adata = ad.AnnData(cxg_df)
adata.obs["fov"] = pd.DataFrame(np.zeros((len(adata), 1)))
adata.obs["size"] = pd.DataFrame(np.zeros((len(adata), 1)))
adata.obs["x_coords"] = pd.DataFrame(np.zeros((len(adata), 1)))
adata.obs["y_coords"] = pd.DataFrame(np.zeros((len(adata), 1)))
adata.obs["z_coords"] = pd.DataFrame(np.zeros((len(adata), 1)))

# Create dictionary with coordinates of each FOV
y_size = 72869
x_size = 82427

# Primary FOVs
y_step = 4000
x_step = 4000
pos = 0
primary_coords = {}
for x_start in range(0, x_size, x_step):
    for y_start in range(0, y_size, y_step):
        fov = pos
        primary_coords[fov] = {
            "y": (y_start, y_start + y_step),
            "x": (x_start, x_start + x_step),
        }
        pos += 1

pool_size = 8


def getContour(tif, i):
    binimg = deepcopy(tif)
    binimg[binimg != i] = 0
    binimg[binimg > 0] = 1
    binimg = binimg.astype("uint8")
    contours = []
    for n in range(np.shape(binimg)[0]):
        contours.append(
            cv2.findContours(binimg[n, :, :], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[
                -2
            ]
        )
    del binimg
    # print(f"{i}: {psutil.virtual_memory()[3]/1000000000}GB")
    return contours, i


# avail_fovs = all_fovs
avail_fovs = [f for f in range(fov_count)]

for f in sel_fovs:
    if Path(im_st.format(f)).is_file():
        im = skimage.io.imread(im_st.format(f))
        im = clear_border(im)
        with Pool(pool_size) as pool:
            results = pool.starmap(getContour, zip(repeat(im), list(set(im.flatten()))))
        all_contours = {i: contour for contour, i in results}
        for k, v in all_contours.items():
            if k == 0 or str(k) not in adata.obs_names:  # don't run on bg
                continue
            moments = []
            for n in range(len(v)):
                if len(v[n]) > 0:
                    moments.append(cv2.moments(v[n][0]))
                else:
                    moments.append({"m00": 0, "m01": 0, "m10": 0})

            total_size = sum([n["m00"] for n in moments])

            adata.obs.loc[adata.obs.index.isin([str(k)]), "fov"] = f
            adata.obs.loc[adata.obs.index.isin([str(k)]), "size"] = total_size
            if total_size != 0:
                adata.obs.loc[adata.obs.index.isin([str(k)]), "x_coords"] = (
                    int(sum([n["m10"] for n in moments]) / total_size)
                    + primary_coords[f]["x"][0]
                )
                adata.obs.loc[adata.obs.index.isin([str(k)]), "y_coords"] = (
                    int(sum([n["m01"] for n in moments]) / total_size)
                    + primary_coords[f]["y"][0]
                )
                adata.obs.loc[adata.obs.index.isin([str(k)]), "z_coords"] = (
                    sum([moments[ln]["m00"] * ln for ln in range(len(moments))])
                    / total_size
                )
            # print(adata[str(k)].obs)
            del moments
        del all_contours
        del im
        # all_fovs.remove(f)
        print(f"Completed fov_{f:04}")
    else:
        print(f"Skipping fov_{f:04}")
        # all_fovs.remove(f)

adata.X = np.nan_to_num(adata.X)

adata.write(f"{loc_compl}cxg_adata.h5ad")

adata = ad.read(f"{loc_compl}cxg_adata.h5ad")

# Filter cells in z-slices
filter_values_lo = {1: 300, 2: 800, 3: 1800, 4: 2500, 5: 3500, 6: 5000}
filter_values_hi = {1: 15000, 2: 30000, 3: 50000, 4: 70000, 5: 100000, 6: 180000}

filtered_adatas = []

# filter only to cells that we found in the prior step
sp_adata = adata[~np.isnan(adata.obs["z_coords"])]
tmp_adata = adata[~np.isnan(adata.obs["z_coords"])]

tmp_adata = tmp_adata[
    tmp_adata.obs["size"]
    > tmp_adata.obs["z_coords"].apply(math.floor).map(filter_values_lo)
]
filtered_adatas.append(tmp_adata)
del tmp_adata

print(f"#cells before volume filter: {sp_adata.n_obs}")
sp_adata = sc.concat(filtered_adatas)
print(f"#cells after volume filter: {sp_adata.n_obs}")
del filtered_adatas

# Calculate QC metrics
sc.pp.calculate_qc_metrics(sp_adata, inplace=True)

# Filter data
print(f"#cells before count filter: {sp_adata.n_obs}")

# total_counts
sc.pp.filter_cells(sp_adata, min_counts=10)  # orig 10
# sc.pp.filter_cells(sp_adata, max_counts=1200)

# n_genes_by_counts
sc.pp.filter_cells(sp_adata, min_genes=10)  # orig 10
print(f"#cells after count filter: {sp_adata.n_obs}")
print(f"#genes before minimum cells per gene filter: {sp_adata.n_vars}")

# n_cells_by_counts
sc.pp.filter_genes(sp_adata, min_cells=5)  # HAD TO SET THIS A LOT LOWER
# DUE TO LOW COUNTS...
print(f"#genes after minimum cells per gene filter: {sp_adata.n_vars}")

# Read scRNAseq data
annotation_level = "ann_level_1"
sc_adata = ad.read(f"{sc_loc}{version}_500genes_noraw.h5ad")

# Calculate QC metrics
sc_adata.var["mt"] = sc_adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(sc_adata, qc_vars=["mt"], inplace=True, percent_top=None)

# Filter data

print(f"#cells before count filter: {sc_adata.n_obs}")

# total_counts
sc.pp.filter_cells(sc_adata, min_counts=30)
sc.pp.filter_cells(sc_adata, max_counts=120)

# n_genes_by_counts
sc.pp.filter_cells(sc_adata, min_genes=20)

print(f"#cells after count filter: {sc_adata.n_obs}")

print(f"#genes before minimum cells per gene filter: {sc_adata.n_vars}")

# n_cells_by_counts
sc.pp.filter_genes(sc_adata, min_cells=100)

print(f"#genes after minimum cells per gene filter: {sc_adata.n_vars}")

# Normalize by total counts
sc.pp.normalize_total(sc_adata, inplace=True)

# Log
sc.pp.log1p(sc_adata)

# Z scale
# sc.pp.scale(sc_adata)

# Remove unknown celltypes
sc_adata = sc_adata[sc_adata.obs[annotation_level] != "Unknown"]

# Find marker genes for celltypes (using only genes shared with spatial data)
sc_adata = sc_adata[:, np.isin(sc_adata.var.index, sp_adata.var.index)]
sc.tl.rank_genes_groups(sc_adata, annotation_level, method="wilcoxon")
# sc.tl.filter_rank_genes_groups(sc_adata, annotation_level)
sc.tl.filter_rank_genes_groups(sc_adata)
marker_df = sc.get.rank_genes_groups_df(sc_adata, group=None)
celltypes = sc_adata.uns["rank_genes_groups"]["names"].dtype.names
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

Counter(sc_adata.obs[annotation_level])

# Unsupervised clustering (nothing new here, just verifying that we're getting decent looking clusters)
n_pcs = 50
n_neighbors = 25
res = 0.5

# PCA
start = time.time()
sc.pp.pca(sp_adata, svd_solver="arpack", n_comps=n_pcs, copy=False)
print(time.time() - start)

# Compute neightborhood graph
start = time.time()
sc.pp.neighbors(sp_adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
print(time.time() - start)

# Calculate umap representation
start = time.time()
sc.tl.umap(sp_adata)
print(time.time() - start)

# Leiden clustering
start = time.time()
sc.tl.leiden(sp_adata, resolution=res)
print(time.time() - start)

# UMAP
print(f"n_pcs = {n_pcs}, n_neighbors = {n_neighbors}, resolution = {res}")
sc.pl.umap(sp_adata, color=["leiden"], cmap="tab20")

# Dendrogram
sc.tl.dendrogram(sp_adata, "leiden")
sc.pl.dendrogram(sp_adata, "leiden")

tmp_adata = sp_adata
tmp_adata = tmp_adata[
    tmp_adata.obs["leiden"].isin(
        [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5,",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
        ]
    )
]

fig = plt.figure(figsize=(15, 9), constrained_layout=True, dpi=200)

for cluster in sorted(set(tmp_adata.obs["leiden"].astype(int))):
    cluster_adata = tmp_adata[tmp_adata.obs["leiden"] == str(cluster)]
    plt.scatter(
        cluster_adata.obs["x_coords"],
        cluster_adata.obs["y_coords"],
        label=int(cluster),
        cmap=matplotlib.colors.ListedColormap(plt.get_cmap("tab20").colors[::2]),
        s=1,
        alpha=1,
    )

lgnd = plt.legend(fontsize=14)
for handle in lgnd.legendHandles:
    handle.set_sizes([100])
plt.ylim(-1000, 81000)
plt.xlim(-2000, 70000)
plt.show()

for celltype in sorted(pos_markers):
    print(celltype)
    print("Positive Expression")
    sc.pl.dotplot(
        sp_adata,
        pos_markers[celltype][:20],
        groupby="leiden",
        dendrogram=True,
        title=celltype,
    )
    print("Negative Expression")
    sc.pl.dotplot(
        sp_adata,
        neg_markers[celltype][:20],
        groupby="leiden",
        dendrogram=True,
        title=celltype,
    )

# Here we manually assign cluster labels to the cell type determined by manual inspection.
# Any cluster that doesn't have a matching cell type you can just label as "Unassigned". These could be cell
# types that just aren't in the scRNAseq data so we aren't testing for them, or they could simply be low quality
# cells that somehow got past our filters and clustered together.
# Endothelial
# Epithelial
# Lymphoid
# Myeloid
# Stromal

num_leiden = len(set(sp_adata.obs["leiden"]))
clusters2types = {k: "Unassigned" for k in range(0, num_leiden)}

assigned_clusters = {
    "Endothelial": ["8"],
    "Epithelial": ["2", "4", "6", "1", "7"],
    "Immune": ["0", "3"],
    "Stromal": ["5"],
}
# really unsure:

for k1, v1 in clusters2types.items():
    for k2, v2 in assigned_clusters.items():
        for i in v2:
            clusters2types[int(i)] = k2

sp_adata.obs["cell_type_manual_major"] = [
    clusters2types[int(cluster)] for cluster in sp_adata.obs["leiden"]
]
types = list(set(sp_adata.obs["cell_type_manual_major"]))
type2int = {cell_type: x for x, cell_type in enumerate(types)}
sp_adata.obs["int_type"] = [
    type2int[cell_type] for cell_type in sp_adata.obs["cell_type_manual_major"]
]

clusters2types

# for z_count in range(6):
#    print(z_count - 1)

tmp_adata = sp_adata


fig = plt.figure(figsize=(15, 9), constrained_layout=True, dpi=200)

for celltype in sorted(set(tmp_adata.obs["cell_type_manual_major"])):
    celltype_adata = tmp_adata[tmp_adata.obs["cell_type_manual_major"] == celltype]
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
plt.ylim(-1000, 81000)
plt.xlim(-2000, 70000)
plt.show()

# UMAP
sc.pl.umap(sp_adata, color=["cell_type_manual_major"], cmap="tab20")

annotation_level = "ann_level_2"
sc_adata = ad.read(f"{sc_loc}{version}_500genes_noraw.h5ad")

# Calculate QC metrics
sc_adata.var["mt"] = sc_adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(sc_adata, qc_vars=["mt"], inplace=True, percent_top=None)

# Filter data

print(f"#cells before count filter: {sc_adata.n_obs}")

# total_counts
sc.pp.filter_cells(sc_adata, min_counts=30)
sc.pp.filter_cells(sc_adata, max_counts=120)

# n_genes_by_counts
sc.pp.filter_cells(sc_adata, min_genes=20)

print(f"#cells after count filter: {sc_adata.n_obs}")

print(f"#genes before minimum cells per gene filter: {sc_adata.n_vars}")

# n_cells_by_counts
sc.pp.filter_genes(sc_adata, min_cells=100)

print(f"#genes after minimum cells per gene filter: {sc_adata.n_vars}")

# Normalize by total counts
sc.pp.normalize_total(sc_adata, inplace=True)

# Log
sc.pp.log1p(sc_adata)

# Z scale
sc.pp.scale(sc_adata)

# Remove unknown celltypes
sc_adata = sc_adata[sc_adata.obs[annotation_level] != "Unknown"]

# Subset for celltype
sc_adata = sc_adata[np.isin(sc_adata.obs["ann_level_1"], ["Immune"])]
sc_adata = sc_adata[~np.isin(sc_adata.obs["ann_level_2"], ["Hematopoietic stem cells"])]

# Find marker genes for celltypes (using only genes shared with spatial data)
sc_adata = sc_adata[:, np.isin(sc_adata.var.index, sp_adata.var.index)]
sc.tl.rank_genes_groups(sc_adata, annotation_level, method="wilcoxon")
marker_df = sc.get.rank_genes_groups_df(sc_adata, group=None)
celltypes = sc_adata.uns["rank_genes_groups"]["names"].dtype.names
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

Counter(sc_adata.obs[annotation_level])

immune_adata = sp_adata[sp_adata.obs["cell_type_manual_major"] == "Immune"]

# Unsupervised clustering (nothing new here, just verifying that we're getting decent looking clusters)
n_pcs = 50
n_neighbors = 25
res = 0.1

# PCA
start = time.time()
sc.tl.pca(immune_adata, svd_solver="arpack", n_comps=n_pcs)
print(time.time() - start)

# Compute neightborhood graph
start = time.time()
sc.pp.neighbors(immune_adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
print(time.time() - start)

# Calculate umap representation
start = time.time()
sc.tl.umap(immune_adata)
print(time.time() - start)

# Leiden clustering
start = time.time()
sc.tl.leiden(immune_adata, resolution=res)
print(time.time() - start)

# UMAP
print(f"n_pcs = {n_pcs}, n_neighbors = {n_neighbors}, resolution = {res}")
sc.pl.umap(immune_adata, color=["leiden"], cmap="tab20")

tmp_adata = immune_adata

fig = plt.figure(figsize=(15, 9), constrained_layout=True, dpi=200)

for cluster in sorted(set(tmp_adata.obs["leiden"].astype(int))):
    cluster_adata = tmp_adata[tmp_adata.obs["leiden"] == str(cluster)]
    plt.scatter(
        cluster_adata.obs["x_coords"],
        cluster_adata.obs["y_coords"],
        label=int(cluster),
        cmap=matplotlib.colors.ListedColormap(plt.get_cmap("tab20").colors[::2]),
        s=1,
        alpha=1,
    )

lgnd = plt.legend(fontsize=14)
for handle in lgnd.legendHandles:
    handle.set_sizes([100])
plt.ylim(-1000, 81000)
plt.xlim(-2000, 70000)
plt.show()

# Dendrogram
sc.tl.dendrogram(immune_adata, "leiden")
sc.pl.dendrogram(immune_adata, "leiden")

for celltype in sorted(pos_markers):
    print(celltype)
    print("Positive Expression")
    sc.pl.dotplot(
        immune_adata,
        pos_markers[celltype][:20],
        groupby="leiden",
        dendrogram=True,
        title=celltype,
    )
    print("Negative Expression")
    sc.pl.dotplot(
        immune_adata,
        neg_markers[celltype][:20],
        groupby="leiden",
        dendrogram=True,
        title=celltype,
    )

# Here we manually assign cluster labels to the cell type determined by manual inspection.
# Any cluster that doesn't have a matching cell type you can just label as "Unassigned". These could be cell
# types that just aren't in the scRNAseq data so we aren't testing for them, or they could simply be low quality
# cells that somehow got past our filters and clustered together.
# Endothelial
# Epithelial
# Lymphoid
# Myeloid
# Stromal

num_leiden = len(set(immune_adata.obs["leiden"]))
clusters2types = {k: "Unassigned" for k in range(0, num_leiden)}

assigned_clusters = {"Lymphoid": ["0"], "Myeloid": ["1"]}

for k1, v1 in clusters2types.items():
    for k2, v2 in assigned_clusters.items():
        for i in v2:
            clusters2types[int(i)] = k2

immune_adata.obs["cell_type_manual_minor1"] = [
    clusters2types[int(cluster)] for cluster in immune_adata.obs["leiden"]
]
types = list(set(immune_adata.obs["cell_type_manual_minor1"]))
type2int = {cell_type: x for x, cell_type in enumerate(types)}
immune_adata.obs["int_type"] = [
    type2int[cell_type] for cell_type in immune_adata.obs["cell_type_manual_minor1"]
]

clusters2types

tmp_adata = immune_adata

fig = plt.figure(figsize=(15, 9), constrained_layout=True, dpi=200)

for celltype in sorted(set(tmp_adata.obs["cell_type_manual_minor1"])):
    celltype_adata = tmp_adata[tmp_adata.obs["cell_type_manual_minor1"] == celltype]
    plt.scatter(
        celltype_adata.obs["x_coords"],
        celltype_adata.obs["y_coords"],
        label=celltype,
        cmap="tab20",
        s=1,
        alpha=1,
    )

lgnd = plt.legend(fontsize=14)
for handle in lgnd.legendHandles:
    handle.set_sizes([100])
plt.ylim(-1000, 81000)
plt.xlim(-2000, 70000)
plt.show()

# UMAP
sc.pl.umap(immune_adata, color=["cell_type_manual_minor1"], cmap="tab20")

temp_adata = sp_adata[sp_adata.obs["cell_type_manual_major"] != "Immune"]
sp_adata = sc.concat([temp_adata, immune_adata], join="outer")

dat = datetime.today().strftime("%Y%m%d")
sp_adata.write_h5ad(f"{loc}/cxg_adata_highlevel_wpca_{dat}.h5ad")

adata = sp_adata
non_blanks = [name for name in adata.var_names if "lank" not in name]
adata = adata[:, non_blanks]

filtered = {}
for cat in adata.obs["cell_type_manual_major"].values.categories:
    filtered[cat] = deepcopy(adata[adata.obs["cell_type_manual_major"] == cat])
for cat in adata.obs["cell_type_manual_minor1"].values.categories:
    filtered[cat] = deepcopy(adata[adata.obs["cell_type_manual_minor1"] == cat])
del filtered["Immune"]

avgs = {}
for k, v in filtered.items():
    avgs[k] = np.average(v.X, axis=0)

df_avg = pd.DataFrame.from_dict(avgs, orient="index", columns=adata.var.index)
df_avg

sns.set()
g = sns.clustermap(df_avg, norm=matplotlib.colors.LogNorm())

c_maj = Counter(adata.obs["cell_type_manual_major"].values)
c_min = Counter(adata.obs["cell_type_manual_minor1"].values)
norms = {k: v for (k, v) in c_maj.items() if "Immune" not in k}
for k, v in c_min.items():
    if isinstance(k, str):
        norms[k] = v
norms

norms_sort = collections.OrderedDict(sorted(norms.items()))
norm_total = sum([v for (k, v) in norms.items()])
df_normed = df_avg.multiply(
    [(norm_total - x) / norm_total * 100 for x in norms_sort.values()], axis=0
)
df_normed

g = sns.clustermap(df_normed, norm=matplotlib.colors.LogNorm())

df_comp = df_normed
df_comp

# manually resetting fovs for the ones we ran
sel_fovs = []
for f in range(fov_count):
    if Path(tr_compl.format(f)).is_file():
        sel_fovs.append(f)

adata.obs.index

dupe_ass = {}
all_reass = {}
results = {}
thresh_over = 0.1
thresh_conf = thresh

sel_fovs = [0, 1, 2]  # manual spec for testing

for f in sel_fovs:
    dupe_ass[f] = {}
    tr = pd.read_csv(f"{loc}fov_{f:04}_cellids.csv")
    tr = tr.reset_index()
    im = skimage.io.imread(f"{loc}fov_{f:04}.tiff")

    print(f"Loaded fov_{f:04}.")
    for r, row in tr.iterrows():
        # omit blanks
        if "lank" in row["gene"]:
            continue

        asst = ast.literal_eval(row["cell_ids"])
        if len(asst.keys()) > 1:
            # first, filter out any cell ids that got filtered before labelling
            elg_cells = [k for k in asst.keys() if k in adata.obs.index]
            print(f"{asst} {elg_cells}")

            # now, select for transcripts above threshold
            # these are the only things eligible to be assigned to
            pos_keys = tuple(sorted([k for k in elg_cells if asst[k] > thresh_over]))

            # if it's not eligible to be assigned to anything, we don't care...
            if len(pos_keys) == 0:
                continue

            # next, see if this is a comparison that we even care about
            # need at least one of these cells to be a different type
            majors = [
                adata.obs.loc[[k]]["cell_type_manual_major"].values[0]
                for k in elg_cells
            ]
            minors = [
                adata.obs.loc[[k]]["cell_type_manual_minor1"].values[0]
                for k in elg_cells
            ]
            types = [
                majors[i] if majors[i] != "Immune" else minors[i]
                for i in range(len(majors))
            ]

            # we don't care if everything is the same type
            if len(set(types)) < 2:
                continue

            transc = (row["gene"], row["index"])
            tp = tuple(sorted(elg_cells))
            # OK NOW. we are looking to reassign transc to one of pos_keys
            # while we are looking at a comparison of anything in tp

            if tp not in dupe_ass[f].keys():
                dupe_ass[f][tp] = {}

            if pos_keys not in dupe_ass[f][tp].keys():
                dupe_ass[f][tp][pos_keys] = [transc]
            else:
                dupe_ass[f][tp][pos_keys].append(transc)

    # pruning step: reduce entries where not all comparisons are made
    # this is a lot more complicated with n possible comparisons...

    # print(f"pre prune dupes {dupe_ass[f]}")

    to_del = []
    to_add = {}
    for k, v in dupe_ass[f].items():
        # remove comparisons where there are no overlapping possible cell assignments
        if list(set([len(i) for i in v.keys()])) == [1]:
            to_del.append(k)
        # if this can be better crammed inside a more specific comparison, do so.
        if (
            max([len(i) for i in v.keys()]) == len(k) - 1
            and len([len(i) for i in v.keys() if i == len(k) - 1]) == 1
        ):
            k_sel = [i for i in v.keys() if len(i) == len(k) - 1][0]
            to_add[k_sel] = k
            to_del.append(k)
    for k, k_big in to_add.items():
        for sm in dupe_ass[f][k].keys():
            dupe_ass[f][k][sm].extend(dupe_ass[f][k_big][sm])
    for k in to_del:
        dupe_ass[f].pop(k, None)

    # find and save all confident assignments, to avoid
    # repeated iteration for each pairing
    conf_tr = {}
    for r, row in tr.iterrows():
        asst = ast.literal_eval(row["cell_ids"])
        if len(asst.keys()) > 1:
            for key, v in asst.items():
                if v > thresh_conf and "lank" not in row["gene"]:
                    if key in conf_tr.keys():
                        conf_tr[key].append(row["gene"])
                    else:
                        conf_tr[key] = [row["gene"]]

    # this only matters within an FOV
    all_reass_check = set()

    # if we don't have any comparisons, skip ahead...
    if len(dupe_ass[f].keys()) == 0:
        continue

    # start with 2 way comparisons, working our way up to n-way
    max_comp = max([len(k) for k in dupe_ass[f].keys()])

    for ln in range(2, max_comp + 1):
        # print(l)
        for k, gene_lis in dupe_ass[f].items():
            if len(k) != ln:
                continue

            reass_inds = {}

            # get our celltypes
            types = [
                adata.obs.loc[[k[i]]]["cell_type_manual_major"].values[0]
                for i in range(ln)
            ]
            types = [
                (
                    adata.obs.loc[[k[i]]]["cell_type_manual_minor1"].values[0]
                    if types[i] == "Immune"
                    else types[i]
                )
                for i in range(ln)
            ]

            # need to go through and retrieve the confident transcripts
            conf_inds = []
            for v in range(ln):
                temp = []
                if k[v] in conf_tr.keys():
                    for gene in conf_tr[k[v]]:
                        t = df_comp[gene].copy()
                        t = t.loc[types]
                        # t.name = f"{gene}_{types[l]}"
                        temp.append(t)
                    conf_inds.append(pd.concat(temp, axis=1))
                else:
                    conf_inds.append([])

            amb_inds = {}
            for inds, genes in gene_lis.items():

                # treat genes only in one of these zones separately
                if len(inds) == 1:
                    t = max([i for i in range(ln) if inds[0] == k[i]])
                    for gene, index in genes:
                        temp = df_comp[gene].copy()
                        temp = temp.loc[types]
                        if len(conf_inds[t]) > 0:
                            conf_inds[t] = pd.concat([conf_inds[t], temp], axis=1)
                        else:
                            conf_inds[t] = temp
                else:
                    # this is the overlap zone
                    amb_inds[inds] = []
                    reass_inds[inds] = []
                    # print(f"{inds} all genes: {genes}")
                    for gene, index in genes:
                        # if this has been reassigned elsewhere, go with that
                        if index in all_reass.keys():
                            c = all_reass[index]
                            cell_ind = [ks for ks in range(len(k)) if k[ks] == c]
                            if len(cell_ind) != 1:
                                print(
                                    f"Warning! Gene {index} has been assigned to {all_reass[index]}, which is not one of the eligible cells in this comparison!"
                                )
                                continue
                            temp = df_comp[gene].copy()
                            temp = temp.loc[types]
                            if len(conf_inds[cell_ind[0]]) > "0":
                                conf_inds[cell_ind[0]] = pd.concat(
                                    [conf_inds[cell_ind[0]], temp], axis=1
                                )
                            else:
                                conf_inds[cell_ind[0]] = temp
                        else:
                            temp = df_comp[gene].copy()
                            temp = temp.loc[types]
                            amb_inds[inds].append(temp)
                            reass_inds[inds].append(index)

            ov_check = []
            for ind, v in reass_inds.items():
                ov_check.extend(v)
            overlap = set.intersection(all_reass_check, set(ov_check))
            if len(overlap) > 0:
                print(f"Warning! Transcripts {overlap} have already been reassigned!")

            top_val = 0
            winning_combo = []

            print(amb_inds.keys())
            for combo in itertools.product(
                *[(*cellid, "Original") for cellid in amb_inds.keys()]
            ):
                sel_vals = {}
                for v in range(len(conf_inds)):
                    if len(conf_inds[v]) > 0:  # not interested if there aren't any
                        if hasattr(
                            conf_inds[v].values[v], "__len__"
                        ):  # it's possible to just have one cell
                            sel_vals[k[v]] = list(conf_inds[v].values[v])
                        else:
                            sel_vals[k[v]] = [conf_inds[v].values[v]]

                key_ref = list(amb_inds.keys())
                for v_tup in range(len(combo)):
                    cell_ind = combo[v_tup]
                    sel_keyset = key_ref[v_tup]
                    if cell_ind != "Original":
                        ind = list(sel_keyset).index(cell_ind)

                        t_df = pd.concat(amb_inds[sel_keyset], axis=1)

                        if cell_ind in sel_vals.keys():
                            n_val = t_df.values[ind]
                            if hasattr(n_val, "__len__"):
                                sel_vals[cell_ind].extend(n_val)
                            else:
                                sel_vals[cell_ind].append(n_val)
                        else:
                            sel_vals[cell_ind] = [t_df.values[ind]]
                    else:
                        for tr_tup in gene_lis[sel_keyset]:
                            tr_id = tr_tup[1]
                            row = tr.loc[tr["index"] == int(tr_id)]
                            one_cell_ind = im[row["y"] - 1, row["x"] - 1][0]
                            if str(one_cell_ind) in list(sel_keyset):
                                ind = list(sel_keyset).index(str(one_cell_ind))
                            else:
                                if one_cell_ind != 0:
                                    print(
                                        f"Warning! Original cell assignment {one_cell_ind} for transcript {tr_id} not in possible comparison list {sel_keyset}!"
                                    )
                                continue

                            t_df = df_comp[tr_tup[0]].copy()
                            t_df = t_df.loc[types]

                            if cell_ind in sel_vals.keys():
                                n_val = t_df.values[ind]
                                if hasattr(n_val, "__len__"):
                                    sel_vals[cell_ind].extend(n_val)
                                else:
                                    sel_vals[cell_ind].append(n_val)
                            else:
                                sel_vals[cell_ind] = [t_df.values[ind]]

                everything = [
                    v for row in [v for k, v in sel_vals.items()] for v in row
                ]
                avg = np.average(everything)

                # idk what causes the bug that requires this but...
                while hasattr(avg, "__len__"):
                    avg = np.average(avg)

                if avg > top_val:
                    winning_combo = combo
                    top_val = avg

            key_ref = list(reass_inds.keys())
            for v in range(len(winning_combo)):
                for ge in reass_inds[key_ref[v]]:
                    all_reass[ge] = winning_combo[v]
                if winning_combo[v] != "Original":
                    ctype = adata.obs.loc[[winning_combo[v]]][
                        "cell_type_manual_major"
                    ].values[0]
                    if ctype == "Immune":
                        ctype = adata.obs.loc[[winning_combo[v]]][
                            "cell_type_manual_minor1"
                        ].values[0]
                    results[key_ref[v]] = ctype
                else:
                    results[key_ref[v]] = "Original"

            # if a_vals.mean() > b_vals.mean():
            #    results[k] = gene_inds.index[0]
            #    all_reass.update({i:k[0] for i in reass_inds}) # gene ind : cell ind

            for k, v in reass_inds.items():
                all_reass_check.update(set(v))

Counter(results.values())
