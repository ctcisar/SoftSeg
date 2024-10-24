import ast
import glob
import logging
import time
from copy import deepcopy
from datetime import datetime
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import anndata as ad
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parse
import skimage
import skimage.io
from skimage.measure import regionprops
from skimage.segmentation import clear_border


class SoftAssigner:
    def __init__(self, csv_loc, im_loc, complete_loc, pool_size=1):
        """Initialize internal parameters.

        csv_loc: location of transcript file, per fov.
        im_loc: location of image mask tiff, per fov.
        complete_loc: location of output files. NOT per fov: a string
           for each fov's modified transcript table will be appended.

        All of these variables will be formatted with `.format(fov_number)`.
        """
        self.csv_loc = csv_loc
        self.im_loc = im_loc
        self.complete_loc = complete_loc
        self.complete_csv_name = f"{complete_loc}fov_{{:04}}_cellids.csv"
        self.pool_size = pool_size
        self.logger = logging.getLogger()
        logging.baseConfig(filename=f"{complete_loc}{datetime.datetime.now()}run.log", encoding='utf-8', level=logging.DEBUG)

    def size_filter(im, min_size=None, max_size=None):
        """Removes all items from mask that do not meet size requirements.

        im: Labelled input image
        min_size: minimum size for a mask, in pixels.
        max_size: maximum size for a mask, in pixels.

        returns: Labelled image without out-of-range-sized masks.
        """
        props = regionprops(im)
        to_del = []
        for region in props:
            if min_size is not None and region.area < min_size:
                to_del.append(region.label)
            if max_size is not None and region.area > max_size:
                to_del.append(region.label)

        def will_del(y):
            return 0 if y in to_del else y

        return np.vectorize(will_del)(im)

    def get_all_fovs(self):
        """Returns a list of all valid fov indicies for this object."""
        ims = glob.glob(self.im_loc.format("****"))
        ims = [parse.parse(self.im_loc, im)[0] for im in ims]

        trs = glob.glob(self.csv_loc.format("****"))
        trs = [parse.parse(self.csv_loc, tr)[0] for tr in trs]

        return list(set(ims).intersection(trs))

    def get_complete_fovs(self):
        """Returns a list of all fov indicies that have output files from this object."""
        compl = glob.glob(self.complete_csv_name.format("****"))
        compl = [parse.parse(self.complete_csv_name, comp)[0] for comp in compl]

        return compl

    def get_incomplete_fovs(self):
        """Returns a list of all fov indicies that still need to be run by this object."""
        all_fovs = self.get_all_fovs()
        compl = self.complete_fovs()

        return list(set(all_fovs) - set(compl))

    def blur_fov(self, f, min_size, dilate, sigma, thresh):
        """Run the first step of soft-segmentation, where masks are blurred and
        multiple float values assigned to each transcript, corresponding to which
        cells they may be members of and their relative likelihoods.

        f: fov number
        min_size: minimum size for eligible masks.
        dilate: radius to dilate masks by, in pixels.
        sigma: the sigma value for the gaussian blur.
        thresh: minimum value of a blurred mask pixel for a transcript to be eligible.

        writes to disk: self.complete_csv_name.format(f)
        returns: (tr, intensities)
         tr: dataframe of transcript information
         intensities: list of all assigned intensity values
        """
        t1 = time.time()
        im = skimage.io.imread(self.im_loc.format(f))
        # filter this before we do anything else to save time...
        im = self.size_filter(im, min_size=min_size)

        tr = pd.read_csv(self.csv_loc.format(f), index_col=0)
        tr = tr.reset_index()

        if len(tr) > 0:
            self.logger.info(f"[{datetime.datetime.now()}] starting fov_{f:04}...")

            if "cell_ids" in tr.keys():
                tr.drop(["cell_ids"], axis=1, inplace=True)

            tr["cell_ids"] = [{} for _ in range(len(tr))]

            if len(np.shape(im)) > 2:
                elig_z = [plane for plane in range(np.shape(0))]
                # going to assume that the z-axis is the 0th

            for m in np.unique(im):
                if m == 0:  # not a cell
                    continue

                # print(f"cell {m}")
                temp_im = im == m

                def fn(y):
                    return skimage.morphology.binary_dilation(
                        y, skimage.morphology.disk(dilate)
                    )

                if len(np.shape(temp_im)) == 2:  # two dimensional
                    filt = fn(temp_im)
                    filt = skimage.filters.gaussian(filt, sigma=sigma)
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
                else:  # three dimensional (presumably)
                    filt = np.array([fn(sl) for sl in temp_im])
                    filt = np.array(
                        [skimage.filters.gaussian(sl, sigma=sigma) for sl in filt]
                    )
                    # slices in image may not line up with data...
                    tr["cell_ids"] = tr.apply(
                        lambda row: (
                            row["cell_ids"]
                            if row["global_z"] not in elig_z
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

            tr.to_csv(self.complete_csv_name.format(f), index=False)

            self.logger.info(
                f"[{datetime.datetime.now()}] saved fov_{f:04}\n\t{len(tr)} transcripts, now unpacking intensities..."
            )

            tr = tr.set_index("index")

            intensities = []
            for i, row in tr.iterrows():
                for k, v in row["cell_ids"].items():
                    intensities.append(v)

            print(f"Completed fov_{f:04}.")
            print(
                f"\t{len(intensities)} assigned to cells, {len(tr)} transcripts total."
            )
            print(f"\ttime taken:{(time.time() - t1)/60} minutes.")

            return (tr, intensities)

        else:
            print(f"Skipping fov_{f:04}, no transcripts found.")
            return None

    def blur_all_fovs(self, min_size, dilate, sigma, thresh):
        """Run the first step of soft-segmentation, where masks are blurred and
        multiple float values assigned to each transcript, corresponding to which
        cells they may be members of and their relative likelihoods.

        This method will run on all eligible FOVs, using the multiprocessing pool.

        pool_size: the number of threads to be used.
        min_size: minimum size for eligible masks.
        dilate: radius to dilate masks by, in pixels.
        sigma: the sigma value for the gaussian blur.
        thresh: minimum value of a blurred mask pixel for a transcript to be eligible.

        writes to disk: self.complete_csv_name.format(f) for all FOVs.
        """
        sel_fovs = self.get_incomplete_fovs()
        with Pool(self.pool_size) as pool:
            pool.starmap(
                self.blur_fov,
                zip(
                    sel_fovs,
                    repeat(min_size),
                    repeat(dilate),
                    repeat(sigma),
                    repeat(thresh),
                ),
            )

    def combine_result_csvs(self):
        """Combines all soft-assignment csvs into one master file."""
        full_tr = None
        for f in self.get_complete_fovs():
            tr = pd.read_csv(self.complete_csv_name.format(f))
            if full_tr is None:
                full_tr = tr
            else:
                full_tr = pd.concat([full_tr, tr])

        full_tr.to_csv(f"{self.complete_loc}all_soft_cellids.csv", index=False)
        self.num_transcripts = len(full_tr)

    def calculate_confident_threshold(self, show_plots=False):
        """
        Finds the ideal threshold for what should be considered a 'confident' assignment.

        returns:
        "aggresive": defines threshold as elbow point of assigned transcripts vs
            multiply-assigned transcripts plot.
        "conservative": defines threshold as lowest value where no transcript is
            multiply-assigned.
        """
        if self.num_transcripts is not None:
            tr_tally = self.num_transcripts
        else:
            tr_tally = 10000
        n_wid = 5  # initial guess for max number of cells a transcript is assigned to
        cutoffs = np.empty((tr_tally, n_wid))
        c_row = 0

        sel_fovs = self.get_complete_fovs()

        for f in sel_fovs:
            tr = pd.read_csv(self.complete_csv_name.format(f))
            for r, row in tr.iterrows():
                assigned = ast.literal_eval(row["cell_ids"])
                ind = 0

                # make the mat wider if needed
                if np.shape(cutoffs)[1] < len(assigned):
                    n_wid = len(assigned) - np.shape(cutoffs)[1]
                    cutoffs = np.append(cutoffs, np.empty((tr_tally, n_wid)), axis=1)

                if np.shape(cutoffs)[0] == c_row:
                    cutoffs = np.append(cutoffs, np.empty((tr_tally, n_wid)), axis=0)

                # transfer in our floats...
                for k, v in assigned.items():
                    cutoffs[c_row, ind] = v
                    ind += 1
                c_row += 1
            self.logger.info(f"Just completed fov_{f:04}, current c_row {c_row}")

        assigned_tr = []
        dupe_tr = []
        num = 100  # resolution for estimate
        rang = np.linspace(0, 1, num=num)

        for thresh in rang:
            assd = cutoffs > thresh
            assigned_tr.append(np.nansum(assd))
            dupe_tr.append(np.nansum(assd[np.nansum(assd, axis=1) > 1]))

        dif = np.diff(np.diff(dupe_tr))

        if show_plots():
            fig, ax = plt.subplots()
            im = ax.scatter(assigned_tr, dupe_tr, c=rang)
            # ax.set_xscale("log")
            # ax.set_yscale("log")
            ax.set_xlabel("Total transcripts assigned")
            ax.set_ylabel("Number of transcripts assigned to more than one cell")
            fig.colorbar(im, ax=ax)
            plt.show(block=False)

            fig, ax = plt.subplots()
            im = ax.scatter(assigned_tr[2:], dif, c=rang[2:])
            # ax.set_xscale("log")
            # ax.set_yscale("log")
            ax.set_xlabel("Total transcripts assigned")
            ax.set_ylabel(
                "Number of transcripts assigned to more than one cell, 2nd derivative"
            )
            fig.colorbar(im, ax=ax)
            plt.show(block=False)

        percentile = np.argmax(dif)

        ass_per = [dupe_tr[i] / assigned_tr[i] * 100 for i in range(num)]
        fir = ass_per.index(0)

        if show_plots:
            plt.show()

        return {
            "aggresive": {
                "threshold": rang[percentile + 2],
                "total_assigned": assigned_tr[percentile + 2],
                "multi_assigned": dupe_tr[percentile + 2],
            },
            "conservative": {
                "threshold": rang[fir],
                "total_assigned": assigned_tr[fir],
                "multi_assigned": dupe_tr[fir],
            },
        }

    def convert_to_adata(self, fov_locs, thresh):
        """
        Converts all completed analyses to adata format.

        fov_locs: dict containing the start positions of each fov
        thresh: transcripts with value lower than this will not be retained.
        """
        cxg_dict = {}
        fovs = self.get_complete_fovs()
        for f in fovs:
            tr = pd.read_csv(self.complete_csv_name.format(f))
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
            self.logger.info(
                f"[{datetime.datetime.now()}] completed reading in fov_{f:04}."
            )

        cxg_df = pd.DataFrame.from_dict(cxg_dict, orient="index")
        cxg_df

        adata = ad.AnnData(cxg_df)
        adata.obs["fov"] = pd.DataFrame(np.zeros((len(adata), 1)))
        adata.obs["size"] = pd.DataFrame(np.zeros((len(adata), 1)))
        adata.obs["x_coords"] = pd.DataFrame(np.zeros((len(adata), 1)))
        adata.obs["y_coords"] = pd.DataFrame(np.zeros((len(adata), 1)))
        adata.obs["z_coords"] = pd.DataFrame(np.zeros((len(adata), 1)))

        def getContour(tif, i):
            binimg = deepcopy(tif)
            binimg[binimg != i] = 0
            binimg[binimg > 0] = 1
            binimg = binimg.astype("uint8")
            contours = []
            for n in range(np.shape(binimg)[0]):
                contours.append(
                    cv2.findContours(
                        binimg[n, :, :], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                    )[-2]
                )
            del binimg
            return contours, i

        for f in fovs:
            if Path(self.im_loc.format(f)).is_file():
                im = skimage.io.imread(self.im_loc.format(f))
                im = clear_border(im)
                with Pool(self.pool_size) as pool:
                    results = pool.starmap(
                        getContour, zip(repeat(im), list(set(im.flatten())))
                    )
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
                            + fov_locs[f]["x"][0]
                        )
                        adata.obs.loc[adata.obs.index.isin([str(k)]), "y_coords"] = (
                            int(sum([n["m01"] for n in moments]) / total_size)
                            + fov_locs[f]["y"][0]
                        )
                        adata.obs.loc[adata.obs.index.isin([str(k)]), "z_coords"] = (
                            sum([moments[ln]["m00"] * ln for ln in range(len(moments))])
                            / total_size
                        )
                    # print(adata[str(k)].obs)
                    del moments
                del all_contours
                del im
                self.logger.info(
                    f"[{datetime.datetime.now()}] completed converting fov_{f:04}."
                )
            else:
                self.logger.info(
                    f"[{datetime.datetime.now()}] skipped converting fov_{f:04}."
                )

        adata.X = np.nan_to_num(adata.X)

        adata.write(f"{self.complete_loc}cxg_adata.h5ad")

        return adata
