import ast
import glob
import itertools
import logging
import random
import time
from collections import Counter
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
import yaml
from matplotlib.patches import Circle
from skimage.measure import regionprops
from skimage.segmentation import clear_border
from tqdm import tqdm


def sigm(x, x0=0, k=0.05):
    return 1 / (1 + np.e ** (-1 * k * (x - x0)))


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
    return contours, i


def remove_border(im):
    result = []
    removed = set()
    for z in im:
        result.append(clear_border(z))
        removed.update(set(np.unique(z)) - set(np.unique(result[-1])))

    result = np.array(result)
    # print(np.shape(result))

    for r in removed:
        result[result == r] = 0

    return result


def flatten(lis):
    if all([not hasattr(e, "__len__") for e in lis]):
        return lis
    else:
        target = []
        for li in lis:
            if not hasattr(li, "__len__"):
                target.append(li)
            else:
                target.extend(flatten(li))
        return target


class SoftAssigner:
    def __init__(
        self,
        csv_loc,
        im_loc,
        complete_loc,
        pool_size=1,
        conf_thresh=0.7,
        decay_func=None,
    ):
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
        self.complete_csv_name = f"{complete_loc}fov_{{:0>4}}_cellids.csv"
        self.pool_size = pool_size
        self.logger = logging.getLogger()
        logging.basicConfig(
            filename=f"{complete_loc}{datetime.now()}run.log",
            level=logging.DEBUG,
        )
        self.conf_thresh = conf_thresh
        if decay_func is None:
            self.decay_func = sigm
        else:
            self.decay_func = decay_func

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
        compl = self.get_complete_fovs()

        return list(set(all_fovs) - set(compl))

    def random_complete_fov(self):
        fovs = self.get_complete_fovs()
        random.shuffle(fovs)
        while len(fovs) > 0:
            yield fovs.pop()

    def random_cell_in_fov(self, fov):
        """
        Yields random cell ids from cells that have transcripts mapped to them.
        """

        im = skimage.io.imread(self.im_loc.format(fov))
        tr = pd.read_csv(self.complete_csv_name.format(fov))

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
            trs = pd.read_csv(self.complete_csv_name.format(f))
            for i, row in trs.iterrows():
                for c, v in ast.literal_eval(row["cell_ids"]).items():
                    if int(c) in cell_ids:
                        results[int(c)] = f
                        if len(results) == len(cell_ids):
                            return results
        return results

    def plot_completed_cell(self, fov, cells, tr_hi=None):
        im = skimage.io.imread(self.im_loc.format(fov))
        tr = pd.read_csv(self.complete_csv_name.format(fov))

        # collect relevant transcripts
        sel_tr = []
        for r, row in tr.iterrows():
            assigned = ast.literal_eval(row["cell_ids"])
            if "lank" not in row["gene"] and any(
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
                        "gene": row["gene"],
                        "index": row["index"],
                    }
                )
        sel_tr = pd.DataFrame(sel_tr)

        # find our bounding box
        x_0 = np.nanmin(sel_tr["x"]) - 10
        x_1 = np.nanmax(sel_tr["x"]) + 10
        y_0 = np.nanmin(sel_tr["y"]) - 10
        y_1 = np.nanmax(sel_tr["y"]) + 10

        im8 = [(im == cell + 1).astype(np.uint8)[:, y_0:y_1, x_0:x_1] for cell in cells]

        # go through each z slice
        for z in range(np.shape(im8)[1]):
            clen = np.shape(im8)[0]

            c_valid = []
            cnt = []
            for c in range(clen):
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

            if not any(c_valid):
                print("No valid cells in this slice.")
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
                            raw_dist[i, j, c] = cv2.pointPolygonTest(
                                cnt[c][0], (j, i), True
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

            for cn in cnt:
                draw_im = cv2.drawContours(drawing, cn, 0, (1, 1, 1), 1)

            ax.imshow(draw_im)

            # go through and plot the relevant transcripts
            for r, row in sel_tr.iterrows():
                if int(row["z"]) == z:
                    facecolor = (row["c1"], row["c2"], row["c0"])
                    if tr_hi is not None:
                        for k, vs in tr_hi.items():
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

    def blur_fov(self, f, min_size, max_dist):
        """Run the first step of soft-segmentation, where masks are blurred and
        multiple float values assigned to each transcript, corresponding to which
        cells they may be members of and their relative likelihoods.

        f: fov number
        min_size: minimum size for eligible masks.
        max_dist: the maximum distance between a transcript and a mask to be considered eligible.

        writes to disk: self.complete_csv_name.format(f)
        returns: (tr, intensities)
         tr: dataframe of transcript information
         intensities: list of all assigned intensity values
        """
        t1 = time.time()
        im = skimage.io.imread(self.im_loc.format(f))
        # filter this before we do anything else to save time...
        im = size_filter(im, min_size=min_size)

        tr = pd.read_csv(self.csv_loc.format(f), index_col=0)
        tr = tr.reset_index()

        if len(tr) > 0:
            self.logger.info(f"[{datetime.now()}] starting fov_{f:0>4}...")

            # override if if it's alraedy there'
            if "cell_ids" in tr.keys():
                tr.drop(["cell_ids"], axis=1, inplace=True)

            # add empty column that we're about to populate'
            tr["cell_ids"] = [{} for _ in range(len(tr))]

            if len(np.shape(im)) > 2:
                elig_z = [plane for plane in range(np.shape(im)[0])]
                # going to assume that the z-axis is the 0th

            for m in np.unique(im):
                if m == 0:  # background, not a cell
                    continue
                temp_im = (im == m).astype(np.uint8)

                if len(np.shape(im)) == 2:  # two dimensional
                    cnt, _ = cv2.findContours(
                        temp_im, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if len(cnt) > 1:
                        self.logger.info(
                            f"WARNING! Cell ID {m-1} has {len(cnt)} total contours, only using the first."
                        )
                    elif len(cnt) == 0:
                        self.logger.info(
                            f"WARNING! Cell ID {m-1} has no eligible contours. Skipping cell ID {m-1}."
                        )
                        continue

                    tr["cell_ids"] = tr.apply(
                        lambda row: (
                            row["cell_ids"]  # default to keeping the dict the same
                            if cv2.pointPolygonTest(
                                cnt[0], (row["x"] - 1, row["y"] - 1), True
                            )
                            < -1
                            * max_dist  # if this assignment distance passes our threshold
                            else dict(  # add this index to the dict of eligible assignments
                                row["cell_ids"],
                                **{
                                    str(
                                        m - 1
                                    ): self.decay_func(  # this is where we account for images being 1-indexed
                                        cv2.pointPolygonTest(
                                            cnt[0], (row["x"] - 1, row["y"] - 1), True
                                        )
                                    )
                                },
                            )
                        ),
                        axis=1,
                    )
                else:  # three dimensional (presumably)
                    # slices in image may not line up with data...
                    cntz = {}
                    for z in elig_z:
                        cnt, _ = cv2.findContours(
                            temp_im[z], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                        )
                        if len(cnt) > 1:
                            self.logger.info(
                                f"WARNING! Cell ID {m-1} zslice {z} has {len(cnt)} total contours, only using the first."
                            )
                        elif len(cnt) == 0:
                            self.logger.info(
                                f"WARNING! Cell ID {m-1} zslice {z} has no eligible contours. Skipping zslice {z}."
                            )
                            continue

                        cntz[z] = cnt[0]

                    tr["cell_ids"] = tr.apply(
                        lambda row: (
                            row["cell_ids"]  # default to keeping the dict the same
                            if row["global_z"] not in elig_z
                            or row["global_z"] not in cntz.keys()
                            or cv2.pointPolygonTest(
                                cntz[row["global_z"]],
                                (row["x"] - 1, row["y"] - 1),
                                True,
                            )
                            < -1
                            * max_dist  # if this assignment is eligible by its zslice and also passes the threshold
                            else dict(  # add this index to the dict of eligible assignments
                                row["cell_ids"],
                                **{
                                    str(
                                        m - 1
                                    ): self.decay_func(  # this is where we account for images being 1-indexed
                                        cv2.pointPolygonTest(
                                            cntz[row["global_z"]],
                                            (row["x"] - 1, row["y"] - 1),
                                            True,
                                        )
                                    )
                                },
                            )
                        ),
                        axis=1,
                    )
            del im

            tr.to_csv(self.complete_csv_name.format(f), index=False)

            self.logger.info(
                f"[{datetime.now()}] saved fov_{f:0>4}\n\t{len(tr)} transcripts, now unpacking intensities..."
            )

            tr = tr.set_index("index")

            intensities = []
            for i, row in tr.iterrows():
                for k, v in row["cell_ids"].items():
                    intensities.append(v)

            print(f"Completed fov_{f:0>4}.")
            print(
                f"\t{len(intensities)} assigned to cells, {len(tr)} transcripts total."
            )
            print(f"\ttime taken:{(time.time() - t1)/60} minutes.")

            return (tr, intensities)

        else:
            print(f"Skipping fov_{f:0>4}, no transcripts found.")
            return None

    def blur_all_fovs(self, min_size, max_dist, sel_fovs=None):
        """Run the first step of soft-segmentation, where masks are blurred and
        multiple float values assigned to each transcript, corresponding to which
        cells they may be members of and their relative likelihoods.

        This method will run on all eligible FOVs, using the multiprocessing pool.

        pool_size: the number of threads to be used.
        min_size: minimum size for eligible masks.
        max_dist: the maximum distance between a transcript and a mask to be considered eligible.

        writes to disk: self.complete_csv_name.format(f) for all FOVs.
        """
        if sel_fovs is None:
            sel_fovs = self.get_incomplete_fovs()
        with Pool(self.pool_size) as pool:
            pool.starmap(
                self.blur_fov,
                zip(sel_fovs, repeat(min_size), repeat(max_dist)),
            )

    def combine_soft_csvs(self):
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
            if (min_thresh is None or float(v) / total > min_thresh) and float(v) > 0.5:
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
            tr = pd.read_csv(self.complete_csv_name.format(f))
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

    def convert_to_adata(self, fov_locs, min_thresh=None):
        """
        Converts all completed analyses to adata format.

        Note that this involves converting the transcript table into
        a cell by gene matrix. For this reason, min_thresh is used
        to decide which of the multiple possible assignments are valid.

        fov_locs: dict containing the start positions of each fov
        min_thresh: transcripts with value lower than this will not be retained.
        """
        cxg_dict = {}
        fovs = self.get_complete_fovs()
        for f in fovs:
            tr = pd.read_csv(self.complete_csv_name.format(f))
            for i, row in tr.iterrows():
                assigned = ast.literal_eval(row["cell_ids"])
                k = self.assign_to_cell(assigned, min_thresh)
                if k is not None:
                    if k in cxg_dict.keys():
                        if row["gene"] in cxg_dict[k].keys():
                            cxg_dict[k][row["gene"]] += 1
                        else:
                            cxg_dict[k][row["gene"]] = 1
                    else:
                        cxg_dict[k] = {row["gene"]: 1}
            # print(cxg_dict)
            del tr
            self.logger.info(f"[{datetime.now()}] completed reading in fov_{f:0>4}.")

        cxg_df = pd.DataFrame.from_dict(cxg_dict, orient="index")
        cxg_df

        adata = ad.AnnData(cxg_df)
        adata.obs["fov"] = pd.DataFrame(np.zeros((len(adata), 1)))
        adata.obs["size"] = pd.DataFrame(np.zeros((len(adata), 1)))
        adata.obs["x_coords"] = pd.DataFrame(np.zeros((len(adata), 1)))
        adata.obs["y_coords"] = pd.DataFrame(np.zeros((len(adata), 1)))
        adata.obs["z_coords"] = pd.DataFrame(np.zeros((len(adata), 1)))

        with pd.option_context("display.max_seq_items", None):
            self.logger.debug(f"All valid cell ids: {adata.obs_names}")

        for f in fovs:
            if Path(self.im_loc.format(f)).is_file():
                im = skimage.io.imread(self.im_loc.format(f))
                im = remove_border(im)
                with Pool(self.pool_size) as pool:
                    results = pool.starmap(getContour, zip(repeat(im), np.unique(im)))
                # THIS is where we account for the fact that image IDs are 1-indexed
                all_contours = {cs[1] - 1: cs[0] for cs in results if cs is not None}

                for k, v in all_contours.items():
                    if k == -1 or str(k) not in adata.obs_names:  # don't run on bg
                        self.logger.debug(f"[{datetime.now()}] cell id {k} not valid.")
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
                    f"[{datetime.now()}] completed converting fov_{f:0>4}."
                )
            else:
                self.logger.info(f"[{datetime.now()}] skipped converting fov_{f:0>4}.")

        adata.X = np.nan_to_num(adata.X)

        adata.write(f"{self.complete_loc}cxg_adata.h5ad")

        return adata

    def get_scoring_matrix(self, adata, cats, normed=True):
        """
        Need to run this before evaluate_overlapping_regions
        cats is the possible cell types as they are described in adata
        formatted as
        { "column name":["cell type", "cell type"], ...}
        """
        filtered = {}
        for k, v in cats.items():
            for cat in v:
                filtered[cat] = deepcopy(adata[adata.obs[k] == cat])

        avgs = {}
        for k, v in filtered.items():
            avgs[k] = np.average(v.X, axis=0)

        df_avg = pd.DataFrame.from_dict(avgs, orient="index", columns=adata.var.index)
        if normed:
            norm_total = sum([len(x) for k, x in filtered.items()])
            df_normed = df_avg.multiply(
                [(norm_total - len(x)) / norm_total * 100 for x in filtered.values()],
                axis=0,
            )

            self.df_comp = df_normed
        else:
            self.df_comp = df_avg

    def evaluate_overlapping_regions(self, adata, cats, sel_fovs=None, min_thresh=None):
        def extract_celltype(cats, row):
            this_type = [
                row[n].values[0] for n, m in cats.items() if row[n].values[0] in m
            ]
            if len(this_type) > 1:
                print(
                    f"Warning! Cell {row.index} has multiple valid celltypes: {this_type}. Discarding..."
                )
                return
            if len(this_type) > 0:
                return this_type[0]

        if sel_fovs is None:
            sel_fovs = self.get_complete_fovs()

        all_reass = {}
        results = {}
        dupe_ass = {}
        tr_writer = open(f"{self.complete_loc}{datetime.now()}_changed_trs.csv", "w")
        tr_writer.write("transcript ID,cell ID\n")

        dict_loc = f"{self.complete_loc}{datetime.now()}_overlap_eval.pydict"

        fov_tally = 0

        for f in sel_fovs:
            dupe_ass[f] = {}
            tr = pd.read_csv(self.complete_csv_name.format(f))
            tr = tr.reset_index()
            im = skimage.io.imread(self.im_loc.format(f))

            self.logger.info(
                f"[{datetime.now()}] evaluate_overlapping_regions starting fov_{f:0>4}..."
            )

            print(f"Starting fov_{f:0>4} ({100*fov_tally/len(sel_fovs):.2f}%):")
            print("\treading transcripts...")
            fov_tally += 1

            # first step: identify regions where transcripts have the potential to be
            # assigned to more than one cell.
            with tqdm(total=len(tr)) as pbar:
                for r, row in tr.iterrows():
                    pbar.update(1)

                    # omit blanks
                    if "lank" in row["gene"]:
                        continue

                    asst = ast.literal_eval(row["cell_ids"])
                    # keys are cell ids, values are assigned likelihoods
                    if len(asst.keys()) > 1:
                        # first, filter out any cell ids that got filtered before labelling
                        elg_cells = [k for k in asst.keys() if k in adata.obs.index]
                        self.logger.debug(f"{asst} {elg_cells}")

                        # now, select for transcripts above threshold
                        # these are the only things eligible to be assigned to
                        pos_keys = tuple(
                            sorted(
                                [
                                    k
                                    for k in elg_cells
                                    # if (min_thresh is None or asst[k] > min_thresh)
                                    # ^ this is from when we had a seperate threshold
                                    # from the confident one. We never actually used it.
                                ]
                            )
                        )

                        # if it's not eligible to be assigned to anything, we don't care...
                        if len(pos_keys) == 0:
                            continue

                        # next, see if this is a comparison that we even care about
                        # need at least one of these cells to be a different type
                        types = [
                            extract_celltype(cats, adata.obs.loc[[i]])
                            for i in elg_cells
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
            print("\tpruning comparisons...")
            with tqdm(total=len(dupe_ass[f])) as pbar:
                for k, v in dupe_ass[f].items():
                    pbar.update(1)
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

            self.logger.info(
                f"[{datetime.now()}] complete set of duplicated regions for fov_{f:0>4}\n{yaml.dump(dupe_ass[f])}"
            )
            self.logger.debug(
                f"[{datetime.now()}] cell combos removed for fov_{f:0>4}: {to_del}"
            )

            # find and save all confident assignments, to avoid
            # repeated iteration for each pairing
            print("\tidentifying confident transcripts...")
            conf_tr = {}
            with tqdm(total=len(tr)) as pbar:
                for r, row in tr.iterrows():
                    pbar.update(1)
                    if "lank" not in row["gene"]:
                        asst = ast.literal_eval(row["cell_ids"])
                        target = self.assign_to_cell(asst, min_thresh)
                        if target is not None:
                            if target in conf_tr.keys():
                                conf_tr[target].append(row["gene"])
                            else:
                                conf_tr[target] = [row["gene"]]

            # this only matters within an FOV
            all_reass_check = set()

            # if we don't have any comparisons, skip ahead...
            if len(dupe_ass[f].keys()) == 0:
                print("\tno comparisons to be made, skipping FOV.")
                continue

            # start with 2 way comparisons, working our way up to n-way
            max_comp = max([len(k) for k in dupe_ass[f].keys()])
            comp_tally = 0

            print("\tperforming n-way comparisons...")

            with tqdm(total=len(dupe_ass[f])) as pbar:
                for ln in range(2, max_comp + 1):
                    # print(l)
                    for k, gene_lis in dupe_ass[f].items():
                        if len(k) != ln:
                            continue

                        reass_inds = {}

                        # get our celltypes
                        types = [extract_celltype(cats, adata.obs.loc[[i]]) for i in k]

                        # need to go through and retrieve the confident transcripts
                        conf_inds = []
                        for v in range(ln):
                            temp = []
                            if k[v] in conf_tr.keys():
                                for gene in conf_tr[k[v]]:
                                    if gene in self.df_comp.columns:
                                        t = self.df_comp[gene].copy()
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
                                    temp = self.df_comp[gene].copy()
                                    temp = temp.loc[types]
                                    if len(conf_inds[t]) > 0:
                                        conf_inds[t] = pd.concat(
                                            [conf_inds[t], temp], axis=1
                                        )
                                    else:
                                        conf_inds[t] = temp
                            else:
                                # this is the overlap zone
                                amb_inds[inds] = []
                                reass_inds[inds] = []
                                # print(f"{inds} all genes: {genes}")
                                for gene, index in genes:
                                    if (
                                        gene in self.df_comp.columns
                                    ):  # possible for genes to get dropped in filtering
                                        # if this has been reassigned elsewhere, go with that
                                        if index in all_reass.keys():
                                            c = all_reass[index]
                                            cell_ind = [
                                                ks for ks in range(len(k)) if k[ks] == c
                                            ]
                                            if len(cell_ind) != 1:
                                                self.logger.info(
                                                    f"Warning! Gene {index} has been assigned to {all_reass[index]}, which is not one of the eligible cells in this comparison!"
                                                )
                                                continue
                                            temp = self.df_comp[gene].copy()
                                            temp = temp.loc[types]
                                            if len(conf_inds[cell_ind[0]]) > 0:
                                                conf_inds[cell_ind[0]] = pd.concat(
                                                    [conf_inds[cell_ind[0]], temp],
                                                    axis=1,
                                                )
                                            else:
                                                conf_inds[cell_ind[0]] = temp
                                        else:
                                            temp = self.df_comp[gene].copy()
                                            temp = temp.loc[types]
                                            amb_inds[inds].append(temp)
                                            reass_inds[inds].append(index)

                        ov_check = []
                        for ind, v in reass_inds.items():
                            ov_check.extend(v)
                        overlap = set.intersection(all_reass_check, set(ov_check))
                        if len(overlap) > 0:
                            self.logger.info(
                                f"Warning! Transcripts {overlap} have already been reassigned!"
                            )

                        top_val = 0
                        winning_combo = []

                        for combo in itertools.product(
                            *[(*cellid, "Original") for cellid in amb_inds.keys()]
                        ):
                            sel_vals = {}
                            for v in range(len(conf_inds)):
                                if (
                                    len(conf_inds[v]) > 0
                                ):  # not interested if there aren't any
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
                                    if len(amb_inds[sel_keyset]) == 0:
                                        self.logger.info(
                                            f"Combo {sel_keyset} has no ambiguous transcripts."
                                        )
                                        continue

                                    t_df = pd.concat(amb_inds[sel_keyset], axis=1)

                                    n_val = t_df.values[ind]
                                    if cell_ind in sel_vals.keys():
                                        if hasattr(n_val, "__len__"):
                                            sel_vals[cell_ind].extend(n_val)
                                        else:
                                            sel_vals[cell_ind].append(n_val)
                                    else:
                                        # THIS was what fixed the nested list problem
                                        # TODO: figure out why this matters!!
                                        if hasattr(n_val, "__len__"):
                                            sel_vals[cell_ind] = n_val
                                        else:
                                            sel_vals[cell_ind] = [n_val]
                                else:
                                    for tr_tup in gene_lis[sel_keyset]:
                                        tr_id = tr_tup[1]
                                        row = tr.loc[tr["index"] == int(tr_id)]
                                        if len(np.shape(im)) == 2:
                                            one_cell_ind = (
                                                im[row["y"] - 1, row["x"] - 1][0] - 1
                                            )
                                        else:
                                            one_cell_ind = (
                                                im[
                                                    row["global_z"],
                                                    row["y"] - 1,
                                                    row["x"] - 1,
                                                ][0]
                                                - 1
                                            )
                                        if str(one_cell_ind) in list(sel_keyset):
                                            ind = list(sel_keyset).index(
                                                str(one_cell_ind)
                                            )
                                        else:
                                            if one_cell_ind > 0:
                                                self.logger.info(
                                                    f"Warning! Original cell assignment {one_cell_ind} for transcript {tr_id} not in possible comparison list {sel_keyset}!"
                                                )
                                            continue

                                        if tr_tup[0] in self.df_comp.columns:
                                            t_df = self.df_comp[tr_tup[0]].copy()
                                            t_df = t_df.loc[types]

                                            n_val = t_df.values[ind]
                                            if cell_ind in sel_vals.keys():
                                                if hasattr(n_val, "__len__"):
                                                    sel_vals[cell_ind].extend(n_val)
                                                else:
                                                    sel_vals[cell_ind].append(n_val)
                                            else:
                                                sel_vals[cell_ind] = [n_val]

                            everything = [
                                v
                                for row in [val for k, val in sel_vals.items()]
                                for v in row
                            ]

                            # Below should never be run, this bug has been isolated.
                            if any([hasattr(e, "__len__") for e in everything]):
                                print("Erroneous nested list!")
                                for k, v in sel_vals.items():
                                    if any([hasattr(m, "__len__") for m in v]):
                                        print(f"{k}:{v}")

                            # again, this flatten *shouldn't* matter, but leaving it in as a safety anyways.'
                            everything = flatten(everything)
                            avg = np.average(everything)

                            if avg > top_val:
                                winning_combo = combo
                                top_val = avg

                        comp_tally += 1
                        pbar.update(1)
                        # if comp_tally % 20 == 0:
                        #     print(f"\t{comp_tally}/{len(dupe_ass[f])}")
                        # print(amb_inds.keys())

                        key_ref = list(reass_inds.keys())
                        for v in range(len(winning_combo)):
                            for ge in reass_inds[key_ref[v]]:
                                all_reass[ge] = winning_combo[v]
                                if winning_combo[v] != "Original":
                                    tr_writer.write(f"{ge},{all_reass[ge]}\n")
                            if winning_combo[v] != "Original":
                                results[key_ref[v]] = extract_celltype(
                                    cats, adata.obs.loc[[winning_combo[v]]]
                                )
                            else:
                                results[key_ref[v]] = "Original"

                        # if a_vals.mean() > b_vals.mean():
                        #    results[k] = gene_inds.index[0]
                        #    all_reass.update({i:k[0] for i in reass_inds}) # gene ind : cell ind

                        for k, v in reass_inds.items():
                            all_reass_check.update(set(v))

                        # we'll just manually re-save this after each FOV finishes
                        # in the event of error we'll still have something'
                        with open(dict_loc, "w") as fl:
                            fl.write(str(results))

        tr_writer.close()
        return results

    def update_reassigned_segs(
        self, changed_tr, ann_adata, min_thresh=None, ncol_name="reassigned"
    ):
        """
        Given a dataframe of changed transcripts, create an updated anndata object
        with the new gene by cell assignments. Also adds a column by the name of
        ncol_name to each transcript table.
        """

        cxg_dict = {}
        fovs = self.get_complete_fovs()

        # first, we convert changed_tr from a dataframe to a dict,
        # which saves us a TON of time on lookup later.

        tr_dict = {}
        for i, row in changed_tr.iterrows():
            tr_dict[row["transcript ID"]] = row["cell ID"]

        for f in fovs:
            tr = pd.read_csv(self.complete_csv_name.format(f))
            self.logger.info(f"[{datetime.now()}] starting fov_{f:0>4}...")

            ncol = []
            for i, row in tr.iterrows():
                assigned = ast.literal_eval(row["cell_ids"])
                k = self.assign_to_cell(assigned, min_thresh)

                # if this is a transcript we reassigned, we need to update that
                if row["index"] in tr_dict.keys():
                    k = tr_dict[row["index"]]

                if k is not None:
                    k = int(k)  # reading in from ast will be a string
                    ncol.append(k)
                    if k in cxg_dict.keys():
                        if row["gene"] in cxg_dict[k].keys():
                            cxg_dict[k][row["gene"]] += 1
                        else:
                            cxg_dict[k][row["gene"]] = 1
                    else:
                        cxg_dict[k] = {row["gene"]: 1}
                else:
                    ncol.append(np.nan)
            tr[ncol_name] = ncol

            # save modified transcript table
            tr.to_csv(self.complete_csv_name.format(f), sep=",")
            del tr

        cxg_df = pd.DataFrame.from_dict(cxg_dict, orient="index")
        new_adata = ad.AnnData(cxg_df)

        self.logger.debug(f"[{datetime.now()}] Converted new matrix to anndata.")

        # now, cleaning it up to match the annotated anndata
        col_oi = [col for col in ann_adata.obs.columns if "count" not in col]
        sel_cells = []
        for cell in new_adata.obs.index.to_list():
            if cell in ann_adata.obs.index:
                sel_cells.append(cell)

        new_adata = new_adata[[cell in sel_cells for cell in new_adata.obs.index]]
        for col in col_oi:
            new_adata.obs[col] = pd.DataFrame(np.zeros((len(new_adata), 1)))
            new_adata.obs.loc[sel_cells, col] = ann_adata.obs.loc[sel_cells, col]

        dat = datetime.today().strftime("%Y%m%d")
        filename = f"{self.complete_loc}cxg_adata_resegmented_{dat}.h5ad"
        new_adata.write_h5ad(filename)
        print(f"Saved {filename}")
