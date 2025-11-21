import ast
import glob
import logging
import multiprocessing
import random
from collections import Counter
from contextlib import closing
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
from matplotlib.patches import Circle
#  from memory_profiler import profile
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

    def plot_completed_cell(self, fov, cells, dist_between_slices=None, tr_hi=None):
        """
        tr_hi formatting:
        OPTIONAL dict key: color to use, values are dicts below:
        dict key: column to compare to transcript dataframe
           value: single value to match against OR list of values
        """
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
                            if len(np.unique(im8[c][z - 1, :, :])) == 2:
                                c_neighbor[i] = True
                                c_valid[i] = True
                                cnt[i] = cv2.findContours(
                                    im8[c][z - 1, :, :],
                                    cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE,
                                )[0]
                        if z < np.shape(im8[c])[0] - 1:
                            # check z+1
                            if len(np.unique(im8[c][z + 1, :, :])) == 2:
                                c_neighbor[i] = True
                                c_valid[i] = True
                                cnt[i] = cv2.findContours(
                                    im8[c][z + 1, :, :],
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

    def blur_fov(
        self, f, min_size, max_dist, dist_between_slices=None, disable_tqdm=False
    ):
        """Run the first step of soft-segmentation, where masks are blurred and
        multiple float values assigned to each transcript, corresponding to which
        cells they may be members of and their relative likelihoods.

        f: fov number
        min_size: minimum size for eligible masks.
        max_dist: the maximum distance between a transcript and a mask to be considered eligible.
        dist_between_slices: if provided and the images are 3d, slices with no contours
         within 1 zslice of a slice with a valid contour will project that contour with
         this added distance

        writes to disk: self.complete_csv_name.format(f)
        returns: (tr, intensities)
         tr: dataframe of transcript information
         intensities: list of all assigned intensity values
        """
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

            if "global_z" in tr.columns:
                z_global = list(tr["global_z"])
            x = list(tr["x"])
            y = list(tr["y"])
            cell_ids = list(tr["cell_ids"])

            all_masks = np.unique(im)
            if not disable_tqdm:
                pbar = tqdm(total=len(all_masks))

            for m in all_masks:
                if m == 0:  # background, not a cell
                    if not disable_tqdm:
                        pbar.update(1)
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

                    for j in range(len(tr)):
                        dist = cv2.pointPolygonTest(
                            cnt[0],
                            (x[j] - 1, y[j] - 1),
                            True,
                        )
                        if dist > -1 * max_dist:
                            cell_ids[j] = dict(
                                cell_ids[j], **{str(m - 1): self.decay_func(dist)}
                            )
                else:  # three dimensional (presumably)
                    # slices in image may not line up with data...
                    cntz = {}
                    cntz_best_neighbor = {}
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
                                f"WARNING! Cell ID {m-1} zslice {z} has no eligible contours. Skipping zslice {z} for now."
                            )
                            if dist_between_slices is not None:
                                cntz_best_neighbor[z] = None
                            continue

                        cntz[z] = cnt[0]

                    if dist_between_slices is not None:
                        for z in list(cntz_best_neighbor.keys()):
                            # realistically we should not have a case where there are two neighboring slices
                            # that both have contours surrounding a slice with no contours. So just do basic check.
                            if z - 1 in cntz.keys():
                                cntz_best_neighbor[z] = cntz[z - 1]
                                self.logger.info(
                                    f"Assigning cell ID {m-1} zslice {z} nearest neighbor as {z-1}."
                                )
                            elif z + 1 in cntz.keys():
                                cntz_best_neighbor[z] = cntz[z + 1]
                                self.logger.info(
                                    f"Assigning cell ID {m-1} zslice {z} nearest neighbor as {z+1}."
                                )
                            else:
                                del cntz_best_neighbor[z]
                                self.logger.info(
                                    f"WARNING! Cell ID {m-1} zslice {z} has no eligible neighbors. Skipping zslice {z}."
                                )

                    for j in range(len(tr)):
                        if z_global[j] in elig_z:
                            dist = None
                            if z_global[j] in cntz.keys():
                                dist = cv2.pointPolygonTest(
                                    cntz[z_global[j]],
                                    (x[j] - 1, y[j] - 1),
                                    True,
                                )
                            elif z_global[j] in cntz_best_neighbor.keys():
                                dist = cv2.pointPolygonTest(
                                    cntz_best_neighbor[z_global[j]],
                                    (x[j] - 1, y[j] - 1),
                                    True,
                                )
                                if dist < 0:
                                    dist = -1 * np.sqrt(
                                        dist**2 + dist_between_slices**2
                                    )
                                else:
                                    dist = max(
                                        -1 * np.sqrt(dist**2 + dist_between_slices**2),
                                        dist_between_slices * -1,
                                    )

                            if dist is not None and dist > -1 * max_dist:
                                cell_ids[j] = dict(
                                    cell_ids[j], **{str(m - 1): self.decay_func(dist)}
                                )
                if not disable_tqdm:
                    pbar.update(1)
            del im

            if not disable_tqdm:
                pbar.close()

            tr["cell_ids"] = cell_ids
            tr.to_csv(self.complete_csv_name.format(f), index=False)

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
        self, min_size, max_dist, dist_between_slices=None, sel_fovs=None
    ):
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

        with tqdm(total=len(sel_fovs)) as pbar:
            with closing(Pool(self.pool_size)) as pool:
                results = pool.imap_unordered(
                    self.__func_wrapper__,
                    zip(
                        repeat(self.blur_fov),
                        sel_fovs,
                        repeat(min_size),
                        repeat(max_dist),
                        repeat(dist_between_slices),
                        repeat(True),
                    ),
                )
                for result in results:
                    pbar.update(1)

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

    def convert_to_adata(
        self,
        min_thresh=None,
        assigned_col=None,
        fov_locs=None,
        prev_adata=None,
        sel_fovs=None,
    ):
        """
        Converts all completed analyses to adata format.

        There are two different ways that two aspects of this can be run.

        When determining what transcripts go into what cells:
        - If assigned_col is provided, will pull the cell id from that column.
        - Otherwise, self.assign_to_cell will be passed the "cell_ids" column.
          If min_thresh is provided, that will be passed on to that method.

        When labelling cells with location metadata:
        - If prev_adata is provided, all columns from adata.obs that do not contain the
          word "coun" will be transferred to the cell with the same id in the new adata.
        - If fov_locs is provided, images will be loaded in and cell centroids will be
          calculated from contours in that image.
        Note that one of these two variables MUST be provided.
        """

        if fov_locs is None and prev_adata is None:
            raise ValueError("One of fov_locs or prev_adata must be provided.")

        cxg_dict = {}
        if sel_fovs is None:
            sel_fovs = self.get_complete_fovs()

        print("Reading cell by gene data from transcript tables.")
        for f in tqdm(sel_fovs):
            tr = pd.read_csv(self.complete_csv_name.format(f))
            self.logger.info(f"[{datetime.now()}] started reading in fov_{f:0>4}.")

            if assigned_col is not None:
                if assigned_col not in tr.columns:
                    self.logger.info(
                        f"{assigned_col} not in df for fov_{f:0>4}, skipping."
                    )
                    continue
                tr = tr[~pd.isnull(tr[assigned_col])]
                tallies = Counter(list(zip(tr["gene"], tr[assigned_col])))
            else:
                cell_col = tr.cell_ids.apply(
                    lambda x: self.assign_to_cell(ast.literal_eval(x), min_thresh)
                )
                inds = cell_col.apply(lambda x: x is not None)
                tallies = Counter(list(zip(tr["gene"][inds], cell_col[inds])))

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
        cxg_df

        adata = ad.AnnData(cxg_df)

        if prev_adata is not None:
            col_oi = [col for col in prev_adata.obs.columns if "count" not in col]
            sel_cells = []
            for cell in tqdm(adata.obs.index.to_list()):
                if cell in prev_adata.obs.index:
                    sel_cells.append(cell)

            adata = adata[[cell in sel_cells for cell in adata.obs.index]]
            for col in col_oi:
                adata.obs[col] = pd.DataFrame(np.zeros((len(adata), 1)))
                adata.obs.loc[sel_cells, col] = prev_adata.obs.loc[sel_cells, col]
        else:
            adata.obs["fov"] = pd.DataFrame(np.zeros((len(adata), 1)))
            adata.obs["size"] = pd.DataFrame(np.zeros((len(adata), 1)))
            adata.obs["x_coords"] = pd.DataFrame(np.zeros((len(adata), 1)))
            adata.obs["y_coords"] = pd.DataFrame(np.zeros((len(adata), 1)))
            adata.obs["z_coords"] = pd.DataFrame(np.zeros((len(adata), 1)))

            with pd.option_context("display.max_seq_items", None):
                self.logger.debug(f"All valid cell ids: {adata.obs_names}")

            print("Finding cell centroid data and copying to anndata object")
            for f in tqdm(sel_fovs):
                if Path(self.im_loc.format(f)).is_file():
                    im = skimage.io.imread(self.im_loc.format(f))
                    im = remove_border(im)
                    with Pool(self.pool_size) as pool:
                        results = pool.starmap(
                            getContour, zip(repeat(im), np.unique(im))
                        )
                    # THIS is where we account for the fact that image IDs are 1-indexed
                    all_contours = {
                        cs[1] - 1: cs[0] for cs in results if cs is not None
                    }

                    for k, v in all_contours.items():
                        if k == -1 or str(k) not in adata.obs_names:  # don't run on bg
                            self.logger.debug(
                                f"[{datetime.now()}] cell id {k} not valid."
                            )
                            continue
                        moments = []
                        for n in range(len(v)):
                            if len(v[n]) > 0:
                                moments.append(cv2.moments(v[n][0]))
                            else:
                                moments.append({"m00": 0, "m01": 0, "m10": 0})

                        total_size = sum([n["m00"] for n in moments])

                        adata.obs.loc[adata.obs.index.isin([str(k)]), "fov"] = f
                        adata.obs.loc[adata.obs.index.isin([str(k)]), "size"] = (
                            total_size
                        )
                        if total_size != 0:
                            adata.obs.loc[
                                adata.obs.index.isin([str(k)]), "x_coords"
                            ] = (
                                int(sum([n["m10"] for n in moments]) / total_size)
                                + fov_locs[f]["x"][0]
                            )
                            adata.obs.loc[
                                adata.obs.index.isin([str(k)]), "y_coords"
                            ] = (
                                int(sum([n["m01"] for n in moments]) / total_size)
                                + fov_locs[f]["y"][0]
                            )
                            adata.obs.loc[
                                adata.obs.index.isin([str(k)]), "z_coords"
                            ] = (
                                sum(
                                    [
                                        moments[ln]["m00"] * ln
                                        for ln in range(len(moments))
                                    ]
                                )
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
                    self.logger.info(
                        f"[{datetime.now()}] skipped converting fov_{f:0>4}."
                    )

        adata.X = np.nan_to_num(adata.X)

        dat = datetime.today().strftime("%Y%m%d_%H%M")
        filename = f"{self.complete_loc}cxg_adata_{dat}"
        if assigned_col is not None:
            filename += "_resegmented"
        filename += ".h5ad"

        adata.write(filename)
        print(f"Saved {filename}")
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
        """
        tr = pd.read_csv(self.complete_csv_name.format(f), index_col=0)
        if assigned_col in tr.columns:
            self.logger.info(
                f"[{datetime.now()}] skipping fov_{f:0>4}, {assigned_col} already present"
            )
            return

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
            if omit_blanks and "lank" in row["gene"]:
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
                self.tr_to_gene[index] = row["gene"]
                if conf_cell is not None:
                    # transcript has confident assignment
                    if conf_cell in conf_trs.keys():
                        conf_trs[conf_cell].append(index)
                    else:
                        conf_trs[conf_cell] = [index]
                else:
                    # dealing with a non-confident transcript
                    unconf_tup = tuple(sorted([k for k in assigned.keys()]))
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
                            continue
                        else:
                            skip_cached.append(unconf_tup)
                            self.logger.info(
                                f"Throwing out tuple {unconf_tup}, contains single cell type."
                            )
                            continue

                    if unconf_tup not in unconf_trs.keys():
                        unconf_trs[unconf_tup] = {}

                    # adding values of each possible assignment to proper dict entry
                    for cell, prob in assigned.items():
                        tr_val = (index, prob)
                        if cell in unconf_trs[unconf_tup].keys():
                            unconf_trs[unconf_tup][cell].append(tr_val)
                        else:
                            unconf_trs[unconf_tup][cell] = [tr_val]

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

        seg_is_default = {}
        # key: unconf_tup
        # value: True if using original masks, False if using novel mask

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

        if pbar is not None:
            pbar.close()

        self.logger.info(
            f"[{datetime.now()}] saving pydict and updating tr for fov_{f:0>4}"
        )

        # saving dict of transcripts that we have reassigned
        dict_loc = f"{self.complete_loc}overlap_eval_{assigned_col}_fov_{f:0>4}.pydict"
        with open(dict_loc, "w") as fl:
            fl.write(str(assigned_trs))

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

        # last bit of cleanup before we save it:
        tr = tr.loc[:, ~tr.columns.str.contains("^Unnamed")]
        tr.to_csv(self.complete_csv_name.format(f), sep=",")

        # save deltas while we're here
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

            dict_loc = f"{self.complete_loc}delta_tallies_{assigned_col}_fov_{f:0>4}.pydict"
            with open(dict_loc, "w") as fl:
                fl.write(str(assigned_trs))

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

    def evaluate_all_overlapping_regions(
        self,
        sel_fovs=None,
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

        with tqdm(total=len(sel_fovs)) as pbar:
            for subset_fovs in fov_pool:
                self.logger.info(f"[{datetime.now()}] starting sub-pool: {subset_fovs}")
                with closing(Pool(processes=self.pool_size)) as pool:
                    results = pool.imap_unordered(
                        self.__func_wrapper__,
                        zip(
                            repeat(self.evaluate_overlapping_regions_single_fov),
                            subset_fovs,
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
                        ),
                    )
                    for result in results:
                        pbar.update(1)
