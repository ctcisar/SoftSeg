import parse
import glob
import logging
import time
import ast
from datetime import datetime
from itertools import repeat
from multiprocessing import Pool

import numpy as np
import pandas as pd
import skimage
import skimage.io
from skimage.measure import regionprops


class SoftAssigner:
    def __init__(self, csv_loc, im_loc, complete_loc):
        """ Initialize internal parameters.

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

    def size_filter(im, min_size=None, max_size=None):
        """ Removes all items from mask that do not meet size requirements.

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
            logging.info(f"[{datetime.datetime.now()}] starting fov_{f:04}...")

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

                def fn(y): return skimage.morphology.binary_dilation(
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

            logging.info(
                f"[{datetime.datetime.now()}] saved fov_{f:04}\n\t{len(tr)} transcripts, now unpacking intensities..."
            )

            tr = tr.set_index("index")

            intensities = []
            for i, row in tr.iterrows():
                for k, v in row["cell_ids"].items():
                    intensities.append(v)

            print(f"Completed fov_{f:04}.")
            print(f"\t{len(intensities)} assigned to cells, {len(tr)} transcripts total.")
            print(f"\ttime taken:{(time.time() - t1)/60} minutes.")

            return (tr, intensities)

        else:
            print(f"Skipping fov_{f:04}, no transcripts found.")
            return None

    def blur_all_fovs(self, pool_size, min_size, dilate, sigma, thresh):
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
        with Pool(pool_size) as pool:
            pool.starmap(
                self.blur_fov,
                zip(
                    sel_fovs,
                    repeat(min_size),
                    repeat(dilate),
                    repeat(sigma),
                    repeat(thresh)
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

    def calculate_confident_threshold(self):
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
