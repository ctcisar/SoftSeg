import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.io


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

    def load_fov(self, fov):
        self.masks = skimage.io.imread(self.mask_str.format(fov))
        self.image = skimage.exposure.equalize_hist(
            skimage.io.imread(self.seg_str.format(fov))
        )

    def mask_from_img(self, ind):
        return (self.masks == ind + 1).astype(np.uint8)

    def cross_to_slice(cross):
        return (
            slice(min(cross["zs"]), max(cross["zs"])),
            slice(cross["x0"], cross["x1"]),
            slice(cross["y0"], cross["y1"]),
        )

    def plot_subset(self, target_cell=None, highlight_cells=None, adata=None, highlight_type=None):
        """
        target_cell: if provided, will zoom in on that cell specifically
        highlight_cells: cell ids listed will be highlighted in red
        adata: anndata with cell ids and other information. Only needed for highlight_types
        highlight_type: cells that match this type will be highlighted in white
        """
        if self.masks is None:
            raise Exception("load_fov must be called before performing this operation.")
        if target_cell is not None:
            mask = self.mask_from_img(target_cell)
            cross = SegImagePlotter.get_range_dict(mask, border=100)
            crossection = SegImagePlotter.cross_to_slice(cross)
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

                if (
                    highlight_cells is not None
                    and c - 1 in highlight_cells
                ):
                    color = (1, 0, 0)
                elif (
                    adata is not None and highlight_type is not None
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
            plt.imshow(this_img)
            plt.show()
