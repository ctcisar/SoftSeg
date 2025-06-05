import matplotlib.pyplot as plt
import numpy as np


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
    plt.tick_params("x", length=10, labelsize="small")
    plt.ylabel("KL-divergence")
    if title is not None:
        plt.title(title)

    plt.legend(legend)
    plt.show(block=False)
