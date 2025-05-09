import numpy as np


class SupportFuncs:
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
