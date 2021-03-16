import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import pairwise_kernels
from functools import partial

N_JOBS = 2


def linear_kernel(XA, XB=None):
    """
    Builds a dot product kernel

    ---Arguments---
    XA, XB: vectors of data from which to build the kernel,
        where each row is a sample and each column is a feature.
        If XA or XB is a list, the kernel is averaged over the list

    ---Returns---
    K: dot product kernel between XA (and XB)
    """

    flag_A = isinstance(XA, list)
    XA_ = XA.copy()
    if flag_A:
        XA_ = np.array([np.mean(xa, axis=0) for xa in XA])

    flag_B = isinstance(XB, list)
    if XB is not None:
        XB_ = XB.copy()
        if flag_B:
            XB_ = np.array([np.mean(xb, axis=0) for xb in XB])
    else:
        XB_ = XA_.copy()

    return pairwise_kernels(
        XA_, XB_, metric="linear", filter_params=True, n_jobs=N_JOBS
    )


def gaussian_kernel(XA, XB=None, gamma=1.0):
    """
    Build a Gaussian kernel between all samples in XA
    """

    K = np.zeros((len(XA), len(XA)))
    my_kernel = partial(
        pairwise_kernels, metric="rbf", gamma=gamma, filter_params=True, n_jobs=N_JOBS
    )

    flag_A = isinstance(XA, list)
    flag_B = isinstance(XB, list)

    # XA and XB structures
    if flag_A and flag_B:
        for idx_i in range(len(XA)):
            for idx_j in range(len(XB)):
                K[idx_i, idx_j] = np.mean(my_kernel(XA[idx_i], XB[idx_j]))

    # XA structures, XB environments
    elif flag_A:
        for idx_i in range(len(XA)):
            K[idx_i] = np.mean(my_kernel(XA[idx_i], XB), axis=0)

    # XA environments, XB structures
    elif flag_B:
        for idx_j in range(len(XB)):
            K[:, idx_j] = np.mean(my_kernel(XA, XB[idx_j]), axis=1)

    # XA and XB environments
    else:
        K = my_kernel(XA, XB)

    return K
