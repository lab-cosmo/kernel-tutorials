import numpy as np
from scipy.spatial.distance import cdist


def linear_kernel(XA, XB):
    """
        Builds a dot product kernel

        ---Arguments---
        XA, XB: vectors of data from which to build the kernel,
            where each row is a sample and each column is a feature

        ---Returns---
        K: dot product kernel between XA and XB
    """

    K = np.matmul(XA, XB.T)

    return K


def summed_kernel(XA, XB, kernel_func, **kwargs):
    """
        Provides the kernel for for properties which are learned on the sum of
        soap vectors. Necessary for non-linear kernels.
    """
    K = np.zeros((XA.shape[0], XB.shape[0]))

    for idx_i in range(XA.shape[0]):
        for idx_j in range(XB.shape[0]):
            kij = kernel_func(XA[idx_i], XB[idx_j], **kwargs)
            K[idx_i, idx_j] = kij.sum()

    return K


def gaussian_kernel(XA, XB, gamma=1.0):
    """
        Builds a gaussian kernel of the type k(x, x') = np.exp(-gamma*(x-x'))

        ---Arguments---
        XA, XB: vectors of data from which to build the kernel,
            where each row is a sample and each column is a feature

        ---Returns---
        K: gaussian kernel between XA and XB
    """
    D = cdist(XA, XB, metric='sqeuclidean')
    return np.exp(-D*gamma)


def center_kernel(K, reference=None):
    """
        Centers a kernel

        ---Arguments---
        K: kernel to center
        reference: kernel relative to whos RKHS one should center
                   defaults to K

        ---Returns---
        Kc: centered kernel

        ---References---
        1.  https://en.wikipedia.org/wiki/Kernel_principal_component_analysis
        2.  M. Welling, 'Kernel Principal Component Analysis',
            https://www.ics.uci.edu/~welling/classnotes/papers_class/Kernel-PCA.pdf
    """

    K_ref = reference
    if K_ref is None:
        K_ref = K
    else:
        if K.shape[1] != K_ref.shape[0]:
            raise ValueError(
                "The kernel to be centered and the reference have inconsistent sizes")
    if K_ref.shape[0] != K_ref.shape[1]:
        raise ValueError(
            "The reference kernel is not square, and does not define a RKHS")
    oneMN = np.ones((K.shape[0], K.shape[1]))/K.shape[1]
    oneNN = np.ones((K.shape[1], K.shape[1]))/K.shape[1]
    Kc = K - np.matmul(oneMN, K_ref) - np.matmul(K, oneNN) + \
        np.matmul(np.matmul(oneMN, K_ref), oneNN)

    return Kc