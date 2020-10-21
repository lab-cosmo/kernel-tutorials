from ._base import _BaseSelection
import numpy as np
from abc import abstractmethod
from scipy.sparse.linalg import eigs as speig
import scipy


class _CUR(_BaseSelection):
    """
    Super-class defined for CUR selection methods

    Parameters
    ----------
    alpha : float
        mixing parameter, as described in PCovR
        stored in `self.alpha`
    iterative: boolean
        whether to orthogonalize the matrices after each selection
    k : int
        number of eigenvectors to compute the importance score with
    matrix : ndarray of shape (n x m)
        Data to select from -
        Feature selection will choose a subset of the `m` columns
        Samples selection will choose a subset of the `n` rows
        stored in `self.A`
    precompute : int
        Number of selections to precompute
    progress_bar : bool, callable
        Option to include tqdm progress bar or a callable progress bar
        implemented in `self.progress(iterable)`
    tolerance : float
        Threshold below which values will be considered 0
        stored in `self.tol`
    Y (optional) : ndarray of shape (n x p)
        Array to include in biased selection when alpha < 1
        Required when alpha < 1, throws AssertionError otherwise
        stored in `self.Y`

    Attributes
    ----------
    A : ndarray of shape (n x m)
        corresponds to `matrix` passed in constructor
    A_current : ndarray of shape (n x m)
        copy of `A` which has been orthogonalized by previous selections
    alpha : float
    idx : list
        contains the indices of the feature or sample selections made
    product : ndarray
        shape is (m x m) (feature selection) or (n x n) (sample selection)
        inner or outer product of A and Y, as defined by PCov-CUR
    tol : float
        corresponds to `tolerance` passed in constructor
    Y (optional) : ndarray of shape (n x p)
    Y_current : ndarray of shape (n x p)
        copy of `Y` which has been orthogonalized by previous selections
        only initialized when Y is specified

    # """

    def __init__(self, matrix, alpha=1.0, iterative=True, tolerance=1E-12, k=1, **kwargs):
        super().__init__(matrix=matrix, alpha=alpha, tolerance=tolerance, **kwargs)

        self.k = k

        self.iter = iterative
        if(self.iter):
            self.A_current = self.A.copy()
            if(alpha < 1.0):
                self.Y_current = self.Y.copy()

        self.product = self.get_product()

    def select(self, n):
        """Method for CUR select based upon a product of the input matrices

        Parameters
        ----------
        n : number of selections to make

        Returns
        -------
        idx: list of n selections
        """
        if(len(self.idx) > n):
            return self.idx[:n]
        else:
            for nn in self.progress(range(len(self.idx), n)):
                if(self.iter):
                    v, U = speig(self.product, k=self.k)
                    U = U[:, np.flip(np.argsort(v))]
                    pi = (np.real(U)[:, :self.k]**2.0).sum(axis=1)
                pi[self.idx] = 0.0
                self.idx.append(pi.argmax())
                self.orthogonalize()

            return self.idx

    @abstractmethod
    def orthogonalize(self):
        """Abstract method for orthogonalizing, implemented by subclasses
           Edits A_current, Y_current (if exists), and product in place
        """
        return

    @abstractmethod
    def get_product(self):
        """Abstract method for computing the product (C or K) of the input matrices
        """
        return


class sampleCUR(_CUR):
    def __init__(self, matrix, alpha=1.0, iterative=True, tolerance=1E-12, **kwargs):
        super().__init__(matrix=matrix, alpha=alpha,
                         tolerance=tolerance, iterative=iterative, **kwargs)

    def get_product(self):
        """
            Creates the PCovR modified kernel distances
            ~K = (alpha) * X X^T +
                 (1-alpha) * Y Y^T

        """

        K = np.zeros((self.A_current.shape[0], self.A_current.shape[0]))
        if(self.alpha < 1):
            K += (1 - self.alpha) * self.Y_current @ self.Y_current.T
        if(self.alpha > 0):
            K += (self.alpha) * self.A_current @ self.A_current.T

        return K

    def orthogonalize(self):
        if(not self.iter):
            return

        if(self.alpha < 1):
            self.Y_current -= self.A_current @ (np.linalg.pinv(
                self.A_current[self.idx].T @ self.A_current[self.idx], rcond=self.tol) @ self.A_current[self.idx].T) @ self.Y_current[self.idx]

        Ajnorm = np.dot(self.A_current[self.idx[-1]],
                        self.A_current[self.idx[-1]])
        for i in range(self.A.shape[0]):
            self.A_current[i] -= (np.dot(self.A_current[i], self.A_current[self.idx[-1]]
                                         ) / Ajnorm) * self.A_current[self.idx[-1]]

        self.product = self.get_product()


class featureCUR(_CUR):
    def __init__(self, matrix, alpha=1.0, iterative=True, tolerance=1E-12, **kwargs):

        super().__init__(matrix=matrix, alpha=alpha,
                         tolerance=tolerance, iterative=iterative, **kwargs)

    def get_product(self):
        """
            Creates the PCovR modified covariance
            ~C = (alpha) * X^T X +
                 (1-alpha) * (X^T X)^(-1/2) ~Y ~Y^T (X^T X)^(-1/2)

            where ~Y is the properties obtained by linear regression.
        """

        C = np.zeros(
            (self.A_current.shape[1], self.A_current.shape[1]), dtype=np.float64)

        cov = self.A_current.T @ self.A_current

        if(self.alpha < 1):
            # changing these next two lines can cause a LARGE error
            Cinv = np.linalg.pinv(cov)
            Cisqrt = scipy.linalg.sqrtm(Cinv)

            # parentheses speed up calculation greatly
            Y_hat = Cisqrt @ (self.A_current.T @ self.Y_current)
            Y_hat = Y_hat.reshape((C.shape[0], -1))
            Y_hat = np.real(Y_hat)

            C += (1 - self.alpha) * Y_hat @ Y_hat.T

        if(self.alpha > 0):
            C += (self.alpha) * cov

        return C

    def orthogonalize(self):
        if(not self.iter):
            return

        Aci = self.A_current[:, self.idx]

        if(self.alpha < 1):
            v = np.linalg.pinv(np.matmul(Aci.T, Aci), rcond=self.tol)
            v = np.matmul(Aci, v)
            v = np.matmul(v, Aci.T)

            self.Y_current -= np.matmul(v, self.Y_current)

        v = self.A_current[:, self.idx[-1]] / np.sqrt(
            np.matmul(self.A_current[:, self.idx[-1]], self.A_current[:, self.idx[-1]]))
        self.v = v

        for i in range(self.A_current.shape[1]):
            self.A_current[:, i] -= v * np.dot(v, self.A_current[:, i])

        self.product = self.get_product()