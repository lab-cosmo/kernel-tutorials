import numpy as np
from abc import abstractmethod, ABCMeta
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from scipy.sparse.linalg import eigs


class _BasePCovR(TransformerMixin, RegressorMixin, BaseEstimator, metaclass=ABCMeta):
    """
    Super-class defined for PCovR style methods
    # """

    def __init__(self, mixing, n_components=None, regularization=1E-12, tol=0):
        # TODO
        self.mixing = mixing
        self.n_components = n_components
        self.regularization = regularization
        self.tol = tol

    @abstractmethod
    def fit(self, X, Y, Yhat=None):
        """Placeholder for fit. Subclasses should implement this method!

        Fit the model with X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        Y : array-like, shape (n_samples, n_properties)
            Training data, where n_samples is the number of samples and
            n_properties is the number of properties
        Yhat : array-like, shape (n_samples, n_properties), optional
            Regressed training data, where n_samples is the number of samples and
            n_properties is the number of properties. If not supplied, computed
            by ridge regression.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

    @abstractmethod
    def transform(self, X):
        """Placeholder for transform. Subclasses should implement this method!

        Transforms the model with X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

    @abstractmethod
    def predict(self, X):
        """Placeholder for transform. Subclasses should implement this method!

        Predicts the outputs given X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

    def _project(self, A, projector):
        """Apply a projector to matrix A


        Parameters
        ----------
        A : array-like, shape (n, a)
        projector: string corresponding to the named projection matrix of shape (a, p)

        Returns
        -------
        A' : array-like, shape (n, p)

        Examples
        --------

        >>> todo
        """
        check_is_fitted(self, projector)

        A = check_array(A)
        A_transformed = np.dot(A, self.__dict__[projector])
        return A_transformed

    def _eig_solver(self, matrix, full_matrix=False):
        if(full_matrix == False):
            v, U = eigs(matrix, k=self.n_components, tol=self.tol)
        else:
            v, U = np.linalg.eig(matrix)

        U = np.real(U[:, np.argsort(-v)])
        v = np.real(v[np.argsort(-v)])

        U = U[:, v > self.tol]
        v = v[v > self.tol]

        if(len(v) == 1):
            U = U.reshape(-1, 1)

        return v, U
