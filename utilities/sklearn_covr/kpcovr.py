import numpy as np
from sklearn.kernel_ridge import KernelRidge as KRR
from sklearn.utils.validation import check_X_y
from scipy.sparse.linalg import eigs
from numpy.linalg import multi_dot as mdot

from ._base import _BasePCovR
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import pairwise_kernels

class KernelPCovR(_BasePCovR):
    """
    Performs KernelPCovR

    Parameters
    ----------
    mixing: float
        Mixing parameter for KPCovR model. Coincides with `alpha` parameter in
        PCovR/KPCovR literature.

    n_components : int, default=None
        Number of components. If None, all non-zero components are kept.

    kernel : "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
        Kernel. Default="linear".

    gamma : float, default=1/n_features
        Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other
        kernels.

    degree : int, default=3
        Degree for poly kernels. Ignored by other kernels.

    coef0 : float, default=1
        Independent term in poly and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : mapping of string to any, default=None
        Parameters (keyword arguments) and values for kernel passed as
        callable object. Ignored by other kernels.

    alpha : int, default=1.0
        Hyperparameter of the ridge regression that learns the
        inverse transform (when fit_inverse_transform=True).

    fit_inverse_transform : bool, default=False
        Learn the inverse transform for non-precomputed kernels.
        (i.e. learn to find the pre-image of a point)

    eigen_solver : string ['auto'|'dense'|'arpack'], default='auto'
        Select eigensolver to use. If n_components is much less than
        the number of training samples, arpack may be more efficient
        than the dense eigensolver.

    tol : float, default=0
        Convergence tolerance for arpack.
        If 0, optimal value will be chosen by arpack.

    max_iter : int, default=None
        Maximum number of iterations for arpack.
        If None, optimal value will be chosen by arpack.

    remove_zero_eig : boolean, default=False
        If True, then all components with zero eigenvalues are removed, so
        that the number of components in the output may be < n_components
        (and sometimes even zero due to numerical instability).
        When n_components is None, this parameter is ignored and components
        with zero eigenvalues are removed regardless.

    random_state : int, RandomState instance, default=None
        Used when ``eigen_solver`` == 'arpack'. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

        .. versionadded:: 0.18

    copy_X : boolean, default=True
        If True, input X is copied and stored by the model in the `X_fit_`
        attribute. If no further changes will be done to X, setting
        `copy_X=False` saves memory by storing a reference.

        .. versionadded:: 0.18

    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionadded:: 0.18

    Attributes
    ----------
    lambdas_ : array, (n_components,)
        Eigenvalues of the centered kernel matrix in decreasing order.
        If `n_components` and `remove_zero_eig` are not set,
        then all values are stored.

    alphas_ : array, (n_samples, n_components)
        Eigenvectors of the centered kernel matrix. If `n_components` and
        `remove_zero_eig` are not set, then all components are stored.

    dual_coef_ : array, (n_samples, n_features)
        Inverse transform matrix. Only available when
        ``fit_inverse_transform`` is True.

    X_transformed_fit_ : array, (n_samples, n_components)
        Projection of the fitted data on the kernel principal components.
        Only available when ``fit_inverse_transform`` is True.

    X_fit_ : (n_samples, n_features)
        The data used to fit the model. If `copy_X=False`, then `X_fit_` is
        a reference. This attribute is used for the calls to transform.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.decomposition import KernelPCA
    >>> X, _ = load_digits(return_X_y=True)
    >>> transformer = KernelPCA(n_components=7, kernel='linear')
    >>> X_transformed = transformer.fit_transform(X)
    >>> X_transformed.shape
    (1797, 7)

    References
    ----------
    Kernel PCA was introduced in:
        Bernhard Schoelkopf, Alexander J. Smola,
        and Klaus-Robert Mueller. 1999. Kernel principal
        component analysis. In Advances in kernel methods,
        MIT Press, Cambridge, MA, USA 327-352.
    """

    def __init__(self, mixing,
                 n_components=None, kernel="linear",
                 gamma=None, degree=3, coef0=1, kernel_params=None,
                 alpha=1.0, fit_inverse_transform=False, eigen_solver='auto',
                 tol=0, max_iter=None, remove_zero_eig=False,
                 random_state=None, copy_X=True, n_jobs=None,
                 krr_params={}
                ):
        if fit_inverse_transform and kernel == 'precomputed':
            raise ValueError(
                "Cannot fit_inverse_transform with a precomputed kernel.")

        super().__init__(mixing=mixing, n_components=n_components,
                tol=tol)

        self.mixing = mixing
        self.krr_params = krr_params

        self.n_components = n_components
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform
        self.eigen_solver = eigen_solver
        self.remove_zero_eig = remove_zero_eig
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.copy_X = copy_X

        self.Yhat = None
        self.W = None

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, n_jobs=self.n_jobs,
                                **params)


    def _fit_transform(self, K):
        """ Fit's using kernel K"""
        # center kernel
        K = self._centerer.fit_transform(K)

        Kt = self.mixing * K + (1 - self.mixing) * np.matmul(self.Yhat, self.Yhat.T)

        if self.n_components is None:
            n_components = Kt.shape[0]
        else:
            n_components = min(Kt.shape[0], self.n_components)


        v, U = self._eig_solver(Kt, full_matrix=False)

        P_krr = np.matmul(self.W, self.Yhat.T)

        P_kpca = np.eye(K.shape[0]) #/ (np.trace(K) / K.shape[0])

        P = (self.mixing * P_kpca) + (1.0 - self.mixing) * P_krr

        v_inv = np.linalg.pinv(np.diagflat(v))

        self.pkt_ = mdot([P, U, np.sqrt(v_inv)])
        T = self._project(K, self.pkt_)
        return T

    def _fit_inverse_transform(self, T, X):
        # todo add necessary checks

        self.ptx_ = mdot([np.linalg.pinv(np.matmul(T.T, T)), T.T, X])

    def _fit_predict(self, T, Y):
        # todo add necessary checks

        self.pty_ = mdot([np.linalg.pinv(np.matmul(T.T, T)), T.T, Y])
        self.pky_ = np.matmul(self.pkt_, self.pty_)

    def _compute_Yhat(self, K, Y):
        """
        Method for computing the approximation of Y
        """

        if self.Yhat is None:
            if self.W is not None:
                self.Yhat = np.dot(K, self.W)
            else:
                krr = KRR(kernel='precomputed', **self.krr_params)  # some sort of args
                krr.fit(K, Y)
                self.Yhat = krr.predict(K)
                self.W = krr.dual_coef_.T

        if self.W is None:
            self.W = np.linalg.lstsq(K, self.Yhat)[0]


    def fit(self, X, Y, Yhat=None, W=None):

        X, Y = check_X_y(X, Y, y_numeric=True, multi_output=True)

        self._centerer = KernelCenterer() # todo  place our centerer

        K = self._get_kernel(X)

        self.Yhat = Yhat
        self.W = W

        if self.Yhat is None or self.W is None:
            self._compute_Yhat(K, Y)

        T = self._fit_transform(K)

        if self.fit_inverse_transform:
            self._fit_inverse_transform(T, X)

        self._fit_predict(T, Y)

        self.X_fit_ = X

    # def fit_transform(self, X, Y, )

    def transform(self, X):
        """Transform X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        check_is_fitted(self)

        # Compute centered gram matrix between X and training data X_fit_
        K = self._centerer.transform(self._get_kernel(X, self.X_fit_))

        return self._project(K, self.pkt_)

    def inverse_transform(self, T):
        """Transform X back to original space.

        Parameters
        ----------
        T : array-like, shape (n_samples, n_components)

        Returns
        -------
        X : array-like, shape (n_samples, n_features)

        References
        ----------
        "Learning to Find Pre-Images", G BakIr et al, 2004.
        """
        if not self.fit_inverse_transform:
            raise NotFittedError("The fit_inverse_transform parameter was not"
                                 " set to True when instantiating and hence "
                                 "the inverse transform is not available.")

        return self._project(T, self.ptx_)



    def predict(self, X):
        """Transform X into the regression Y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        Y : array-like, shape (n_samples, n_properties)
        """
        check_is_fitted(self)

        # Compute centered gram matrix between X and training data X_fit_
        K = self._centerer.transform(self._get_kernel(X, self.X_fit_))


        return self._project(K, self.pky_)
