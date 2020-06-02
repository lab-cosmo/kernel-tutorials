import numpy as np
from abc import abstractmethod
from sklearn.linear_model import LinearRegression as LR
from sklearn.utils import check_array
from sklearn.utils.extmath import svd_flip
from sklearn.utils.validation import check_is_fitted, check_X_y
from scipy.sparse.linalg import svds, eigs


class _BasePCovR():
    """
    Super-class defined for PCovR style methods
    #
    # ----Attributes----
    # PTY: projector from latent space to property space
    # Yhat: regressed properties
    # alpha: (float) mixing parameter between decomposition and regression
    #
    # ----Inherited Attributes----
    # PTX: projector from latent space to input space
    # PXT: projector from input space to latent space
    # n_PC (int) number of principal components to store
    # PXY: projector from input space to property space
    # center: (boolean) whether to shift all inputs to zero mean
    # regularization: (float) parameter to offset all small eigenvalues for
    #                 regularization
    # scale: (boolean) whether to scale all inputs to unit variance
    #
    # ----Inherited Methods----
    # postprocess: un-centers and un-scales outputs according to scale and center
    #              attributes
    # preprocess: centers and scales provided inputs according to scale and center
    #             attributes
    # """

    def __init__(self, alpha, n_components, regularization, tol):
        # TODO
        self.alpha = alpha
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

    def _transform(self, X, projector):
        """Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.
        projector: projection matrix

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        Examples
        --------

        >>> todo
        """
        check_is_fitted(self)

        X = check_array(X)
        # if self.mean_ is not None: # TODO with scaling
        #     X = X - self.mean_
        X_transformed = np.dot(X, projector)
        # if self.whiten:
        #     X_transformed /= np.sqrt(self.explained_variance_)
        return X_transformed

    def _inverse_transform(self, T, projector):
        """Transform data back to its original space.

        In other words, return an input X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of components.
        projector: projecton matrix

        Returns
        -------
        X_original array-like, shape (n_samples, n_features)

        Notes
        -----
        TODO
        If whitening is enabled, inverse_transform will compute the
        exact inverse operation, which includes reversing whitening.
        """
        # if self.whiten:
        #    return np.dot(X, np.sqrt(self.explained_variance_[:, np.newaxis]) *
        #                    self.components_) + self.mean_
        # else:

        return np.dot(T, projector) #+ self.mean_

    def _predict(self, X, projector):
        T = self.transform(X)
        return np.dot(T, projector)


class PCovR(_BasePCovR):
    """
    Performs PCovR, detecting whether the data set is in Sample or Feature Space

    ----Attributes----
    space: whether to compute in feature or sample space

    ----Inherited Attributes----
    PTX: projector from latent space to input space
    PTY: projector from latent space to property space
    PXT: projector from input space to latent space
    PXY: projector from input space to property space
    Yhat: regressed properties
    alpha: (float) mixing parameter between decomposition and regression
    center: (boolean) whether to shift all inputs to zero mean
    n_PC (int) number of principal components to store
    regularization: (float) parameter to offset all small eigenvalues for
                    regularization
    scale: (boolean) whether to scale all inputs to unit variance

    ----Inherited Methods----
    postprocess: un-centers and un-scales outputs according to scale and
                 center attributes
    preprocess: centers and scales provided inputs according to scale and
                center attributes


    ----References----
        1.  S. de Jong, H. A. L. Kiers, 'Principal Covariates
            Regression: Part I. Theory', Chemometrics and Intelligent
            Laboratory Systems 14(1): 155-164, 1992
        2.  M. Vervolet, H. A. L. Kiers, W. Noortgate, E. Ceulemans,
            'PCovR: An R Package for Principal Covariates Regression',
            Journal of Statistical Software 65(1):1-14, 2015
    """

    def __init__(self, alpha=0.0, n_components=None,
                 space=None, lr_args={}, *args, **kwargs,
                 ):
        super().__init__(alpha=alpha, n_components=n_components, *args, **kwargs)
        self.space = space
        self.lr_args = lr_args
        self.eig_solver = 'sparse'
        self.Yhat = None
        self.W = None
        self.n_components = n_components

    def fit(self, X, Y, Yhat=None):
        # as required by the superclass

        X, Y = check_X_y(X, Y, y_numeric=True, multi_output=True)

        if Yhat is None:
            self._compute_Yhat(X, Y)
        else:
            self.Yhat = Yhat

        if min(X.shape) <= self.n_components:
            self.eig_solver = 'full'

        self._fit(X, Y)

    def _fit(self, X, Y):
        #
        # # Handle n_components==None
        # if self.n_components is None:
        #     self.n_components = min(X.shape)

        if self.space is None:
            if X.shape[0] > X.shape[1]:
                self.space = 'feature'
            else:
                self.space = 'structure'
        print(self.space)

        if self.space == 'feature':
            self._fit_feature_space(X, Y)
        else:
            self._fit_sample_space(X, Y)

    def _compute_Yhat(self, X, Y):

        if self.Yhat is None:
            lr = LR(self.lr_args)  # some sort of args
            lr.fit(X, Y)
            self.Yhat = lr.predict(X)
            self.W = lr.coef_.T

        if self.W is None:
            W = np.linalg.pinv(np.dot(X.T, X), rcond=self.regularization)
            W = np.linalg.multi_dot([W, X.T, Y])
            self.W = W

    def _fit_feature_space(self, X, Y):

        C = np.dot(X.T, X)

        # it is necessary to use the full SVD to decompose C
        U, S, V = np.linalg.svd(C, full_matrices=False)
        U, V = svd_flip(U[:, ::-1], V[::-1])

        U = U[:, S > self.tol]
        S = S[S > self.tol]

        Csqrt = np.linalg.multi_dot([U, np.diagflat(S), U.T])
        iCsqrt = np.linalg.multi_dot([U, np.diagflat(1.0 / S), U.T])

        C_lr = iCsqrt @ X.T @ self.Yhat
        C_lr = C_lr @ C_lr.T

        # note about Ctilde definition
        Ct = self.alpha * C + (1.0 - self.alpha) * C_lr

        v, U = self._eig_solver(Ct)
        S = v ** 0.5

        self.pxt_ = np.linalg.multi_dot([iCsqrt, U, np.diagflat(S)])
        self.ptx_ = np.linalg.multi_dot([np.diagflat(1.0/S), U.T, Csqrt])
        self.pty_ = np.linalg.multi_dot(
            [np.diagflat(1.0/S), U.T, iCsqrt, X.T, Y])

    def _fit_sample_space(self, X, Y):

        # note about Ktilde definition
        Kt = (self.alpha * np.dot(X, X.T)) + \
             (1.0 - self.alpha) * np.dot(self.Yhat, self.Yhat.T)

        v, U = self._eig_solver(Kt)
        S = v ** 0.5

        T = np.dot(U, np.diagflat(S))

        P = (self.alpha * X.T) + (1.0 - self.alpha) * \
            np.dot(self.W, self.Yhat.T)
        self.pxt_ = np.linalg.multi_dot([P, U, np.diagflat(1/S)])
        self.pty_ = np.linalg.multi_dot([np.diagflat(1/S**2.0), T.T, Y])
        self.ptx_ = np.linalg.multi_dot([np.diagflat(1/S**2.0), T.T, X])

    def _eig_solver(self,matrix):
        if(self.eig_solver=='sparse'):
            v, U= eigs(matrix, k=self.n_components, tol=self.tol)
        else:
            v, U = np.linalg.eig(matrix)

        U = np.real(U[:, np.argsort(v)])
        v = np.real(v[np.argsort(v)])

        U = U[:, v > self.tol]
        v = v[v > self.tol]

        if(len(v)==1):
            U = U.reshape(-1,1)

        return v, U

    def transform(self, X):
        return super()._transform(X, self.pxt_)

    def inverse_transform(self, T):

        # TODO: check that T is of correct shape
        return super()._inverse_transform(T, self.ptx_)

    def predict(self, X):

        # Predict based on X only
        return super()._predict(X, self.pty_)
