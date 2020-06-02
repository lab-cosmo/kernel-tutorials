import numpy as np
from abc import ABCMeta, abstractmethod
from .general import FPS, get_stats, sorted_eig, eig_inv, center_matrix, normalize_matrix
from sklearn.linear_model import LinearRegression as LR
# class Model:
#     """
#     Super-class defined for all models
#
#     ----Attributes----
#     center: (boolean) whether to shift all inputs to zero mean
#     regularization: (float) parameter to offset all small eigenvalues
#                     for regularization
#     scale: (boolean) whether to scale all inputs to unit variance
#
#     ----Methods----
#     postprocess: un-centers and un-scales outputs according to scale and center
#                  attributes
#     preprocess: centers and scales provided inputs according to scale and center
#                 attributes
#     """
#
#     def __init__(self, regularization=1e-12, scale=False, center=False, *args, **kwargs):
#         self.regularization = regularization
#
#         self.center = center
#         self.scale = scale
#         self.X_center = None
#         self.X_scale = None
#
#         # self.X_center = 0
#         # self.Y_center = 0
#         # self.X_scale = 1
#         # self.Y_scale = 1
#         # self.K_ref = None
#
#     def preprocess(self, X=None, Y=None, K=None,
#                    X_ref=None, Y_ref=None, K_ref=None, rcond=1.0E-12,
#                    *args, **kwargs):
#         """
#         Scale and center the input data as designated by the model parameters
#         `scale` and `center`. These parameters are set on the model level to
#         enforce the same centering and scaling for all data supplied to the
#         model.  i.e. if a model is trained on centered input X, it must be
#         supplied a similarly centered input X' for transformation.
#         """
#
#         if(X_ref is None and X is not None):
#             X_ref = X.copy()
#
#         if(Y_ref is None and Y is not None):
#             Y_ref = Y.copy()
#
#         if self.center:
#             if X_ref is not None and self.X_center is None:
#                 self.X_center = X_ref.mean(axis=0)
#
#             if isinstance(self, Regression):
#                 if Y_ref is not None and self.Y_center is None:
#                     self.Y_center = Y_ref.mean(axis=0)
#
#             if isinstance(self, Kernelized):
#                 if K_ref is None and K is not None:
#                     K_ref = K
#
#                 if K_ref is not None and self.K_ref is None:
#                     self.K_ref = K_ref
#
#         if self.scale:
#             if X_ref is not None and self.X_scale is None:
#                 self.X_scale = np.linalg.norm(X_ref - self.X_center) / np.sqrt(X_ref.shape[0])
#
#             if isinstance(self, Regression):
#                 if Y_ref is not None and self.Y_scale is None:
#                     self.Y_scale = np.linalg.norm(Y_ref - self.Y_center, axis=0) / np.sqrt(Y_ref.shape[0] / Y_ref.shape[1])
#
#         if X is not None:
#             Xcopy = X.copy()
#             if self.center:
#                 Xcopy = center_matrix(Xcopy, self.X_center)
#             if self.scale:
#                 Xcopy = normalize_matrix(Xcopy, scale=self.X_scale)
#         else:
#             Xcopy = None
#
#         if Y is not None:
#             Ycopy = Y.copy()
#             if self.center:
#                 Ycopy = center_matrix(Ycopy, self.Y_center)
#             if self.scale:
#                 Ycopy = normalize_matrix(Ycopy, scale=self.Y_scale)
#         else:
#             Ycopy = None
#
#         if K is not None:
#             Kcopy = K.copy()
#             if self.center and isinstance(self, Sparsified):
#                 K_center = np.mean(self.K_ref, axis=0)
#                 Kcopy = center_matrix(Kcopy, K_center)
#             elif self.center:
#                 Kcopy = center_kernel(Kcopy, reference=self.K_ref)
#             if self.scale and isinstance(self, Sparsified):
#                 try:
#                     K_ref_centered = self.K_ref - np.mean(self.K_ref, axis=0)
#                     self.K_scale = K_ref_centered @ np.linalg.pinv(self.Kmm, rcond=rcond) @ K_ref_centered.T
#                     self.K_scale = np.sqrt(np.trace(self.K_scale) / self.K_ref.shape[0])
#                     Kcopy = normalize_matrix(Kcopy, scale=self.K_scale)
#                 except AttributeError:
#                     print("Error: Kmm is required for the scaling but it has not been set")
#             elif self.scale:
#                 self.K_scale = np.trace(center_kernel(self.K_ref)) / self.K_ref.shape[0]
#                 Kcopy = normalize_matrix(Kcopy, scale=self.K_scale)
#         else:
#             Kcopy = None
#
#         return Xcopy, Ycopy, Kcopy
#
#     def postprocess(self, X=None, Y=None, *args, **kwargs):
#         """
#         Undoes any scaling and center on the output data for comparison
#         """
#
#         if X is not None:
#             Xcopy = X.copy()
#             if self.scale:
#                 Xcopy = normalize_matrix(Xcopy, scale=(1.0 / self.X_scale))
#             if self.center:
#                 Xcopy = center_matrix(Xcopy, -self.X_center)
#         else:
#             Xcopy = None
#
#         if Y is not None:
#             Ycopy = Y.copy()
#             if self.scale:
#                 Ycopy = normalize_matrix(Ycopy, scale=(1.0 / self.Y_scale))
#             if self.center:
#                 Ycopy = center_matrix(Ycopy, -self.Y_center)
#         else:
#             Ycopy = None
#
#         return Xcopy, Ycopy
#
#
# class Decomposition(Model):
#     """
#     Super-class defined for any decompositions ala PCA
#
#     ----Attributes----
#     PTX: projector from latent space to input space
#     PXT: projector from input space to latent space
#     n_PC (int) number of principal components to store
#
#     ----Inherited Attributes----
#     center: (boolean) whether to shift all inputs to zero mean
#     regularization: (float) parameter to offset all small eigenvalues for
#                     regularization
#     scale: (boolean) whether to scale all inputs to unit variance
#
#     ----Inherited Methods----
#     postprocess: un-centers and un-scales outputs according to scale and center
#                  attributes
#     preprocess: centers and scales provided inputs according to scale and center
#                 attributes
#     """
#
#     def __init__(self, n_PC=2, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.n_PC = n_PC
#         self.PXT = None
#         self.PTX = None
#
#
# class Regression(Model):
#     """
#     Super-class defined for any regressions
#
#     ----Attributes----
#     PXY: projector from input space to property space
#
#     ----Inherited Attributes----
#     center: (boolean) whether to shift all inputs to zero mean
#     regularization: (float) parameter to offset all small eigenvalues for
#                     regularization
#     scale: (boolean) whether to scale all inputs to unit variance
#
#     ----Inherited Methods----
#     postprocess: un-centers and un-scales outputs according to scale and center
#                  attributes
#     preprocess: centers and scales provided inputs according to scale and center
#                 attributes
#
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.PXY = None
#         self.Y_center = None
#         self.Y_scale = None
#
#
# class Kernelized(Model):
#     """
#     Super-class defined for any kernelized methods
#
#     ----Attributes----
#     PKT: projector from kernel space to latent space
#     PKY: projector from kernel space to property space
#     PTK: projector from latent space to kernel space
#     X: input used to train the model, if further kernels need be constructed
#     kernel: function to construct the kernel of the input data
#
#     ----Inherited Attributes----
#     center: (boolean) whether to shift all inputs to zero mean
#     regularization: (float) parameter to offset all small eigenvalues for
#                     regularization
#     scale: (boolean) whether to scale all inputs to unit variance
#
#     ----Inherited Methods----
#     postprocess: un-centers and un-scales outputs according to scale and center
#                  attributes
#     preprocess: centers and scales provided inputs according to scale and center
#                 attributes
#
#     """
#
#     def __init__(self, kernel_type='linear', *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         self.kernel = None
#         if isinstance(kernel_type, str):
#             if kernel_type in kernels:
#                 self.kernel = kernels[kernel_type]
#         elif callable(kernel_type):
#             self.kernel = kernel_type
#
#         if self.kernel is None:
#             raise Exception(
#                 'Kernel Error: Please specify either {} or pass a suitable \
#                 kernel function.'.format(kernels.keys())
#             )
#
#         self.PKT = None
#         self.PTK = None
#         self.PKY = None
#         self.X = None
#         self.K_ref = None
#         self.K_scale = None
#
#
# class Sparsified(Kernelized):
#     """
#     Super-class defined for any kernelized methods
#
#     ----Attributes----
#     n_active: (int) size of the active set
#     X_sparse: active set used to train the model, if further kernels need
#               be constructed
#
#     ----Inherited Attributes----
#     PKT: projector from kernel space to latent space
#     PTK: projector from latent space to kernel space
#     PTX: projector from latent space to input space
#     X: input used to train the model, if further kernels need be constructed
#     center: (boolean) whether to shift all inputs to zero mean
#     kernel: function to construct the kernel of the input data
#     regularization: (float) parameter to offset all small eigenvalues for
#                     regularization
#     scale: (boolean) whether to scale all inputs to unit variance
#
#     ----Inherited Methods----
#     postprocess: un-centers and un-scales outputs according to scale and center
#                  attributes
#     preprocess: centers and scales provided inputs according to scale and center
#                 attributes
#     """
#
#     def __init__(self, n_active, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         self.n_active = n_active
#         self.X_sparse = None


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

    def __init__(self, alpha, n_PC, regularization):
        super(PCovRBase, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.PTY = None
        self.Yhat = None

    @abstractmethod
    def fit(X, Y, Yhat=None):
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
        #if self.whiten:
        #    return np.dot(X, np.sqrt(self.explained_variance_[:, np.newaxis]) *
        #                    self.components_) + self.mean_
        #else:

        return np.dot(T, projector) + self.mean_

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

    def __init__(self, alpha=0.0, n_PC=None,
                    space=None, *args, **kwargs,
                    lr_args=dict(),
                    ):
        super().__init__(alpha=alpha, n_PC=n_PC, *args, **kwargs)
        self.space = space
        self.lr_args = lr_args
        self.svd_solver = svds
        self.Yhat = None
        self.W = None

    def fit(X, Y, Yhat=None):
        # as required by the superclass

        X, Y = check_X_y(X, Y, y_numeric=True, multi_output=True)

        if Yhat is None:
            self._compute_Yhat(X, Y)
        else:
            self.Yhat = Yhat

        self._fit(X, Y)

    def _fit(X, Y):


        # Handle n_components==None
        if self.n_components is None:
            n_components = min(X.shape)
        else:
            n_components = self.n_components

        if self.space is None:
            if X.shape[0] > X.shape[1]:
                self.space = 'feature'
            else:
                self.space = 'structure'

        if self.space == 'feature':
            self._fit_feature_space(X, Y)
        else:
            self._fit_sample_space(X, Y)

    def _compute_Yhat(X, Y):

        if self.Yhat is None :
            lr = LR(self.lr_args) #some sort of args
            lr.fit(X, Y)
            self.Yhat = lr.predict(X)
            self.W = lr.coef_

        if self.W is None:
            W = np.linalg.pinv(np.dot(X.T, X), rcond=self.regularization)
            W = np.linalg.multi_dot([W, X.T, Y])
            self.W = W

    def _fit_feature_space(self, X, Y):

        C = np.dot(X.T, X)

        # it is necessary to use the full SVD to decompose C
        U, S, V = linalg.svd(C, full_matrices=False)
        U, V =  svd_flip(U[:, ::-1], V[::-1])

        U = U[:, S > self.tol]
        S = S[S > self.tol]

        Csqrt = np.linalg.multi_dot(U, np.diagflat(S), U.T)
        iCsqrt = np.linalg.multi_dot(U, np.diagflat(1.0 / S), U.T)

        C_lr = iCsqrt @ X.T @ self.Yhat
        C_lr = C_lr @ C_lr.T

        # note about Ctilde definition
        Ct =  self.alpha * C + (1.0 - self.alpha) * C_lr

        U, S, V = self.svd_solver(Ct, k=self.n_components, tol=self.tol)
        U, V =  svd_flip(U[:, ::-1], V[::-1])

        self.pxt_ = np.linalg.multi_dot([iCsqrt, U, np.diagflat(S)])
        self.ptx_ = np.linalg.multi_dot([np.diagflat(1.0/S), U.T, Csqrt])
        self.pty_ = np.linalg.multi_dot([np.diagflat(1.0/S), U.T, iCsqrt, X.T, Y])

    def _fit_structure_space(self, X, Y):

        # note about Ktilde definition
        Kt = (self.alpha * np.dot(X, X.T)) + \
             (1.0 - self.alpha) * np.dot(self.Yhat, self.Yhat.T)

        U, S, V = self.svd_solver(Kt, k=self.n_components, tol=self.tol)
        U, V =  svd_flip(U[:, ::-1], V[::-1])

        T = np.dot(U, np.diagflat(S))

        P = (self.alpha * X.T) + (1.0 - self.alpha) * np.dot(self.W, self.Yhat.T)
        self.pxt_ = np.linalg.multi_dot([P, U, np.diagflat(1/S)])
        self.pty_ = np.linalg.multi_dot([np.diagflat(1/S), T.T, Y])
        self.pty_ = np.linalg.multi_dot([np.diagflat(1/S), T.T, X])

    def transform(self, X):
        return super()._transform(X, self.pxt_)

    def inverse_transform(self, T): 

        # TODO: check that T is of correct shape
        return super()._inverse_transform(T, self.ptx_)

    def predict(self, X):

        # Predict based on X only
        T = self.transform(X)
        return np.dot(T, self.pty_)

    def loss(self, X, Y):
        T, Yp, Xr = self.transform(X)

        Lpca = np.linalg.norm(X - Xr) ** 2 / np.linalg.norm(X) ** 2
        Llr = np.linalg.norm(Y - Yp) ** 2 / np.linalg.norm(Y) ** 2

        return Lpca, Llr

    def statistics(self, X, Y):
        T, Yp, Xr = self.transform(X)

        return get_stats(x=X, y=Y, yp=Yp, t=T, xr=Xr)


class KPCovR(PCovRBase, Kernelized):
    """
    Performs KPCovR, as described in Helfrecht (2020), which combines Kernel
    Principal Components Analysis (KPCA) and Kernel Ridge Regression (KRR)

    ----Inherited Attributes----
    PKT: projector from kernel space to latent space
    PKY: projector from kernel space to property space
    PTK: projector from latent space to kernel space
    PTX: projector from latent space to input space
    PTY: projector from latent space to property space
    PXT: projector from input space to latent space
    PXY: projector from input space to property space
    X: input used to train the model, if further kernels need be constructed
    Yhat: regressed properties
    alpha: (float) mixing parameter between decomposition and regression
    center: (boolean) whether to shift all inputs to zero mean
    kernel: function to construct the kernel of the input data
    n_PC (int) number of principal components to store
    regularization: (float) parameter to offset all small eigenvalues for
                    regularization
    scale: (boolean) whether to scale all inputs to unit variance

    ----Inherited Methods----
    postprocess: un-centers and un-scales outputs according to scale and
                 center attributes
    preprocess: centers and scales provided inputs according to scale and
                center attributes

    ---References---
    1.  S. de Jong, H. A. L. Kiers, 'Principal Covariates
        Regression: Part I. Theory', Chemometrics and Intelligent
        Laboratory Systems 14(1): 155-164, 1992
    2.  M. Vervolet, H. A. L. Kiers, W. Noortgate, E. Ceulemans,
        'PCovR: An R Package for Principal Covariates Regression',
        Journal of Statistical Software 65(1):1-14, 2015
    """

    def __init__(self, n_PC=None, *args, **kwargs):
        super(KPCovR, self).__init__(n_PC=n_PC, *args, **kwargs)

    def fit(self, X, Y, K=None, Yhat=None):
        X, Y, K = self.preprocess(X=X, X_ref=X, Y=Y, Y_ref=Y, K=K, K_ref=K)

        if K is None:
            K = self.kernel(X, X)
            self.K = K
            K = center_kernel(K)
        else:
            self.K = K

        if X is not None:
            self.X = X
        else:
            print(
                "No input data supplied during fitting. \n"
                "Transformations/statistics only available for kernel inputs."
            )

        # Compute maximum eigenvalue of kernel matrix
        if Yhat is None:
            Yhat = K @ np.linalg.pinv(K, rcond=self.regularization) @ Y

        if len(Y.shape) == 1:
            Yhat = Yhat.reshape(-1, 1)

        K_pca = K
        K_lr = Yhat @ Yhat.T

        Kt = (self.alpha * K_pca) + (1.0 - self.alpha) * K_lr

        self.v, self.U = sorted_eig(Kt, thresh=self.regularization, n=self.n_PC)

        P_krr = np.linalg.solve(K + np.eye(len(K)) * self.regularization, Yhat)
        P_krr = P_krr @ Yhat.T

        P_kpca = np.eye(K.shape[0])

        P = (self.alpha * P_kpca) + (1.0 - self.alpha) * P_krr

        v_inv = eig_inv(self.v[:self.n_PC])

        self.PKT = P @ self.U[:, :self.n_PC] @ np.diagflat(np.sqrt(v_inv))
        T = K @ self.PKT

        PT = np.linalg.pinv(T.T @ T) @ T.T

        self.PTK = PT @ K
        self.PTY = PT @ Y
        self.PTX = PT @ X

    def transform(self, X=None, K=None):
        if self.PKT is None:
            raise Exception("Error: must fit the PCovR model before transforming")
        elif X is None and K is None:
            raise Exception("Either the kernel or input data must be specified.")
        else:
            X, _, K = self.preprocess(X=X, K=K)

            if K is None and self.X is not None:
                K = self.kernel(X, self.X)
                K = center_kernel(K, reference=self.K)
            elif K is None:
                raise Exception("This functionality is not available.")
                return

            T = K @ self.PKT
            Yp = T @ self.PTY
            Xr = T @ self.PTX

            Xr, Yp = self.postprocess(X=Xr, Y=Yp)

            if T.shape[1] == 1:
                T = np.array((T.reshape(-1), np.zeros(T.shape[0]))).T

            return T, Yp, Xr

    def loss(self, X=None, Y=None, K=None):
        if K is None and self.X is not None:
            K = self.kernel(X, self.X)

            K = center_kernel(K, reference=self.K)
        elif K is None:
            raise ValueError(
                "Must provide a kernel or a feature vector, in which case the "
                "train features should be available in the class"
            )

        T, Yp, Xr = self.transform(X=X, K=K)
        Kapprox = T @ self.PTK

        Lkpca = np.linalg.norm(K - Kapprox)**2 / np.linalg.norm(K)**2
        Lkrr = np.linalg.norm(Y - Yp)**2 / np.linalg.norm(Y)**2

        return Lkpca, Lkrr

    def statistics(self, X, Y, K=None):
        """
        Computes the loss values and reconstruction errors for
        KPCovR for the given predictor and response variables

        ---Arguments---
            X: independent (predictor) variable
            Y: dependent (response) variable

        ---Returns---
            dictionary of available statistics
        """

        if K is None and self.X is not None:
            K = self.kernel(X, self.X)
            K = center_kernel(K, reference=self.K)

        T, Yp, Xr = self.transform(X=X, K=K)

        Kapprox = T @ self.PTK

        return get_stats(x=X, y=Y, yp=Yp, t=T, xr=Xr,
                         k=K, kapprox=Kapprox)


class SparseKPCovR(PCovRBase, Sparsified):
    """
    Performs KPCovR, as described in Helfrecht (2020)
    which combines Kernel Principal Components Analysis (KPCA)
    and Kernel Ridge Regression (KRR), on a sparse active set
    of variables

    ----Inherited Attributes----
    PKT: projector from kernel space to latent space
    PKY: projector from kernel space to property space
    PTK: projector from latent space to kernel space
    PTX: projector from latent space to input space
    PTY: projector from latent space to property space
    PXT: projector from input space to latent space
    PXY: projector from input space to property space
    X: input used to train the model, if further kernels need be constructed
    X_sparse: active set used to train the model, if further kernels need be
              constructed
    Yhat: regressed properties
    alpha: (float) mixing parameter between decomposition and regression
    center: (boolean) whether to shift all inputs to zero mean
    kernel: function to construct the kernel of the input data
    n_PC (int) number of principal components to store
    n_active: (int) size of the active set
    regularization: (float) parameter to offset all small eigenvalues for
                    regularization
    scale: (boolean) whether to scale all inputs to unit variance

    ----Inherited Methods----
    postprocess: un-centers and un-scales outputs according to scale and
                 center attributes
    preprocess: centers and scales provided inputs according to scale and
                center attributes

    ---References---
        1.  S. de Jong, H. A. L. Kiers, 'Principal Covariates
            Regression: Part I. Theory', Chemometrics and Intelligent
            Laboratory Systems 14(1): 155-164, 1992
        2.  M. Vervolet, H. A. L. Kiers, W. Noortgate, E. Ceulemans,
            'PCovR: An R Package for Principal Covariates Regression',
            Journal of Statistical Software 65(1):1-14, 2015
    """

    def __init__(self, n_PC=None, *args, **kwargs):
        super(SparseKPCovR, self).__init__(n_PC=n_PC, *args, **kwargs)

    def fit(self, X, Y, X_sparse=None, Kmm=None, Knm=None):
        X, Y, _ = self.preprocess(X=X, X_ref=X, Y=Y, Y_ref=Y)

        if X_sparse is None:
            fps_idxs, _ = FPS(X, self.n_active)
            self.X_sparse = X[fps_idxs, :]
        else:
            self.X_sparse = X_sparse

        if Kmm is None:
            Kmm = self.kernel(self.X_sparse, self.X_sparse)

        if self.center:
            Kmm = center_kernel(Kmm)
        self.Kmm = Kmm

        if Knm is None:
            Knm = self.kernel(X, self.X_sparse)

        _, _, self.Knm = self.preprocess(K=Knm, K_ref=Knm)

        vmm, Umm = sorted_eig(
            Kmm, thresh=self.regularization, n=self.n_active)
        vmm_inv = eig_inv(vmm[:self.n_active - 1])

        phi_active = self.Knm @ Umm[:, :self.n_active - 1] @ np.diagflat(np.sqrt(vmm_inv))

        C = phi_active.T @ phi_active

        v_C, U_C = sorted_eig(C, thresh=0)
        U_C = U_C[:, v_C > 0]
        v_C = v_C[v_C > 0]
        v_C_inv = eig_inv(v_C)

        Csqrt = U_C @ np.diagflat(np.sqrt(v_C)) @ U_C.T
        iCsqrt = U_C @ np.diagflat(np.sqrt(v_C_inv)) @ U_C.T

        C_pca = C

        C_lr = np.linalg.pinv(C + self.regularization * np.eye(C.shape[0]))
        C_lr = iCsqrt @ phi_active.T @ phi_active @ C_lr @ phi_active.T

        if len(Y.shape) == 1:
            C_lr = C_lr @ Y.reshape(-1, 1)
        else:
            C_lr = C_lr @ Y

        C_lr = C_lr @ C_lr.T

        Ct = self.alpha * C_pca + (1 - self.alpha) * C_lr

        v_Ct, U_Ct = sorted_eig(Ct, thresh=0)

        PPT = iCsqrt @ U_Ct[:, :self.n_PC] @ np.diag(np.sqrt(v_Ct[:self.n_PC]))

        PKT = Umm[:, :self.n_active - 1] @ np.diagflat(np.sqrt(vmm_inv))

        self.PKT = PKT @ PPT

        T = self.Knm @ self.PKT

        PT = np.linalg.pinv(T.T @ T) @ T.T
        self.PTY = PT @ Y
        self.PTX = PT @ X

    def transform(self, X, Knm=None):
        X, _, Knm = self.preprocess(X=X, K=Knm)

        if self.PKT is None:
            raise Exception("Error: must fit the PCovR model before transforming")
        else:
            if Knm is None:
                Knm = self.kernel(X, self.X_sparse)
                _, _, Knm = self.preprocess(K=Knm)

            T = Knm @ self.PKT
            Yp = T @ self.PTY
            Xr = T @ self.PTX

            Xr, Yp = self.postprocess(X=Xr, Y=Yp)

            return T, Yp, Xr

    def loss(self, X, Y, Knm=None):
        if Knm is None:
            Knm = self.kernel(X, self.X_sparse)
            Knm -= np.mean(self.K_ref, axis=0)

        T, Yp, Xr = self.transform(X, Knm=Knm)

        Lkpca = np.linalg.norm(Xr - X)**2 / np.linalg.norm(X)**2
        Lkrr = np.linalg.norm(Y - Yp)**2 / np.linalg.norm(Y)**2

        return Lkpca, Lkrr

    def statistics(self, X, Y, Knm=None):
        if Knm is None:
            Knm = self.kernel(X, self.X_sparse)
            Knm -= np.mean(self.K_ref, axis=0)

        T, Yp, Xr = self.transform(X, Knm=Knm)

        return get_stats(x=X, y=Y, yp=Yp, t=T, xr=Xr)
