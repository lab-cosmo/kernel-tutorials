import numpy as np
from .general import FPS, get_stats, sorted_eig, eig_inv, center_matrix, normalize_matrix
from .kernels import linear_kernel, center_kernel, gaussian_kernel, summed_kernel

kernels = {
    "gaussian": gaussian_kernel,
    "linear": linear_kernel,
    "summed_gaussian": lambda x1, x2: summed_kernel(x1, x2, gaussian_kernel)
}


class Model:
    """
        Super-class defined for all models

        ----Attributes----
        center: (boolean) whether to shift all inputs to zero mean
        regularization: (float) parameter to offset all small eigenvalues for regularization
        scale: (boolean) whether to scale all inputs to unit variance

        ----Methods----
        postprocess: un-centers and un-scales outputs according to scale and center attributes
        preprocess: centers and scales provided inputs according to scale and center attributes

    """

    def __init__(self, regularization=1e-12, scale=False, center=False, *args, **kwargs):
        self.regularization = regularization

        self.center = center
        self.scale = scale

        self.X_center = 0
        self.Y_center = 0
        self.X_scale = 1
        self.Y_scale = 1
        self.K_ref = None

    def preprocess(self, X=None, Y=None, K=None,
                   X_ref=None, Y_ref=None, K_ref=None,
                   *args, **kwargs):
        """
            Scale and center the input data as designated by the
            model parameters `scale` and `center`. These parameters
            are set on the model level to enforce the same centering
            and scaling for all data supplied to the model.
            i.e. if a model is trained on centered input X, it must be
            supplied a similarly centered input X' for transformation.

        """

        if(self.center):
            if(X_ref is not None):
                self.X_center = X_ref.mean(axis=0)

            if(Y_ref is not None):
                self.Y_center = Y_ref.mean(axis=0)

            if(self.K_ref is None):
                if(K_ref is None):
                    K_ref = K

                self.K_ref = K_ref

        if(self.scale):
            if(X_ref is not None):
                self.X_scale = np.linalg.norm(
                    X_ref - self.X_center)/np.sqrt(X_ref.shape[0])

            if(Y_ref is not None):
                self.Y_scale = np.linalg.norm(
                    Y_ref - self.Y_center)/np.sqrt(Y_ref.shape[0])

        if(X is not None):
            Xcopy = X.copy()
            if(self.center):
                Xcopy = center_matrix(Xcopy, self.X_center)
            if(self.scale):
                Xcopy = normalize_matrix(Xcopy, scale=self.X_scale)
        else:
            Xcopy = None

        if(Y is not None):
            Ycopy = Y.copy()
            if(self.center):
                Ycopy = center_matrix(Ycopy, self.Y_center)
            if(self.scale):
                Y = normalize_matrix(Ycopy, scale=self.Y_scale)
        else:
            Ycopy = None

        if(K is not None):
            Kcopy = K.copy()
            if(self.center):
                print("Centering Kernel")
                Kcopy = center_kernel(Kcopy, reference=self.K_ref)
        else:
            Kcopy = None

        return Xcopy, Ycopy, Kcopy

    def postprocess(self, X=None, Y=None,
                    *args, **kwargs):
        """
            Undoes any scaling and center on the output data for comparison
        """

        if(X is not None):
            Xcopy = X.copy()
            if(self.center):
                Xcopy = center_matrix(Xcopy, -self.X_center)
            if(self.scale):
                Xcopy = normalize_matrix(Xcopy, scale=1.0/self.X_scale)
        else:
            Xcopy = None

        if(Y is not None):
            Ycopy = Y.copy()
            if(self.center):
                Ycopy = center_matrix(Ycopy, -self.Y_center)
            if(self.scale):
                Y = normalize_matrix(Ycopy, scale=1.0/self.Y_scale)
        else:
            Ycopy = None

        return Xcopy, Ycopy

class Decomposition(Model):
    """
        Super-class defined for any decompositions ala PCA

        ----Attributes----
        PTX: projector from latent space to input space
        PXT: projector from input space to latent space
        n_PCA (int) number of principal components to store

        ----Inherited Attributes----
        center: (boolean) whether to shift all inputs to zero mean
        regularization: (float) parameter to offset all small eigenvalues for regularization
        scale: (boolean) whether to scale all inputs to unit variance

        ----Inherited Methods----
        postprocess: un-centers and un-scales outputs according to scale and center attributes
        preprocess: centers and scales provided inputs according to scale and center attributes

    """

    def __init__(self, n_PC=2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_PCA = n_PC
        self.PXT = None
        self.PTX = None

class Regression(Model):
    """
        Super-class defined for any regressions

        ----Attributes----
        PXY: projector from input space to property space

        ----Inherited Attributes----
        center: (boolean) whether to shift all inputs to zero mean
        regularization: (float) parameter to offset all small eigenvalues for regularization
        scale: (boolean) whether to scale all inputs to unit variance

        ----Inherited Methods----
        postprocess: un-centers and un-scales outputs according to scale and center attributes
        preprocess: centers and scales provided inputs according to scale and center attributes

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.PXY = None

class Kernelized(Model):
    """
        Super-class defined for any kernelized methods

        ----Attributes----
        PKT: projector from kernel space to latent space
        PKY: projector from kernel space to property space
        PTK: projector from latent space to kernel space
        X: input used to train the model, if further kernels need be constructed
        kernel: function to construct the kernel of the input data

        ----Inherited Attributes----
        center: (boolean) whether to shift all inputs to zero mean
        regularization: (float) parameter to offset all small eigenvalues for regularization
        scale: (boolean) whether to scale all inputs to unit variance

        ----Inherited Methods----
        postprocess: un-centers and un-scales outputs according to scale and center attributes
        preprocess: centers and scales provided inputs according to scale and center attributes

    """

    def __init__(self, kernel_type='linear', *args, **kwargs):

        super().__init__(*args, **kwargs)

        if(isinstance(kernel_type, str)):
            if(kernel_type in kernels):
                self.kernel = kernels[kernel_type]
            else:
                raise Exception('Kernel Error: \
                                  Please specify either {}\
                                  or pass a suitable kernel function.\
                                '.format(kernels.keys()))
        elif(callable(kernel_type)):
            self.kernel = kernel_type
        else:
            raise Exception('Kernel Error: \
                              Please specify either {}\
                              or pass a suitable kernel function.\
                            '.format(kernels.keys()))

        self.PKT = None
        self.PTK = None
        self.PKY = None
        self.X = None

class Sparsified(Kernelized):
    """
        Super-class defined for any kernelized methods

        ----Attributes----
        n_active: (int) size of the active set
        X_sparse: active set used to train the model, if further kernels need be constructed

        ----Inherited Attributes----
        PKT: projector from kernel space to latent space
        PTK: projector from latent space to kernel space
        PTX: projector from latent space to input space
        X: input used to train the model, if further kernels need be constructed
        center: (boolean) whether to shift all inputs to zero mean
        kernel: function to construct the kernel of the input data
        regularization: (float) parameter to offset all small eigenvalues for regularization
        scale: (boolean) whether to scale all inputs to unit variance

        ----Inherited Methods----
        postprocess: un-centers and un-scales outputs according to scale and center attributes
        preprocess: centers and scales provided inputs according to scale and center attributes

    """

    def __init__(self, n_active, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_active = n_active
        self.X_sparse = None

class PCovRBase(Decomposition, Regression):
    """
        Super-class defined for any decompositions ala PCA

        ----Attributes----
        PTY: projector from latent space to property space
        Yhat: regressed properties
        alpha: (float) mixing parameter between decomposition and regression

        ----Inherited Attributes----
        PTX: projector from latent space to input space
        PXT: projector from input space to latent space
        n_PCA (int) number of principal components to store
        PXY: projector from input space to property space
        center: (boolean) whether to shift all inputs to zero mean
        regularization: (float) parameter to offset all small eigenvalues for regularization
        scale: (boolean) whether to scale all inputs to unit variance

        ----Inherited Methods----
        postprocess: un-centers and un-scales outputs according to scale and center attributes
        preprocess: centers and scales provided inputs according to scale and center attributes

    """

    def __init__(self, alpha, *args, **kwargs):
        super(PCovRBase, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.PTY = None
        self.Yhat = None

class PCA(Decomposition):
    """
        Performs principal component analysis

        ----Inherited Attributes----
        PTX: projector from latent space to input space
        PXT: projector from input space to latent space
        center: (boolean) whether to shift all inputs to zero mean
        n_PCA: (int) number of principal components to store
        regularization: (float) parameter to offset all small eigenvalues for regularization
        scale: (boolean) whether to scale all inputs to unit variance

        ----Methods----
        fit: fit the PCA
        statistics: provide available statistics for the decomposition
        transform: transform data based on the PCA fit

        ----Inherited Methods----
        postprocess: un-centers and un-scales outputs according to scale and center attributes
        preprocess: centers and scales provided inputs according to scale and center attributes

        ----References----
                1.  https: /  / en.wikipedia.org / wiki / Principal_component_analysis
                2.  M. E. Tipping 'Sparse Kernel Principal Component Analysis',
                    Advances in Neural Information Processing Systems 13, 633 - 639, 2001
    """

    def __init__(self, n_PCA=None, *args, **kwargs):

        super().__init__(n_PC = n_PCA, *args, **kwargs)

    def fit(self, X):
        """
            Fits the PCA

        ----Arguments----
            X: centered data on which to build the PCA
        """

        X, _, _ = self.preprocess(X=X, X_ref=X)

        # Compute covariance
        C = (X.T @ X) / (X.shape[0] - 1)

        # Compute eigendecomposition of covariance matrix
        v, U = sorted_eig(C, thresh=self.regularization, n=self.n_PCA)

        self.PXT = U[:, :self.n_PCA]
        self.PTX = self.PXT.T

    def transform(self, X):
        """
            Transforms the PCA

        ----Arguments----
            X: centered data to transform based on the PCA
        """

        X, _, _ = self.preprocess(X=X)

        if self.PXT is None:
            print("Error: must fit the PCA before transforming")
        else:

            # Compute PCA scores
            T = X @ self.PXT
            return T

    def statistics(self, X):
        T = self.transform(X)

        X_PCA = T @ self.PTX
        X_PCA, _ = self.postprocess(X=X_PCA)

        return get_stats(x=X, t=T, xr=X_PCA)

class LR(Regression):
    """
        Performs linear regression


        ----Inherited Attributes----
        PXY: projector from input space to property space
        center: (boolean) whether to shift all inputs to zero mean
        regularization: (float) parameter to offset all small eigenvalues for regularization
        scale: (boolean) whether to scale all inputs to unit variance

        ----Methods----
        fit: fit the linear regression model by computing regression weights
        statistics: provide available statistics for the regression
        transform: compute predicted Y values

        ----Inherited Methods----
       postprocess: un-centers and un-scales outputs according to scale and center attributes
       preprocess: centers and scales provided inputs according to scale and center attributes

        ----References----
        1.  S. de Jong, H. A. L. Kiers, 'Principal Covariates
            Regression: Part I. Theory', Chemometrics and Intelligents
            Laboratory Systems 14(1): 155-164, 1992
        2.  https://en.wikipedia.org/wiki/Linear_regression
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, Y):
        """
            Fits the linear regression model

        ----Arguments----
            X: centered, independent (predictor) variable
            Y: centered, dependent (response) variable
        """

        X, Y, _ = self.preprocess(X=X, Y=Y, X_ref=X, Y_ref=Y)

        # Compute inverse of covariance
        XTX = (X.T @ X)
        XTX = XTX + self.regularization * np.eye(X.shape[1])
        XTX = np.linalg.pinv(XTX)

        # Compute LR solution
        self.PXY = XTX @ X.T @ Y

    def transform(self, X):
        """
            Computes predicted Y values

        ----Arguments----
            X: centered, independent (predictor) variable
        """

        X, _, _ = self.preprocess(X=X)

        # Compute predicted Y
        Yp = X @ self.PXY
        _, Yp = self.postprocess(Y=Yp)

        return Yp

    def statistics(self, X, Y):
        Yp = self.transform(X)
        return get_stats(x=X, y=Y, yp=Yp)

class KPCA(Kernelized, Decomposition):
    """
        Performs kernel principal component analysis on a dataset
        based on a kernel between all of the constituent data points


        ----Inherited Attributes----
        PKT: projector from kernel space to latent space
        PKY: projector from kernel space to property space (null)
        PTK: projector from latent space to kernel space
        PXT: projector from input space to latent space
        X: input used to train the model, if further kernels need be constructed
        center: (boolean) whether to shift all inputs to zero mean
        kernel: function to construct the kernel of the input data
        n_PCA (int) number of principal components to store
        regularization: (float) parameter to offset all small eigenvalues for regularization
        scale: (boolean) whether to scale all inputs to unit variance

        ----Methods----
        fit: fit the KPCA
        statistics: provide available statistics for the decomposition
        transform: transform data based on the KPCA fit

        ----Inherited Methods----
        postprocess: un-centers and un-scales outputs according to scale and center attributes
        preprocess: centers and scales provided inputs according to scale and center attributes

        ----References----
                1.  https: /  / en.wikipedia.org / wiki / Kernel_principal_component_analysis
                2.  M. E. Tipping, 'Sparse Kernel Principal Component Analysis',
                    Advances in Neural Information Processing Systems 13, 633 - 639, 2001
    """

    def __init__(self, n_KPCA, *args, **kwargs):

        super(KPCA, self).__init__(n_PC=n_KPCA, *args, **kwargs)

    def fit(self, X=None, K=None):

        X, _, K = self.preprocess(X=X, K=K, X_ref=X, K_ref=K)

        if(K is None):
            K = self.kernel(X, X)
            K = center_kernel(K)

        if(X is not None):
            self.X = X
        else:
            print("No input data supplied during fitting. \
                   \nTransformations/statistics only available for kernel inputs.")

        # Compute eigendecomposition of kernel
        v, U = sorted_eig(K,
                          thresh=self.regularization, n=self.n_PCA)

        v_inv = eig_inv(v[:self.n_PCA])

        self.PKT = U[:, :self.n_PCA] @ np.diagflat(np.sqrt(v_inv))

        T = K @ self.PKT
        self.PTK = np.diagflat(v_inv) @ T.T @ K

        if(X is not None):
            self.PTX = np.diagflat(v_inv) @ T.T @ X

    def transform(self, X=None, K=None):

        X, _, K = self.preprocess(X=X, K=K)

        if self.PKT is None:
            print("Error: must fit the KPCA before transforming")
        elif X is None and K is None:
            print("Either the kernel or input data must be specified.")
        else:

            # Compute KPCA transformation
            if(K is None):
                K = self.kernel(X, self.X)
                K = center_kernel(K, reference=self.kernel(self.X, self.X))

            T = K @ self.PKT

            return T

    def statistics(self, X=None, Y=None, K=None):
        if X is None and K is None:
            print("Either the kernel or input data must be specified.")
        else:
            # Compute KPCA transformation

            if(K is None):
                K = self.kernel(X, self.X)
                K = center_kernel(K, reference=self.kernel(self.X, self.X))

            T = K @ self.PKT

            Kapprox = T @ self.PTK

            if(self.PTX is not None):
                Xr = T @ self.PTX
                Xr, _ = self.postprocess(X=Xr)
            else:
                Xr = None

            return get_stats(k=K, kapprox=Kapprox, x=X, xr=Xr,
                             t=T)

class KRR(Kernelized, Regression):
    """
        Performs kernel ridge regression

        ----Inherited Attributes----
        PKT: projector from kernel space to latent space
        PKY: projector from kernel space to property space
        PTK: projector from latent space to kernel space
        X: input used to train the model, if further kernels need be constructed
        center: (boolean) whether to shift all inputs to zero mean
        kernel: function to construct the kernel of the input data
        regularization: (float) parameter to offset all small eigenvalues for regularization
        scale: (boolean) whether to scale all inputs to unit variance

        ----Methods----
        fit: fit the kernel ridge regression model by computing regression weights
        statistics: provide available statistics for the regression
        transform: compute predicted Y values

        ----Inherited Methods----
        postprocess: un-centers and un-scales outputs according to scale and center attributes
        preprocess: centers and scales provided inputs according to scale and center attributes

        ----References----
        1.  M. Ceriotti, M. J. Willatt, G. Csanyi,
            'Machine Learning of Atomic - Scale Properties
            Based on Physical Principles', Handbook of Materials Modeling,
            Springer, 2018
    """

    def __init__(self, *args, **kwargs):

        super(KRR, self).__init__(*args, **kwargs)

    def fit(self, X=None, Y=None, K=None):

        X, Y, K = self.preprocess(X=X, K=K, Y=Y, X_ref=X, K_ref=K, Y_ref=Y)

        if(K is None):
            K = self.kernel(X, X)
            self.K = K.copy()
            K = center_kernel(K, self.K)

        if(X is not None):
            self.X = X
        else:
            print("No input data supplied during fitting. \
                   \nTransformations/statistics only available for kernel inputs.")

        # Regularize the model
        Kreg = K + np.eye(K.shape[0]) * self.regularization

        # Solve the model
        self.PKY = np.linalg.solve(Kreg, Y)

    def transform(self, X=None, K=None):

        X, _, K = self.preprocess(X=X, K=K)

        if self.PKY is None:
            print("Error: must fit the KPCA before transforming")
        elif X is None and K is None:
            print("Either the kernel or input data must be specified.")
        else:

            # Compute KPCA transformation
            if(K is None):
                K = self.kernel(X, self.X)
                K = center_kernel(K, reference=self.K)

            Yp = K @ self.PKY
            _, Yp = self.postprocess(Y=Yp)

            return Yp

    def statistics(self, X=None, Y=None, K=None):

        Yp = self.transform(X=X, K=K)

        return get_stats(x=X, y=Y, yp=Yp, k=K)

class SparseKPCA(Sparsified, Decomposition):
    """
        ----Inherited Attributes----
        PKT: projector from kernel space to latent space
        PKY: projector from kernel space to property space
        PKY: projector from kernel space to property space (null)
        PTK: projector from latent space to kernel space
        PXT: projector from input space to latent space
        X: input used to train the model, if further kernels need be constructed
        X_sparse: active set used to train the model, if further kernels need be constructed
        center: (boolean) whether to shift all inputs to zero mean
        kernel: function to construct the kernel of the input data
        n_KPCA (int) number of principal components to store
        n_active: (int) size of the active set
        regularization: (float) parameter to offset all small eigenvalues for regularization
        scale: (boolean) whether to scale all inputs to unit variance

        ----Methods----
        fit: fit the KPCA
        statistics: provide available statistics for the decomposition
        transform: transform data based on the KPCA fit

        ----Inherited Methods----
        postprocess: un-centers and un-scales outputs according to scale and center attributes
        preprocess: centers and scales provided inputs according to scale and center attributes

        ----References----
                1.  https: /  / en.wikipedia.org / wiki / Kernel_principal_component_analysis
                2.  M. E. Tipping, 'Sparse Kernel Principal Component Analysis',
                    Advances in Neural Information Processing Systems 13, 633 - 639, 2001
    """

    def __init__(self, n_KPCA, *args, **kwargs):

        super(SparseKPCA, self).__init__(n_PC=n_KPCA, *args, **kwargs)

    def fit(self, X, Kmm=None, Knm=None, *args, **kwargs): # args left in for backwards compatibility

        X, _, Knm = self.preprocess(X=X, X_ref=X, K=Knm, K_ref=Kmm)
        _, _, Kmm = self.preprocess(K=Kmm, K_ref=Kmm)

        if(Kmm is None or Knm is None):

            i_sparse, _ = FPS(X, self.n_active)

            self.X_sparse = X[i_sparse, :]

            Kmm = self.kernel(self.X_sparse, self.X_sparse)
            Kmm = center_kernel(Kmm)

            Knm = self.kernel(X, self.X_sparse)
            Knm = center_kernel(Knm, Kmm)

        self.Kmm = Kmm
        self.barKM = np.mean(Knm, axis=0)

        # Compute eigendecomposition of kernel
        vmm, Umm = sorted_eig(
            Kmm, thresh=self.regularization, n=self.n_active)

        U_active = Umm[:, :self.n_active - 1]
        v_invsqrt = np.diagflat(np.sqrt(eig_inv(vmm[0:self.n_active - 1])))
        U_active = U_active @ v_invsqrt

        phi_active = (Knm - self.barKM) @ U_active

        C = phi_active.T @ phi_active

        v_C, U_C = sorted_eig(
            C, thresh=self.regularization, n=self.n_active)

        self.PKT = U_active @ U_C[:, :self.n_PCA]
        T = (Knm - self.barKM) @ self.PKT
        self.PTX = np.diagflat(eig_inv(v_C[:self.n_PCA])) @ T.T @ X

    def transform(self, X, Knm=None):

        X, _, Knm = self.preprocess(X=X, K=Knm)

        if self.PKT is None:
            print("Error: must fit the KPCA before transforming")
        else:
            if(Knm is None and self.X_sparse is not None):
                Knm = self.kernel(X, self.X_sparse)
                Knm = center_kernel(Knm, reference=self.Kmm)

            # Compute KPCA transformation
            T = (Knm - self.barKM) @ self.PKT

            return T

    def statistics(self, X, Knm=None, K_test=None):

        T = self.transform(X, Knm=Knm)
        Kapprox = T @ T.T

        if(K_test is None):
            K_test = self.kernel(X, X)
            K_test = center_kernel(K_test)

        Xr = T @ self.PTX
        Xr, _ = self.postprocess(X=Xr)

        return get_stats(k=K_test, kapprox=Kapprox, x=X, xr=Xr,
                         t=T)

class SparseKRR(Sparsified, Regression):
    """
        Performs sparsified kernel ridge regression

        ----Inherited Attributes----
        PKT: projector from kernel space to latent space
        PTK: projector from latent space to kernel space
        PTX: projector from latent space to input space
        PXY: projector from input space to property space
        X: input used to train the model, if further kernels need be constructed
        X_sparse: active set used to train the model, if further kernels need be constructed
        center: (boolean) whether to shift all inputs to zero mean
        kernel: function to construct the kernel of the input data
        n_active: (int) size of the active set
        regularization: (float) parameter to offset all small eigenvalues for regularization
        scale: (boolean) whether to scale all inputs to unit variance

        ----Inherited Methods----
        postprocess: un-centers and un-scales outputs according to scale and center attributes
        preprocess: centers and scales provided inputs according to scale and center attributes


        ----Methods----
        fit: fit the kernel ridge regression model by computing regression weights
        statistics: provide available statistics for the regression
        transform: compute predicted Y values


        ----References----
                1.  M. Ceriotti, M. J. Willatt, G. Csanyi,
                    'Machine Learning of Atomic - Scale Properties
                    Based on Physical Principles', Handbook of Materials Modeling,
                    Springer, 2018
                2.  A. J. Smola, B. Scholkopf, 'Sparse Greedy Matrix Approximation
                    for Machine Learning', Proceedings of the 17th International
                    Conference on Machine Learning, 911 - 918, 2000
    """

    def __init__(self, *args, **kwargs):

        super(SparseKRR, self).__init__(*args, **kwargs)

    def fit(self, X, Y, Kmm=None, Knm=None):

        X, Y, Knm = self.preprocess(X=X, X_ref=X, Y=Y, Y_ref=Y, K=Knm, K_ref=Kmm)

        i_sparse, _ = FPS(X, self.n_active)

        self.X_sparse = X[i_sparse, :]

        if(Kmm is None):
            Kmm = self.kernel(self.X_sparse, self.X_sparse)
            Kmm = center_kernel(Kmm)
        self.Kmm = Kmm

        if(Knm is None):
            Knm = self.kernel(X, self.X_sparse)
            Knm = center_kernel(Knm, Kmm)

        # Compute max eigenvalue of regularized model
        PKY = np.linalg.pinv(Knm.T @ Knm + (self.regularization * Kmm))
        self.PKY = PKY @ Knm.T @ Y

    def transform(self, X, Knm=None):

        X, _, Knm = self.preprocess(X=X, K=Knm)

        if self.PKY is None:
            print("Error: must fit the KRR model before transforming")
        else:
            if(Knm is None):
                Knm = self.kernel(X, self.X_sparse)

            Yp = Knm @ self.PKY
            _, Yp = self.postprocess(Y=Yp)

            return Yp

    def statistics(self, X, Y, Knm=None):
        Yp = self.transform(X, Knm=Knm)

        return get_stats(x=X, y=Y, yp=P)

class MDS(Decomposition):
    """
        Performs multidimensional scaling

        ----Attributes----
        n_MDS: (int) number of principal components to store

        ----Inherited Attributes----
        PTX: projector from latent space to input space
        PXT: projector from input space to latent space
        center: (boolean) whether to shift all inputs to zero mean
        regularization: (float) parameter to offset all small eigenvalues for regularization
        scale: (boolean) whether to scale all inputs to unit variance

        ----Methods----
        fit: fit the PCA
        statistics: provide available statistics for the decomposition
        transform: transform data based on the PCA fit

        ----Inherited Methods----
        postprocess: un-centers and un-scales outputs according to scale and center attributes
        preprocess: centers and scales provided inputs according to scale and center attributes

        ----References----
        1.  https: /  / en.wikipedia.org / wiki / Multidimensional_scaling
        2.  Torgerson, W.S. 'Multidimensional scaling: I. Theory and method',
            Psychometrika 17, 401 - 419, 1952
            https: /  / doi.org / 10.1007 / BF02288916
    """

    def __init__(self, n_MDS=None, *args, **kwargs):

        # Initialize attributes
        super().__init__(n_PC=n_MDS, *args, **kwargs)

    def fit(self, X):
        """
            Fits the PCA

        ----Arguments----
            X: centered and normalized data on which to build the MDS
        """

        X, _, _ = self.preprocess(X=X, X_ref=X)

        # Compute covariance
        K = X @ X.T

        # Compute eigendecomposition of covariance matrix
        v, U = sorted_eig(K, thresh=self.regularization, n=self.n_PCA)

        T = U[:, :self.n_PCA] @ np.diagflat(np.sqrt(v[:self.n_PCA]))

        self.PXT = np.linalg.lstsq(X, T, rcond=None)[0]
        self.PTX = np.linalg.lstsq(T, X, rcond=None)[0]

    def transform(self, X):
        """
            Transforms using MDS

        ----Arguments----
            X: centered data to transform based on MDS
        """

        X, _, _ = self.preprocess(X=X)


        if self.PXT is None:
            print("Error: must fit the MDS before transforming")
        else:

            # Compute PCA scores
            T = X @ self.PXT
            return T

    def statistics(self, X):
        T = self.transform(X)
        Xr = T @ self.PTX
        Xr, _ = self.postprocess(X=Xr)

        stats = get_stats(x=X, t=T, xr=Xr)
        kernel_error = np.linalg.norm((X @ X.T) - (T @ T.T)) ** 2.0
        stats['Strain'] = kernel_error / np.linalg.norm((X @ X.T))

        return stats

class PCovR(PCovRBase):
    """
        Performs PCovR, detecting whether the data set is in Structure or
        Feature Space

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
        n_PCA (int) number of principal components to store
        regularization: (float) parameter to offset all small eigenvalues for regularization
        scale: (boolean) whether to scale all inputs to unit variance

        ----Inherited Methods----
        postprocess: un-centers and un-scales outputs according to scale and center attributes
        preprocess: centers and scales provided inputs according to scale and center attributes


        ----References----
                1.  S. de Jong, H. A. L. Kiers, 'Principal Covariates
                    Regression: Part I. Theory', Chemometrics and Intelligent
                    Laboratory Systems 14(1): 155-164, 1992
                2.  M. Vervolet, H. A. L. Kiers, W. Noortgate, E. Ceulemans,
                    'PCovR: An R Package for Principal Covariates Regression',
                    Journal of Statistical Software 65(1):1-14, 2015
    """

    def __init__(self, alpha=0.0, n_PCA=None, space='auto', *args, **kwargs):

        super().__init__(alpha=alpha, n_PC=n_PCA, *args, **kwargs)
        self.space = space

    def compute_K(self, X, Y, Yhat=None):

        X, Y, _ = self.preprocess(X=X, X_ref=X, Y=Y, Y_ref=Y)

        # Compute K
        if(Yhat is None):
            lr = LR(regularization=self.regularization)
            lr.fit(X, Y)

            self.Yhat = lr.transform(X)

        else:
            self.Yhat = Yhat

        if(len(Y.shape) == 1):
            self.Yhat = self.Yhat.reshape(-1, 1)

        K_pca = (X @ X.T)
        K_lr = (self.Yhat @ self.Yhat.T)

        self.K = (self.alpha * K_pca) + (1.0 - self.alpha) * K_lr

    def fit_feature_space(self, X, Y, Yhat=None):

        self.compute_K(X, Y, Yhat=Yhat)

        # Compute the inverse square root of the covariance of X
        C = (X.T @ X)
        v_C, U_C = sorted_eig(C, thresh=self.regularization)
        U_C = U_C[:, v_C > self.regularization]
        v_C = v_C[v_C > self.regularization]

        Csqrt = (U_C @ np.diagflat(np.sqrt(v_C)) @ U_C.T)
        iCsqrt = (U_C @ np.diagflat(np.sqrt(eig_inv(v_C))) @ U_C.T)

        Ct = (iCsqrt @ X.T)
        Ct = (Ct @ self.K @ Ct.T)

        v_Ct, U_Ct = sorted_eig(Ct, thresh=self.regularization, n=self.n_PCA)

        v_inv = eig_inv(v_Ct[:self.n_PCA])

        PXV = iCsqrt @ U_Ct[:, :self.n_PCA]

        self.PXT = PXV @ np.diagflat(np.sqrt(v_Ct[:self.n_PCA]))
        self.PTX = np.diagflat(np.sqrt(v_inv)) @ U_Ct[:, :self.n_PCA].T @ Csqrt
        PTY = np.diagflat(np.sqrt(v_inv)) @ U_Ct[:, :self.n_PCA].T @ iCsqrt
        self.PTY = PTY @ X.T @ Y

    def fit_structure_space(self, X, Y, Yhat=None):

        self.compute_K(X, Y, Yhat=Yhat)
        v, U = sorted_eig(
            self.K, thresh=self.regularization, n=self.n_PCA)

        v_inv = eig_inv(v[:self.n_PCA])
        T = U[:, :self.n_PCA] @ np.diagflat(np.sqrt(v[:self.n_PCA]))

        P_lr = (X.T @ X) + np.eye(X.shape[1]) * self.regularization
        P_lr = np.linalg.pinv(P_lr)
        P_lr = P_lr @ X.T @ Y

        if(len(Y.shape) == 1):
            P_lr = P_lr.reshape((-1, 1))

        P_lr = P_lr @ self.Yhat.T

        P_pca = X.T

        P = (self.alpha * P_pca) + (1.0 - self.alpha) * P_lr
        self.PXT = P @ U[:, :self.n_PCA] @ np.diag(np.sqrt(v_inv))
        self.PTY =  np.diagflat(v_inv) @  T.T @ Y

        self.PTX = np.diagflat(v_inv) @ T.T @ X

    def fit(self, X, Y, Yhat=None):

        X, Y, _ = self.preprocess(X=X, Y=Y)

        sample_heavy = X.shape[0] > X.shape[1]
        if((self.space == 'feature' or sample_heavy) and self.space != 'structure'):
            if(X.shape[0] > X.shape[1] and self.space != 'feature'):
                print("# samples > # features, computing in feature space")
            self.fit_feature_space(X, Y, Yhat=Yhat)
        elif(self.space == 'structure' or not sample_heavy):
            if(sample_heavy and self.space != 'structure'):
                print("# samples < # features, computing in structure space")
            self.fit_structure_space(X, Y, Yhat=Yhat)
        else:
            raise Exception('Space Error: \
                              Please specify either space = "structure",\
                              "feature", or "auto" to designate the space to \
                              compute in.\
                            ')

    def transform(self, X):
        if self.PXT is None or self.PTY is None:
            print("Error: must fit the PCovR model before transforming")
        else:
            # Undo whitening of PCA scores and scale by norm
            # Xp = np.linalg.norm(X)*

            T = X @ self.PXT

            Yp = X @ self.PXT @ self.PTY

            Xr = T @ self.PTX

            Xr, Yp = self.postprocess(X=Xr, Y=Yp)

            return T, Yp, Xr

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
        Performs KPCovR, as described in Helfrecht (2020),
        which combines Kernel Principal Components Analysis (KPCA)
        and Kernel Ridge Regression (KRR)

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
        n_PCA (int) number of principal components to store
        regularization: (float) parameter to offset all small eigenvalues for regularization
        scale: (boolean) whether to scale all inputs to unit variance

        ----Inherited Methods----
        postprocess: un-centers and un-scales outputs according to scale and center attributes
        preprocess: centers and scales provided inputs according to scale and center attributes

        ---References---
        1.  S. de Jong, H. A. L. Kiers, 'Principal Covariates
            Regression: Part I. Theory', Chemometrics and Intelligent
            Laboratory Systems 14(1): 155-164, 1992
        2.  M. Vervolet, H. A. L. Kiers, W. Noortgate, E. Ceulemans,
            'PCovR: An R Package for Principal Covariates Regression',
            Journal of Statistical Software 65(1):1-14, 2015
    """

    def __init__(self, *args, **kwargs):

        super(KPCovR, self).__init__(*args, **kwargs)

    def fit(self, X, Y, K=None, Yhat=None):

        X, Y, K = self.preprocess(X=X, X_ref=X, Y=Y, Y_ref=Y, K=K, K_ref=K)

        if(K is None):
            K = self.kernel(X, X)
            self.K = K
            K = center_kernel(K)
        else:
            self.K = K

        if(X is not None):
            self.X = X
        else:
            print("No input data supplied during fitting." +
                  "\nTransformations and statistics only" +
                  "available with pre-computed kernels.")

        # Compute maximum eigenvalue of kernel matrix

        if(Yhat is None):
            Yhat = K @ np.linalg.pinv(K, rcond=self.regularization) @ Y

        if(len(Y.shape) == 1):
            Yhat = Yhat.reshape(-1, 1)

        K_pca = K #/ (np.trace(K) / X.shape[0])
        K_lr = Yhat @ Yhat.T

        self.Kt = (self.alpha * K_pca) + (1.0 - self.alpha) * K_lr

        self.v, self.U = sorted_eig(
            self.Kt, thresh=self.regularization, n=self.n_PCA)

        P_krr = np.linalg.solve(K + np.eye(len(K)) * self.regularization, Yhat)
        P_krr = P_krr @ Yhat.T

        P_kpca = np.eye(K.shape[0]) #/ (np.trace(K) / K.shape[0])

        P = (self.alpha * P_kpca) + (1.0 - self.alpha) * P_krr

        v_inv = eig_inv(self.v[:self.n_PCA])

        self.PKT = P @ self.U[:, :self.n_PCA] @ np.diagflat(np.sqrt(v_inv))
        T = K @ self.PKT

        PT = np.linalg.pinv(T.T @ T) @ T.T

        self.PTK = PT @ K
        self.PTY = PT @ Y
        self.PTX = PT @ X

    def transform(self, X=None, K=None):


        if self.PKT is None:
            print("Error: must fit the PCovR model before transforming")
        elif(X is None and K is None):
            print("Either the kernel or input data must be specified.")
        else:

            X, _, K = self.preprocess(X=X, K=K)

            if(K is None and self.X is not None):
                K = self.kernel(X, self.X)
                K = center_kernel(K, reference=self.K)
            elif(K is None):
                print("This functionality is not available.")
                return

            T = K @ self.PKT
            Yp = T @ self.PTY
            Xr = T @ self.PTX

            Xr, Yp = self.postprocess(X=Xr, Y=Yp)

            if(T.shape[1] == 1):
                T = np.array((T.reshape(-1), np.zeros(T.shape[0]))).T

            return T, Yp, Xr

    def loss(self, X=None, Y=None, K=None):

        if(K is None and self.X is not None):
            K = self.kernel(X, self.X)

            K = center_kernel(K, reference=self.K)
        elif(K is None):
            raise ValueError(
                "Must provide a kernel or a feature vector, in which case the train features should be available in the class")

        T, Yp, Xr = self.transform(X=X, K=K)
        Kapprox = T @ self.PTK

        Lkpca = np.linalg.norm(K - Kapprox)**2/np.linalg.norm(K)**2
        Lkrr = np.linalg.norm(Y - Yp)**2/np.linalg.norm(Y)**2

        return Lkpca, Lkrr

    def statistics(self, X, Y, K=None):
        """
            Computes the loss values and reconstruction errors for
            KPCovR for the given predictor and response variables

            ---Arguments---
            X:              independent (predictor) variable
            Y:              dependent (response) variable

            ---Returns---
            dictionary of available statistics

        """

        if(K is None and self.X is not None):
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
        X_sparse: active set used to train the model, if further kernels need be constructed
        Yhat: regressed properties
        alpha: (float) mixing parameter between decomposition and regression
        center: (boolean) whether to shift all inputs to zero mean
        kernel: function to construct the kernel of the input data
        n_PCA (int) number of principal components to store
        n_active: (int) size of the active set
        regularization: (float) parameter to offset all small eigenvalues for regularization
        scale: (boolean) whether to scale all inputs to unit variance

        ----Inherited Methods----
        postprocess: un-centers and un-scales outputs according to scale and center attributes
        preprocess: centers and scales provided inputs according to scale and center attributes

        ---References---
        1.  S. de Jong, H. A. L. Kiers, 'Principal Covariates
            Regression: Part I. Theory', Chemometrics and Intelligent
            Laboratory Systems 14(1): 155-164, 1992
        2.  M. Vervolet, H. A. L. Kiers, W. Noortgate, E. Ceulemans,
            'PCovR: An R Package for Principal Covariates Regression',
            Journal of Statistical Software 65(1):1-14, 2015
    """

    def __init__(self, *args, **kwargs):

        super(SparseKPCovR, self).__init__(*args, **kwargs)

    def fit(self, X, Y, X_sparse=None, Kmm=None, Knm=None):

        X, Y, Knm = self.preprocess(X=X, X_ref=X, Y=Y, Y_ref=Y, K=Knm, K_ref=Kmm)

        if(X_sparse is None):
            fps_idxs, _ = FPS(X, self.n_active)
            self.X_sparse = X[fps_idxs, :]
        else:
            self.X_sparse = X_sparse

        if(Kmm is None):
            Kmm = self.kernel(self.X_sparse, self.X_sparse)
            self.Kmm = Kmm
            Kmm = center_kernel(Kmm)
        else:
            self.Kmm = Kmm

        if(Knm is None):
            Knm = self.kernel(X, self.X_sparse)
            Knm = center_kernel(Knm, reference=self.Kmm)
        self.Knm = Knm

        vmm, Umm = sorted_eig(
            Kmm, thresh=self.regularization, n=self.n_active)
        vmm_inv = eig_inv(vmm[:self.n_active-1])

        self.barKM = np.mean(self.Knm, axis=0)

        phi_active = self.Knm @ Umm[:, :self.n_active-1] @ np.diagflat(np.sqrt(vmm_inv))

        barPhi = np.mean(phi_active, axis=0)
        phi_active -= barPhi

        C = phi_active.T @ phi_active

        v_C, U_C = sorted_eig(C, thresh=0)
        U_C = U_C[:, v_C > 0]
        v_C = v_C[v_C > 0]
        v_C_inv = eig_inv(v_C)

        Csqrt = U_C @ np.diagflat(np.sqrt(v_C)) @ U_C.T
        iCsqrt = U_C @ np.diagflat(np.sqrt(v_C_inv)) @ U_C.T

        C_pca = C #/ (np.trace(C)/C.shape[0])

        C_lr = np.linalg.pinv(C + self.regularization*np.eye(C.shape[0]))
        C_lr = iCsqrt @ phi_active.T @ phi_active @ C_lr @ phi_active.T

        if(len(Y.shape) == 1):
            C_lr = C_lr @ Y.reshape(-1, 1)
        else:
            C_lr =  C_lr @ Y

        C_lr = C_lr @ C_lr.T

        Ct = self.alpha*C_pca + (1-self.alpha)*C_lr

        v_Ct, U_Ct = sorted_eig(Ct, thresh=0)

        PPT = iCsqrt @  U_Ct[:, :self.n_PCA] @ np.diag(np.sqrt(v_Ct[:self.n_PCA]))

        PKT =  Umm[:, :self.n_active-1] @ np.diagflat(np.sqrt(vmm_inv))

        self.PKT = PKT @ PPT
        self.barT = barPhi @ PPT

        T = (self.Knm @ self.PKT) - self.barT

        PT = np.linalg.pinv(T.T @ T) @ T.T
        self.PTY = PT @ Y
        self.PTX = PT @ X

    def transform(self, X, Knm=None):

        X, _, Knm = self.preprocess(X=X, K=Knm)

        if self.PKT is None:
            print("Error: must fit the PCovR model before transforming")
        else:

            if(Knm is None):
                Knm = self.kernel(X, self.X_sparse)
                Knm = center_kernel(Knm, reference=self.Kmm)

            T = (Knm @ self.PKT) - self.barT
            Yp = T @ self.PTY
            Xr = T @ self.PTX

            Xr, Yp = self.postprocess(X=Xr, Y=Yp)

            return T, Yp, Xr

    def loss(self, X, Y, Knm=None):

        if(Knm is None):
            Knm = self.kernel(X, self.X_sparse)
            Knm = center_kernel(Knm, reference=self.Kmm)

        T, Yp, Xr = self.transform(X, Knm=Knm)

        Lkpca = np.linalg.norm(Xr - X)**2/np.linalg.norm(X)**2
        Lkrr = np.linalg.norm(Y - Yp)**2/np.linalg.norm(Y)**2

        return Lkpca, Lkrr

    def statistics(self, X, Y, Knm=None):

        if(Knm is None):
            Knm = self.kernel(X, self.X_sparse)
            Knm = center_kernel(Knm, reference=self.Kmm)

        T, Yp, Xr = self.transform(X, Knm=Knm)

        return get_stats(x=X, y=Y, yp=Yp, t=T, xr=Xr)
