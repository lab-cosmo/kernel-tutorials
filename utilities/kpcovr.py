import numpy as np
from scipy.spatial.distance import cdist
from .general import get_stats, FPS, sorted_eig, eig_inv
from .classes import KRR


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


def gaussian_kernel(XA, XB, gamma=1.0):
    """
        Builds a gaussian kernel

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


kernels = {"gaussian": gaussian_kernel,
           "linear": linear_kernel,
           "summed_gaussian": lambda x1, x2: summed_kernel(x1, x2, gaussian_kernel)
           }


class KPCovR:

    """
        Performs KPCovR, as described in Helfrecht (2020),
        which combines Kernel Principal Components Analysis (KPCA)
        and Kernel Ridge Regression (KRR)

        ---Arguments---
        X:              independent (predictor) variable
        Y:              dependent (response) variable
        alpha:          tuning parameter
        n_PCA:          number of principal components to retain
        kernel_type:    kernel function, may be either type str or function,
                        defaults to a linear kernel

        ---Returns---
        Xp:             X values projected into the latent (PCA-like) space
        Yp:             predicted Y values
        Xr:             Reconstructed X values from the latent (PCA-like) space
        Lx:             KPCA loss
        Ly:             KR loss

        ---References---
        1.  S. de Jong, H. A. L. Kiers, 'Principal Covariates
            Regression: Part I. Theory', Chemometrics and Intelligent
            Laboratory Systems 14(1): 155-164, 1992
        2.  M. Vervolet, H. A. L. Kiers, W. Noortgate, E. Ceulemans,
            'PCovR: An R Package for Principal Covariates Regression',
            Journal of Statistical Software 65(1):1-14, 2015
    """

    def __init__(self, alpha=0.0, n_PCA=2,
                 kernel_type='linear', regularization=1e-12):

        self.alpha = alpha
        self.n_PCA = n_PCA
        self.regularization = regularization

        self.PKT = None
        self.PTK = None
        self.PTY = None
        self.X = None  # To save the X which was used to train the model

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

    def fit(self, X, Y, K=None):
        """
            Fits the KPCovR to the training inputs and outputs

            ---Arguments---
            X:              independent (predictor) variable
            Y:              dependent (response) variable

            ---Returns---
            W:              Weights projecting X into latent PCA-space
            T:              Projection of X in latent PCA-space
            P:              Projector from latent PCA-space to Y-space
            Px:             Projector from latent PCA-space to X-space
        """

        if(K is None):
            K = self.kernel(X, X)
            K = center_kernel(K)
        self.K = K

        if(X is not None):
            self.X = X
        else:
            print("No input data supplied during fitting." +
                  "\nTransformations and statistics only" +
                  "available with pre-computed kernels.")

        # Compute maximum eigenvalue of kernel matrix

        krr = KRR(regularization=self.regularization, kernel_type=self.kernel)
        krr.fit(X=X, Y=Y, K=K)
        Yhat = krr.transform(X=X, K=K).reshape(-1, Y.shape[-1])

        K_pca = K/(np.trace(K)/X.shape[0])
        K_lr = np.matmul(Yhat, Yhat.T)

        Kt = (self.alpha*K_pca) + (1.0-self.alpha)*K_lr
        self.Kt = Kt

        self.v, self.U = sorted_eig(
            Kt, thresh=self.regularization, n=self.n_PCA)

        P_krr = np.linalg.solve(K+np.eye(len(K))*self.regularization, Yhat)
        P_krr = np.matmul(P_krr, Yhat.T)

        P_kpca = np.eye(K.shape[0])/(np.trace(K)/K.shape[0])

        P = (self.alpha*P_kpca) + (1.0-self.alpha)*P_krr

        v_inv = np.diagflat(eig_inv(self.v[:self.n_PCA]))

        self.PKT = np.matmul(P, np.matmul(self.U[:, :self.n_PCA],
                                          np.sqrt(v_inv)))
        self.T = np.matmul(K, self.PKT)
        self.PTK = np.matmul(v_inv, np.matmul(self.T.T, K))
        self.PTX = np.matmul(v_inv, np.matmul(self.T.T, X))
        self.PTY = np.matmul(v_inv, np.matmul(self.T.T, Y))

    def transform(self, X=None, K=None):
        """
            Transforms a set of inputs using the computed KPCovR

            ---Arguments---
            X:              independent (predictor) variable

            ---Returns---
            Xp:             X values projected into the latent (PCA-like) space
            Yp:             predicted Y values

        """
        if self.PKT is None:
            print("Error: must fit the PCovR model before transforming")
        elif(X is None and K is None):
            print("Either the kernel or input data must be specified.")
        else:

            if(K is None and self.X is not None):
                K = self.kernel(X, self.X)
                K = center_kernel(K, reference=self.K)
            elif(K is None):
                print("This functionality is not available.")
                return

            T = np.matmul(K, self.PKT)
            Yp = np.matmul(T, self.PTY)
            Xr = np.matmul(T, self.PTX)

            if(T.shape[1] == 1):
                T = np.array((T.reshape(-1), np.zeros(T.shape[0]))).T
            return T, Yp, Xr

    def loss(self, X=None, Y=None, K=None):
        """
            Computes the loss values for KPCovR on the given predictor and
            response variables

            ---Arguments---
            X:              independent (predictor) variable
            Y:              dependent (response) variable

            ---Returns---
            Lx:             KPCA loss
            Ly:             KR loss

        """

        if(K is None and self.X is not None):
            K = self.kernel(X, self.X)

            K = center_kernel(K, reference=self.K)
        else:
            raise ValueError(
                "Must provide a kernel or a feature vector, in which case the train features should be available in the class")

        Tp, Yp, Xr = self.transform(X=X, K=K)
        Kapprox = np.matmul(Tp, self.PTK)

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

        Tp, Yp, Xr = self.transform(X=X, K=K)

        Kapprox = np.matmul(Tp, self.PTK)

        return get_stats(x=X, y=Y, yp=Yp, t=Tp, xr=Xr,
                         k=K, kapprox=Kapprox)


class SparseKPCovR:

    """
        Performs KPCovR, as described in Helfrecht (2020)
        which combines Kernel Principal Components Analysis (KPCA)
        and Kernel Ridge Regression (KRR), on a sparse active set
        of variables

        ---Arguments---
        X:              independent (predictor) variable
        Y:              dependent (response) variable
        alpha:          tuning parameter
        n_PCA:          number of principal components to retain
        kernel_type:    kernel function, may be either type str or function,
                        defaults to a linear kernel

        ---Returns---
        Xp:             X values projected into the latent (PCA-like) space
        Yp:             predicted Y values
        Lx:             KPCA loss
        Ly:             KR loss

        ---References---
        1.  S. de Jong, H. A. L. Kiers, 'Principal Covariates
            Regression: Part I. Theory', Chemometrics and Intelligent
            Laboratory Systems 14(1): 155-164, 1992
        2.  M. Vervolet, H. A. L. Kiers, W. Noortgate, E. Ceulemans,
            'PCovR: An R Package for Principal Covariates Regression',
            Journal of Statistical Software 65(1):1-14, 2015
    """

    def __init__(self, alpha, n_PCA, n_active=100, regularization=1e-6, kernel_type="linear"):

        self.alpha = alpha
        self.n_active = n_active
        self.n_PCA = n_PCA
        self.regularization = regularization

        self.PKT = None
        self.PTY = None
        self.PTX = None
        self.X = None  # To save the X which was used to train the model

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

    def fit(self, X, Y, X_sparse=None, Kmm=None, Knm=None):
        """
            Fits the KPCovR to the training inputs and outputs

            ---Arguments---
            X:              independent (predictor) variable
            Y:              dependent (response) variable

            ---Returns---
            W:              Weights projecting X into latent PCA-space
            T:              Projection of X in latent PCA-space
            P:              Projector from latent PCA-space to Y-space
            Px:             Projector from latent PCA-space to X-space
        """

        if(X_sparse is None):
            fps_idxs, _ = FPS(X, self.n_active)
            self.X_sparse = X[fps_idxs, :]
        else:
            self.X_sparse = X_sparse

        if(Kmm is None):
            Kmm = self.kernel(self.X_sparse, self.X_sparse)
            Kmm = center_kernel(Kmm)

        self.Kmm = Kmm

        if(Knm is None):
            Knm = self.kernel(X, self.X_sparse)
            Knm = center_kernel(Knm, reference=self.Kmm)

        self.Knm = Knm

        vmm, Umm = sorted_eig(
            self.Kmm, thresh=self.regularization, n=self.n_active)
        vmm_inv = eig_inv(vmm[:self.n_active-1])

        self.barKM = np.mean(self.Knm, axis=0)

        phi_active = np.matmul(self.Knm, Umm[:, :self.n_active-1])
        phi_active = np.matmul(phi_active, np.diagflat(np.sqrt(vmm_inv)))
        barPhi = np.mean(phi_active, axis=0)
        phi_active -= barPhi

        C = np.dot(phi_active.T, phi_active)

        v_C, U_C = sorted_eig(C, thresh=0)
        U_C = U_C[:, v_C > 0]
        v_C = v_C[v_C > 0]
        v_C_inv = eig_inv(v_C)

        Csqrt = np.matmul(np.matmul(U_C, np.diagflat(np.sqrt(v_C))), U_C.T)
        iCsqrt = np.matmul(
            np.matmul(U_C, np.diagflat(np.sqrt(v_C_inv))), U_C.T)

        C_pca = C / (np.trace(C)/C.shape[0])

        C_lr = np.linalg.pinv(C + self.regularization*np.eye(C.shape[0]))
        C_lr = np.matmul(phi_active, C_lr)
        C_lr = np.matmul(phi_active.T, C_lr)
        C_lr = np.matmul(iCsqrt, C_lr)
        C_lr = np.matmul(C_lr, phi_active.T)
        C_lr = np.matmul(C_lr, Y.reshape(-1, Y.shape[-1]))
        C_lr = np.matmul(C_lr, C_lr.T)

        Ct = self.alpha*C_pca + (1-self.alpha)*C_lr

        v_Ct, U_Ct = sorted_eig(Ct, thresh=0)

        PPT = np.matmul(iCsqrt, U_Ct[:, :self.n_PCA])
        PPT = np.matmul(PPT, np.diag(np.sqrt(v_Ct[:self.n_PCA])))

        PKT = np.matmul(Umm[:, :self.n_active-1],
                        np.diagflat(np.sqrt(vmm_inv)))
        self.PKT = np.matmul(PKT, PPT)
        self.barT = np.matmul(barPhi, PPT)

        T = np.matmul(self.Knm, self.PKT) - self.barT

        PT = np.matmul(T.T, T)
        PT = np.linalg.pinv(PT)
        PT = np.matmul(PT, T.T)
        self.PTY = np.matmul(PT, Y)
        self.PTX = np.matmul(PT, X)

        self.Ct = Ct
        self.C = C
        self.phi_active = phi_active
        self.vmm = vmm
        self.Umm = Umm
        self.PT = PT
        self.v_C = v_C
        self.U_C = U_C
        self.v_Ct = v_Ct
        self.U_Ct = U_Ct
        #
        # K = np.matmul(np.matmul(self.Knm, np.linalg.pinv(self.Kmm)), self.Knm.T)
        #
        # # Compute eigendecomposition of kernel
        # vmm, Umm = sorted_eig(self.Kmm, thresh=self.regularization, n=self.n_active)
        # vmm_inv = np.linalg.pinv([vmm[:self.n_active-1]]).T[0]
        #
        # self.barKM = np.mean(Knm, axis=0)
        # phi_active = Knm-self.barKM
        # phi_active = np.matmul(phi_active, Umm[:,:self.n_active-1])
        # phi_active = np.matmul(phi_active,
        #                        np.diagflat(np.sqrt(vmm_inv)))
        #
        # C = np.matmul(phi_active.T,phi_active)
        #
        # v_C, U_C = sorted_eig(C, thresh=self.regularization, n=self.n_active)
        # v_C_inv = np.linalg.pinv([v_C]).T[0]
        #
        # Csqrt = np.matmul(np.matmul(U_C, np.diagflat(np.sqrt(v_C))), U_C.T)
        # iCsqrt = np.matmul(np.matmul(U_C, np.diagflat(np.sqrt(v_C_inv))), U_C.T)
        #
        # C_pca = C / (np.trace(C)/C.shape[0])
        #
        # C_lr = np.linalg.pinv(C + self.regularization*np.eye(C.shape[0]))
        # C_lr = np.matmul(Csqrt, C_lr)
        # C_lr = np.matmul(C_lr, phi_active.T)
        # C_lr = np.matmul(C_lr, Y.reshape(-1,1))
        # C_lr = np.matmul(C_lr, C_lr.T)
        #
        # Ct = self.alpha*C_pca + (1.0-self.alpha)*C_lr
        #
        # v_Ct, U_Ct = sorted_eig(Ct, thresh=self.regularization, n=self.n_active)
        #
        # PPT = np.matmul(iCsqrt, U_Ct[:, :self.n_PCA])
        # PPT = np.matmul(PPT, np.diag(np.sqrt(v_Ct[:self.n_PCA])))
        #
        # PKT = np.matmul(Umm[:,:self.n_active-1], np.diagflat(np.sqrt(vmm_inv)))
        # self.PKT = np.matmul(PKT[:, :self.n_active-1], PPT)
        #
        # self.T = np.matmul(self.Knm, self.PKT)
        #
        # # PT = np.linalg.pinv(np.matmul(self.T.T, self.T) +
        # #                     self.regularization*np.eye(self.T.shape[1]))
        # # PT = np.matmul(PT, self.T.T)
        #
        # PT = np.matmul(np.diagflat(np.sqrt(v_Ct[:self.n_PCA])),
        #                U_Ct[:, :self.n_PCA].T)
        # PT = np.matmul(PT, iCsqrt)
        # PT = np.matmul(PT, phi_active.T)
        # print(PT.shape)
        #
        # print(Y.shape, X.shape, phi_active.shape, self.T.shape)
        # self.PTY = np.matmul(PT, Y)
        # self.PTX = np.matmul(PT, X)

        # self.K = K
        # self.C = C
        # self.C_lr = C_lr
        # self.C_pca = C_pca
        # self.Ct = Ct
        # self.phi_active = phi_active
        # self.Cs = [Csqrt, iCsqrt]

    def transform(self, X, Knm=None):
        """
            Transforms a set of inputs using the computed Sparse KPCovR

            ---Arguments---
            X:              independent (predictor) variable

            ---Returns---
            Xp:             X values projected into the latent (PCA-like) space
            Yp:             predicted Y values

        """

        if self.PKT is None:
            print("Error: must fit the PCovR model before transforming")
        else:

            if(Knm is None):
                Knm = self.kernel(X, self.X_sparse)
                Knm = center_kernel(Knm, reference=self.Kmm)

            T = np.matmul(Knm, self.PKT) - self.barT
            Yp = np.matmul(T, self.PTY)
            Xr = np.matmul(T, self.PTX)
            return T, Yp, Xr

    def loss(self, X, Y, Knm=None):
        """
            Computes the loss values for KPCovR on the given predictor and
            response variables

            ---Arguments---
            X:              independent (predictor) variable
            Y:              dependent (response) variable

            ---Returns---
            Lx:             KPCA loss
            Ly:             KR loss

        """

        if(Knm is None):
            Knm = self.kernel(X, self.X_sparse)
            Knm = center_kernel(Knm, reference=self.Kmm)

        T, Yp, Xr = self.transform(X, Knm=Knm)

        Lkpca = np.linalg.norm(Xr - X)**2/np.linalg.norm(X)**2
        Lkrr = np.linalg.norm(Y - Yp)**2/np.linalg.norm(Y)**2

        return Lkpca, Lkrr

    def statistics(self, X, Y, Knm=None):
        """
            Computes the loss values and reconstruction errors for
            KPCovR for the given predictor and response variables

            ---Arguments---
            X:              independent (predictor) variable
            Y:              dependent (response) variable

            ---Returns---
            dictionary of available statistics

        """

        if(Knm is None):
            Knm = self.kernel(X, self.X_sparse)
            Knm = center_kernel(Knm, reference=self.Kmm)

        T, Yp, Xr = self.transform(X, Knm=Knm)

        return get_stats(x=X, y=Y, yp=Yp, t=T, xr=Xr)