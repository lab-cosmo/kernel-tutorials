import numpy as np
from scipy.sparse.linalg import svds as svd
from .general import sorted_eig


def approx_A(A, col_idx, row_idx=None):

    A_c = A[:, col_idx]
    S_c = np.linalg.pinv(A_c)

    A_r = A[row_idx]

    if(row_idx is not None):
        S_r = np.linalg.pinv(A_r)
        S = np.matmul(S_c, np.matmul(A, S_r))
    else:
        S = S_c

    return np.matmul(A_c, np.matmul(S, A_r))


def compute_P(A_c, S, A_r, thresh=1e-12):

    SA = np.matmul(S, A_r)
    SA = np.matmul(SA, SA.T)

    v_SA, U_SA = np.linalg.eigh(SA)
    v_SA[v_SA < thresh] = 0

    return np.matmul(U_SA, np.diagflat(np.sqrt(v_SA)))


def get_Ct(X, Y, alpha=0.5, regularization=1e-6):
    """ Creates the PCovR modified covariance"""
    
    cov = np.matmul(X.T, X)
    v_C, U_C = sorted_eig(cov, thresh=regularization)
    U_C = U_C[:,v_C>regularization]
    v_C = v_C[v_C>regularization]
    
    Csqrt = np.matmul(np.matmul(U_C, np.diag(np.sqrt(v_C))), U_C.T)
    Cinv = np.matmul(np.matmul(U_C, np.diag(1.0/(v_C))), U_C.T)
    
    C_lr = np.matmul(Cinv, np.matmul(X.T,Y))
    C_lr = np.matmul(Csqrt, C_lr)
    C_lr = np.matmul(C_lr, C_lr.T)
    
    C_pca = cov
    
    C =  alpha*C_pca +  (1.0-alpha)*C_lr

    return C


def svd_select(A, n, k=1, idxs=None, **kwargs):
    """
        Doc-string needed
    """

    if(idxs is None):
        idxs = []  # indexA is initially empty.

    Acopy = A.copy()

    for nn in range(n):
        if(nn >= len(idxs)):  # len(idxs)<=nn-1):
            (S, v, D) = svd(Acopy, k)
            pi = (D[:k]**2.0).sum(axis=0)
            pi[idxs] = 0  # eliminate possibility of selecting same column twice
            i = pi.argmax()
            idxs.append(i)

        v = Acopy[:, idxs[nn]] / \
            np.sqrt(np.matmul(Acopy[:, idxs[nn]], Acopy[:, idxs[nn]]))

        for i in range(Acopy.shape[1]):
            Acopy[:, i] -= v * np.dot(v, Acopy[:, i])
        print(len(idxs), end='\r')

    return list(idxs)


def pcovr_select(A, n, Y, alpha, k=1, idxs=None, **kwargs):
    """
        Doc-string needed
    """

    Acopy = A.copy()
    Ycopy = Y.copy()

    if(idxs is None):
        idxs = []  # indexA is initially empty.

    for nn in range(n):
        if(nn >= len(idxs)):  # len(idxs)<=nn-1):
            try:
                Ct = get_Ct(Acopy, Ycopy, alpha=alpha)
            except:
                print("Only {} features possible".format(nn))
                return list(idxs)
            v_Ct, U_Ct = sorted_eig(Ct, n=k)

            pi = (np.real(U_Ct)[:, :k]**2.0).sum(axis=1)
            pi[idxs] = 0  # eliminate possibility of selecting same column twice
            j = pi.argmax()
            idxs.append(j)

        v = Acopy[:, idxs[nn]] / \
            np.sqrt(np.matmul(Acopy[:, idxs[nn]], Acopy[:, idxs[nn]]))

        for i in range(Acopy.shape[1]):
            Acopy[:, i] -= v * np.dot(v, Acopy[:, i])
        update(Ycopy, Acopy[:, idxs])

    return list(idxs)


def update(Y_copy, X_c):
    """
        Doc-string needed
    """
    v = np.linalg.pinv(np.matmul(X_c.T, X_c))
    v = np.matmul(X_c, v)
    v = np.matmul(v, X_c.T)

    Y_copy -= np.matmul(v, Y_copy)


selections = dict(svd=svd_select, pcovr=pcovr_select)


class CUR:
    def __init__(self, matrix,
                 precompute=None,
                 feature_select=False,
                 pi_function='svd',
                 tol=1e-4,
                 params={}
                 ):
        """
            Doc-string needed
        """
        self.A = matrix
        self.symmetric = self.A.shape == self.A.T.shape and \
            np.all(np.abs(self.A-self.A.T)) < tol
        self.fs = feature_select
        self.select = selections[pi_function]
        self.params = params

        if(pi_function == 'pcovr'):
            try:
                assert all([x in params for x in ['Y', 'alpha']])
            except:
                print(
                    "For column selection with PCovR, `Y` and `alpha` must be entries in `params`")

        self.idx_c, self.idx_r = None, None
        if(precompute is not None):
            if(isinstance(precompute, int)):
                self.idx_c, self.idx_r = self.compute_idx(
                    precompute, precompute)
            else:
                self.idx_c, self.idx_r = self.compute_idx(*precompute)

    def compute_idx(self, n_c, n_r):
        """
            Doc-string needed
        """
        idx_c = self.select(self.A, n_c, idxs=self.idx_c, **self.params)
        if(self.fs):
            idx_r = np.asarray(range(self.A.shape[1]))
        elif(not self.symmetric):
            idx_r = self.select(self.A.T, n_r, idxs=self.idx_r, **self.params)
        else:
            idx_r = idx_c
        return idx_c, idx_r

    def compute(self, n_c, n_r=None):
        """
            Doc-string needed
        """
        if(self.fs):
            n_r = self.A.shape[1]
        elif(self.symmetric and n_r == None):
            n_r = n_c
        elif(n_r == None):
            print("You must specify a n_r for non-symmetric matrices.")

        if(self.idx_c is None or (not self.fs and self.idx_r is None)):
            idx_c, idx_r = self.compute_idx(n_c, n_r)
            self.idx_c, self.idx_r = idx_c, idx_r
        elif(len(self.idx_c) < n_c or len(self.idx_r) < n_r):
            idx_c, idx_r = self.compute_idx(n_c, n_r)
            self.idx_c, self.idx_r = idx_c, idx_r
        else:
            idx_c = self.idx_c[:n_c]
            idx_r = self.idx_r[:n_r]

        # The CUR Algorithm
        A_c = self.A[:, idx_c]
        if(self.symmetric and not self.fs):
            A_r = A_c.T
        elif(self.fs):
            A_r = self.A.copy()
        else:
            A_r = self.A[idx_r, :]
        # Compute S.
        S = np.matmul(np.matmul(np.linalg.pinv(A_c), self.A),
                      np.linalg.pinv(A_r))
        return A_c, S, A_r

    def compute_P(self, n_c):
        """
            Doc-string needed
        """

        A_c, S, A_r = self.compute(n_c)
        self.P = compute_P(A_c, S, A_r)

    def loss(self, n_c, n_r=None):
        """
            Doc-string needed
        """
        A_c, S, A_r = self.compute(n_c, n_r)
        return np.linalg.norm(self.A - np.matmul(A_c, np.matmul(S, A_r)))/np.linalg.norm(self.A)
