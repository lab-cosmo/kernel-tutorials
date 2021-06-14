import numpy as np
from sklearn.metrics import r2_score as calc_R2
from sklearn.model_selection import train_test_split

from skcosmo.feature_selection import FPS
from skcosmo.preprocessing import KernelNormalizer
from skcosmo.preprocessing import StandardFlexibleScaler

from .kernels import gaussian_kernel


def eig_inv(v, rcond=1e-14):
    """ Inverse of a list (typically of eigenvalues) with thresholding à la pinv """
    thresh = v.max() * rcond
    return np.array([(1 / vv if vv > thresh else 0.0) for vv in v])


def quick_inverse(mat):
    """ Does a quick(er) matrix inverse """
    U_mat, v_mat = sorted_eig(mat, thresh=0)

    return np.matmul(np.matmul(U_mat, np.diagflat(eig_inv(v_mat))), U_mat.T)


def sorted_eig(mat, thresh=0.0, n=None, sps=True):
    """
    Returns n eigenvalues and vectors sorted
    from largest to smallest eigenvalue
    """

    if sps:
        from scipy.sparse.linalg import eigs as speig

        if n is None:
            k = mat.shape[0] - 1
        else:
            k = n
        val, vec = speig(mat, k=k, tol=thresh)
        val = np.real(val)
        vec = np.real(vec)

        idx = sorted(range(len(val)), key=lambda i: -val[i])

        val = val[idx]
        vec = vec[:, idx]

    else:
        val, vec = np.linalg.eigh(mat)
        val = np.flip(val, axis=0)
        vec = np.flip(vec, axis=1)

    if n is not None and len(val[val < thresh]) < n:
        vec[:, val < thresh] = 0
        val[val < thresh] = 0
    else:
        vec = vec[:, val >= thresh]
        val = val[val >= thresh]

    return val[:n], vec[:, :n]


def get_stats(y=None, yp=None, x=None, t=None, xr=None, k=None, kapprox=None, **kwargs):
    """Returns available statistics given provided data"""
    stats = {}
    if y is not None and yp is not None:
        stats["Coefficient of Determination<br>($R^2$)"] = calc_R2(y, yp)
        stats[r"$\ell_{regr}$"] = np.linalg.norm(y - yp) / y.shape[0]
    if x is not None and t is not None:
        stats[r"Dataset Variance<br>$\sigma_X^2$"] = x.var(axis=0).sum()
        stats[r"Projection Variance<br>$\sigma_T^2$"] = t.var(axis=0).sum()
        error = x.var(axis=0).sum() - t.var(axis=0).sum()
        stats[r"Residual Variance<br>$\sigma_X^2 - \sigma_T^2$"] = error
    if x is not None and xr is not None:
        stats[r"$\ell_{proj}$"] = ((x - xr) ** 2).mean(axis=0).sum()
    if k is not None and kapprox is not None:
        error = np.linalg.norm(kapprox - k) ** 2 / np.linalg.norm(k) ** 2.0
        stats[r"$\ell_{gram}$"] = error

    # allow for manual input of statistics (for kpcovr error)
    for k in kwargs:
        stats[k] = kwargs[k]

    return stats


def load_variables(cache_name="../datasets/precomputed.npz", **kwargs):
    try:
        data = dict(np.load(cache_name, allow_pickle=True))
    except (OSError, ImportError):
        print("Returning default data set.")
        from skcosmo.datasets import load_csd_1000r

        X, y = load_csd_1000r(return_X_y=True)
        data = dict(X=X, Y=y, indices=np.array([]))
    return calculate_variables(**dict(data), **kwargs)


def calculate_variables(
    X,
    Y,
    indices,
    n_atoms=None,
    N=10,
    n_FPS=200,
    kernel_func=gaussian_kernel,
    i_train=None,
    i_test=None,
    n_train=None,
    K_train=None,
    K_test=None,
):
    """Loads necessary data for the tutorials"""

    print("Shape of Input Data is ", X.shape, ".")

    if n_FPS is not None and n_FPS < X.shape[1]:
        fps_idxs = FPS(n_to_select=n_FPS).fit(X).selected_idx_
        print("Taking a subsampling of ", n_FPS, "features")
        X = X[:, fps_idxs]

    if i_train is not None:
        print("Shape of testing data is: ", i_train.shape, ".")
    else:
        print("Splitting Data Set")
        if n_train is None:
            n_train = int(len(Y) / 2)

        i_test, i_train = train_test_split(np.arange(len(Y)), train_size=n_train)

    n_train = len(i_train)
    n_test = len(i_test)

    Y_train = Y[i_train]
    Y_test = Y[i_test]

    y_scaler = StandardFlexibleScaler(column_wise=True).fit(Y_train)

    # Center total dataset
    Y = y_scaler.transform(Y)

    # Center training data
    Y_train = y_scaler.transform(Y_train)

    # Center training data
    Y_test = y_scaler.transform(Y_test)

    if len(Y) == len(indices) and n_atoms is not None:
        print(
            "Computing training/testing sets from summed environment-centered soap vectors."
        )
        frame_starts = [sum(n_atoms[:i]) for i in range(len(n_atoms) + 1)]
        X_split = [
            X[frame_starts[i] : frame_starts[i + 1]] for i in range(len(indices))
        ]

        X = np.array([np.mean(xs, axis=0) for xs in X_split])
        X_train = X[i_train]
        X_test = X[i_test]

    else:
        X_split = X.copy()

        X_train = X[i_train]
        X_test = X[i_test]

    x_scaler = StandardFlexibleScaler(column_wise=False).fit(X_train)

    # Center total dataset
    X = x_scaler.transform(X)

    # Center training data
    X_train = x_scaler.transform(X_train)

    # Center training data
    X_test = x_scaler.transform(X_test)

    if K_train is not None and K_test is not None:
        print("Shape of kernel is: ", K_train.shape, ".")
    else:
        if len(Y) == len(indices):
            print(
                "Computing kernels from summing kernels of environment-centered soap vectors."
            )

            K_train = kernel_func(
                [X_split[i] for i in i_train], [X_split[i] for i in i_train]
            )
            K_test = kernel_func(
                [X_split[i] for i in i_test], [X_split[i] for i in i_train]
            )

        else:

            K_train = kernel_func(X_split[i_train], X_split[i_train])
            K_test = kernel_func(X_split[i_test], X_split[i_train])

    k_scaler = KernelNormalizer().fit(K_train)

    K_train = k_scaler.transform(K_train)
    K_test = k_scaler.transform(K_test)

    n_train = len(X_train)
    n_test = len(X_test)
    n_PC = 2

    return dict(
        X=X,
        Y=Y,
        X_split=X_split,
        X_center=x_scaler.mean_,
        Y_center=y_scaler.mean_,
        X_scale=x_scaler.scale_,
        Y_scale=y_scaler.scale_,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        K_train=K_train,
        K_test=K_test,
        i_train=i_train,
        i_test=i_test,
        n_PC=n_PC,
        n_train=n_train,
        n_test=n_test,
    )
