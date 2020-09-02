import numpy as np
from utilities.scalers import NormalizeScaler

# The code is based on methods presented in the paper TODO(currently in review process, will be added as soon as published)

# default cross validation parameters used for the paper
DEFAULT_CROSS_VALIDATION_KWARGS = {"train_test_split": True, "train_ratio": 0.5, "seed": 0x5F3759DF, "nb_folds": 2}

def standardize_features(features, train_idx=None):
    """
    Standardizes the features

    Parameters:
    ----------
    features : array_like
        features, dimensions: samples x features
    train_idx : array_like
        the training indices are only used for standardization

    Returns:
    --------
    standardized_features : array_like
        standardized features
    """
    if train_idx is None:
        return NormalizeScaler().fit(features).transform(features)
    return NormalizeScaler().fit(features[train_idx]).transform(features)

def compute_reconstruction_matrix(features, features_dash, nb_folds):
    """
    Computes the reconstruction matrix P_{FF'} with a leave-one-out-cross-validation

    Parameters:
    ----------
    features : array_like
        features X_F as in the paper, dimensions: samples x features
    features_dash : array_like
        features X_{F'} as in the paper, dimensions samples x features
    nb_folds : int
        number folds for the cross validation

    Returns:
    --------
    reconstruction_matrix : array_like
        reconstruction matrix P_{FF'} = argmin_{P} | X_{F'} - (X_{F})P |
    """

    fold_size = len(features)//nb_folds
    features_folds = [features[i*fold_size:(i+1)*fold_size] for i in range(nb_folds-1)]
    features_folds.append(features[(nb_folds-1)*fold_size:])
    features_dash_folds = [features_dash[i*fold_size:(i+1)*fold_size] for i in range(nb_folds-1)]
    features_dash_folds.append(features_dash[(nb_folds-1)*fold_size:])

    def fold_test_error(regularizer):
        test_error = np.zeros(nb_folds)
        for i in range(nb_folds):
            W = np.linalg.lstsq(features_folds[i], features_dash_folds[i], rcond=regularizer)[0]
            test_fold_idx = list(range(nb_folds))
            test_fold_idx.remove(i)
            features_test = np.concatenate([features_folds[j] for j in test_fold_idx], axis=0)
            features_dash_test = np.concatenate([features_dash_folds[j] for j in test_fold_idx], axis=0)
            test_error[i] = np.linalg.norm(
                features_test.dot(W) - features_dash_test
            ) / np.sqrt(features_test.shape[0])
        return np.mean(test_error)
    # be aware 1 means no regularization
    regularizers = np.hstack( (np.geomspace(1e-9, 1e-1, 9), [0.5, 0.9, 1]) )
    loss = [fold_test_error(reg) for reg in regularizers]
    min_idx = np.argmin(loss)
    return np.linalg.lstsq(features, features_dash, rcond=regularizers[min_idx])[0]

def generate_train_test_idx(nb_samples, train_test_split, train_ratio, seed):
    idx = np.arange(nb_samples)
    if seed is not None:
        np.random.seed(seed)
        np.random.shuffle(idx)
    if not(train_test_split):
        return idx, idx 
    split_id = int(len(idx) * train_ratio)
    return idx[:split_id], idx[split_id:]

def _compute_gfre(features, features_dash, reconstruction_matrix):
    """
    Computes GFRE

    Parameters:
    ----------
    features : array_like
        features X_{F} of space F as in the paper, dimension: samples x features
    features_dash : array_like
        feature_dash X_{F'} of space F as in the paper, dimension: samples x features
    reconstruction_matrix : array_like
        defined by P_{FF'} = argmin_{P} || X_{F'} - (X_F)P ||

    Returns:
    --------
    gfre: double
        GFRE(X_F, X_{F'})
    """
    return np.linalg.norm(
            features.dot(reconstruction_matrix) - features_dash
        ) / np.sqrt(features.shape[0])

def _compute_pointwise_gfre(features, features_dash, reconstruction_matrix):
    """
    Computes pointwise GFRE

    Parameters:
    ----------
    features : array_like
        features X_{F} of space F as in the paper, dimension: samples x features
    features_dash : array_like
        feature_dash X_{F'} of space F as in the paper, dimension: samples x features
    reconstruction_matrix : array_like
        defined by P_{FF'} = argmin_{P} || X_{F'} - (X_F)P ||

    Returns:
    --------
    poinwise_gfre: array_like
        pointwise GFRE(X_F, X_{F'})
    """
    return np.linalg.norm(features.dot(reconstruction_matrix) - features_dash, axis=1)

def _compute_gfrd(features, features_dash, reconstruction_matrix):
    """
    Computes the GFRD(X_F, X_{F'}). Features should be standardized beforehand.

    Parameters:
    ----------
    features : array_like
        features X_F as in the paper, dimensions: samples x features
    features_dash : array_like
        features X_{F'} as in the paper, dimensions samples x features
    reconstruction_matrix : array_like
        defined by P_{FF'} = argmin_{P} || X_{F'} - (X_F)P ||

    Returns:
    --------
    gfrd : double
        GFRD(X_F, X_{F'})
    """
    # P_{FF'} = U Σ V^T
    U, S, _ = np.linalg.svd(reconstruction_matrix)

    # \tilde{X}_F = X_F U
    tilde_features = features.dot(U)[:, :len(S)]
    # \tilde{X}_{F'} = X_F P_{FF'} V = X_F U Σ
    tilde_features_dash = tilde_features.dot(np.diag(S))

    # Solution for the Procrustes problem (see https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem)
    tilde_U, tilde_S, tilde_VT = np.linalg.svd(
        tilde_features.T.dot(tilde_features_dash)
    )
    Q_FF_dash = tilde_U.dot(tilde_VT)

    n_test = features.shape[0]
    # GFRD
    return np.linalg.norm(tilde_features.dot(Q_FF_dash ) - tilde_features_dash) / np.sqrt(n_test)

def _compute_gfrm(features, features_dash, compute_gfrm_func, cross_validation_kwargs):
    """
    Computes a GFRM (global feature space reconstruction measure) including GFRE, pointwise GFRE and GFRD.
    Wrapper function for the cross validation steps required for all measures.

    Parameters:
    ----------
    features : array_like
        features X_{F} of space F as in the paper, dimension: samples x features
    features_dash : array_like
        feature_dash X_{F'} of space F as in the paper, dimension: samples x features
    compute_gfrm_func : function(array_like features, array_like features_dash, array_like reconstruction matrix) -> double gfrm
        functions as `_compute_gfre`, `_compute_pointwise_gfre` or `_compute_gfrd`
    cross_validation_kwargs : dict
        contains keys `train_test_split`, `train_ratio`, `seed` and `nb_folds` 

    Returns:
    --------
    gfrm : double (in case of GFRE and GFRD) / array_like (in case of pointwise GFRE)
        GFRE, pointwise GFRE or GFRD
    """
    # parameters
    train_test_split, train_ratio, seed, nb_folds = cross_validation_kwargs['train_test_split'], cross_validation_kwargs['train_ratio'], cross_validation_kwargs['seed'], cross_validation_kwargs['nb_folds']

    # generate train test idx
    assert(features.shape[0] == features_dash.shape[0])
    nb_samples = features.shape[0]
    train_idx, test_idx = generate_train_test_idx(nb_samples, train_test_split, train_ratio, seed)

    #  standardize
    features = standardize_features(features, train_idx)
    features_dash = standardize_features(features_dash, train_idx)

    reconstruction_matrix = compute_reconstruction_matrix(features[train_idx], features_dash[train_idx], nb_folds)
    return compute_gfrm_func(features[test_idx], features_dash[test_idx], reconstruction_matrix)

def compute_gfre(features, features_dash, cross_validation_kwargs=DEFAULT_CROSS_VALIDATION_KWARGS):
    return _compute_gfrm(features, features_dash, _compute_gfre, cross_validation_kwargs)

def compute_pointwise_gfre(features, features_dash, cross_validation_kwargs=DEFAULT_CROSS_VALIDATION_KWARGS):
    return _compute_gfrm(features, features_dash, _compute_pointwise_gfre, cross_validation_kwargs)

def compute_gfrd(features, features_dash, cross_validation_kwargs=DEFAULT_CROSS_VALIDATION_KWARGS):
    return _compute_gfrm(features, features_dash, _compute_gfrd, cross_validation_kwargs)

def compute_pointwise_lfre(features, features_dash, nb_local_envs, cross_validation_kwargs=DEFAULT_CROSS_VALIDATION_KWARGS):
    """
    Computes the pointwise LFRE

    Parameters:
    ----------
    features : array_like
        features X_{F} of space F as in the paper, dimension: samples x features
    features_dash : array_like
        feature_dash X_{F'} of space F as in the paper, dimension: samples x features
    nb_local_envs : int
        positive integer stating the number of neighbours k used for the LFRE
    cross_validation_kwargs : dict
        contains keys `train_test_split`, `train_ratio`, `seed` and `nb_folds` 

    Returns:
    --------
    pointwise_lfre : array_like
        pointwise LFRE
    """

    train_test_split, train_ratio, seed, nb_folds = cross_validation_kwargs['train_test_split'], cross_validation_kwargs['train_ratio'], cross_validation_kwargs['seed'], cross_validation_kwargs['nb_folds']

    # generate train test idx
    assert(features.shape[0] == features_dash.shape[0])
    nb_samples = features.shape[0]
    train_idx, test_idx = generate_train_test_idx(nb_samples, train_test_split, train_ratio, seed)
    features = standardize_features(features, train_idx)
    features_dash = standardize_features(features_dash, train_idx)
    features_train, features_dash_train, features_test, features_dash_test = features[train_idx], features_dash[train_idx], features, features_dash

    n_test = features_test.shape[0]
    squared_dist = np.sum(features_train**2, axis=1) + np.sum(features_test**2, axis=1)[:,np.newaxis] - 2 * features_test.dot(features_train.T)
    pointwise_lfre = np.zeros(n_test)
    for i in range(n_test):
        # LLE-inspired LFRE
        local_env_idx = np.argsort(squared_dist[i])[:nb_local_envs]
        local_features_train = features_train[local_env_idx]
        local_features_train_mean = np.mean(features_train[local_env_idx], axis=0)
        local_features_dash_train = features_dash_train[local_env_idx]
        local_features_dash_train_mean = np.mean(features_dash_train[local_env_idx], axis=0)
        # standardize
        reconstruction_matrix = compute_reconstruction_matrix(
            local_features_train - local_features_train_mean, local_features_dash_train - local_features_dash_train_mean, nb_folds
        )
        # \|x_i' - \tilde{x}_i' \|
        pointwise_lfre[i] = np.linalg.norm(
            (features_test[i,:][np.newaxis,:] - local_features_train_mean).dot(reconstruction_matrix) + local_features_dash_train_mean
            - features_dash_test[i,:][np.newaxis,:]
        )
    return pointwise_lfre

def compute_lfre(features, features_dash, nb_local_envs, cross_validation_kwargs=DEFAULT_CROSS_VALIDATION_KWARGS):
    """
    Computes the LFRE

    Parameters:
    ----------
    features : array_like
        features X_{F} of space F as in the paper, dimension: samples x features
    features_dash : array_like
        feature_dash X_{F'} of space F as in the paper, dimension: samples x features
    nb_local_envs : int
        positive integer stating the number of neighbours k used for the LFRE
    cross_validation_kwargs : dict
        contains keys `train_test_split`, `train_ratio`, `seed` and `nb_folds` 

    Returns:
    --------
    lfre : double
        LFRE
    """
    return np.linalg.norm(compute_pointwise_lfre(features, features_dash, nb_local_envs, cross_validation_kwargs=DEFAULT_CROSS_VALIDATION_KWARGS))/np.sqrt(features.shape[0])
