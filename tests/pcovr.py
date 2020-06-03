import numpy as np
from sklearn import exceptions
from sklearn.utils.validation import check_X_y

from utilities.sklearn_covr.pcovr import PCovR

def run_tests():
    data = np.load('./tests/CSD-test.npz')
    X = data["X"]
    Y = data["Y"]

    # Basic Test of PCovR Errors
    lr_errors = np.nan * np.zeros(21)
    pca_errors = np.nan * np.zeros(21)
    for i, alpha in enumerate(np.linspace(0,1,21)):
        pcovr = PCovR(alpha = alpha,
                      n_components=2,
                      regularization=1e-6,
                      tol=1e-12)
        pcovr.fit(X, Y)

        T = pcovr.transform(X)
        Xr = pcovr.inverse_transform(T)
        Yp = pcovr.predict(X)
        lr_errors[i] = np.linalg.norm(Y-Yp)**2.0 / np.linalg.norm(Y)**2.0
        pca_errors[i] = np.linalg.norm(X-Xr)**2.0 / np.linalg.norm(X)**2.0

        assert not np.isnan(lr_errors[i]) and not np.isnan(pca_errors[i])

    assert all(lr_errors[i] <= lr_errors[i+1] and pca_errors[i] >= pca_errors[i+1]for i in range(len(lr_errors)-1))

    # Test of PCovR Fitting
    pcovr = PCovR(alpha = 0.5,
                  n_components=2,
                  regularization=1e-6,
                  tol=1e-12)

    try:
        T = pcovr.transform(X)
    except exceptions.NotFittedError:
        print("Premature transformation failed in expected manner")

    # Test of T shape
    pcovr.fit(X, Y)
    T = pcovr.transform(X)

    assert check_X_y(X, T, multi_output=True)



if __name__ == "__main__":
    run_tests()
    print("Everything passed")
