import numpy as np
from sklearn import exceptions
from sklearn.utils.validation import check_X_y

from utilities.sklearn_covr.kpcovr import KernelPCovR

def run_tests():
    data = np.load('./tests/CSD-test.npz')
    X = data["X"]
    Y = data["Y"]

    # Basic Test of PCovR Errors
    krr_errors = np.nan * np.zeros(21)
    kpca_errors = np.nan * np.zeros(21)
    for i, mixing in enumerate(np.linspace(0,1,21)):
        kpcovr = KernelPCovR(mixing = mixing, kernel='linear',
                      n_components=2,
                      # full_eig=False,
                      tol=1e-12,
                        krr_params = dict(alpha=1e-6,)
)
        kpcovr.fit(X, Y)

        T = kpcovr.transform(X)
        Xr = kpcovr.inverse_transform(T)
        Yp = kpcovr.predict(X)
        krr_errors[i] = np.linalg.norm(Y-Yp)**2.0 / np.linalg.norm(Y)**2.0
        kpca_errors[i] = np.linalg.norm(X-Xr)**2.0 / np.linalg.norm(X)**2.0

        assert not np.isnan(krr_errors[i]) and not np.isnan(kpca_errors[i])
        print(f"Passed α = {round(mixing,4)}", end='\r')

    assert all(krr_errors[i] <= krr_errors[i+1] and kpca_errors[i] >= kpca_errors[i+1]for i in range(len(krr_errors)-1))


if __name__ == "__main__":
    run_tests()
    print("Everything passed")
