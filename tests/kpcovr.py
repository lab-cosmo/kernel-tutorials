import numpy as np
from sklearn import exceptions
from sklearn.utils.validation import check_X_y

from utilities.sklearn_covr.kpcovr import KernelPCovR

tolerance = 1e-4

def run_tests():
    data = np.load('./tests/CSD-test-scaled.npz')
    X = data["X"]
    Y = data["Y"]

    # Basic Test of PCovR Errors
    n_alpha = 21
    alphas = np.linspace(0,1,n_alpha)
    krr_errors = np.nan * np.zeros(n_alpha)
    kpca_errors = np.nan * np.zeros(n_alpha)
    for i, mixing in enumerate(alphas):
        kpcovr = KernelPCovR(mixing = mixing, kernel='linear',
                      n_components=8,
                      fit_inverse_transform=True,
                      tol=1e-12,
                      krr_params = dict(alpha=1e-2,)
                    )
        kpcovr.fit(X, Y)

        T = kpcovr.transform(X)
        Xr = kpcovr.inverse_transform(T)
        Yp = kpcovr.predict(X)

        krr_errors[i] = np.linalg.norm(Y-Yp)**2.0 / np.linalg.norm(Y)**2.0
        kpca_errors[i] = np.linalg.norm(X-Xr)**2.0 / np.linalg.norm(X)**2.0

        try:
            assert not np.isnan(krr_errors[i]) and not np.isnan(kpca_errors[i])
            print(f"Passed α = {round(mixing,4)}", end='\r')
        except AssertionError:
            return False

    try:
        assert all(krr_errors[i]-tolerance <= krr_errors[i+1] and kpca_errors[i] >= kpca_errors[i+1]-tolerance for i in range(len(krr_errors)-1))
    except AssertionError:
        for i, alpha in enumerate(alphas[:-1]):
            if(krr_errors[i]-tolerance > krr_errors[i+1]):
                print(f"KRR error at α = {alpha} ({krr_errors[i]}) > α = {alphas[i+1]} {krr_errors[i+1]}")
            if(kpca_errors[i] < kpca_errors[i+1]-tolerance):
                print(f"KPCA error at α = {alpha} ({kpca_errors[i]}) < α = {alphas[i+1]} {kpca_errors[i+1]}")

        return False

    return True
if __name__ == "__main__":
    passing = run_tests()
    if(passing):
        print("Everything passed.")
    else:
        print('Errors occurred.')
