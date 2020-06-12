import numpy as np
from sklearn import exceptions
from sklearn.utils.validation import check_X_y

from utilities.sklearn_covr.kpcovr import KernelPCovR
from utilities.kernels import center_kernel, gaussian_kernel

tolerance = 1e-4

def run_tests():
    data = np.load('./tests/CSD-test-scaled.npz')
    X = data["X"]
    Y = data["Y"]

    # Basic Test of PCovR Errors with linear kernel
    n_mixing = 6
    mixings = np.linspace(0,1,n_mixing)
    krr_errors = np.nan * np.zeros(n_mixing)
    kpca_errors = np.nan * np.zeros(n_mixing)
    for i, mixing in enumerate(mixings):
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
        for i, mixing in enumerate(mixings[:-1]):
            if(krr_errors[i]-tolerance > krr_errors[i+1]):
                print(f"KRR error at α = {mixing} ({krr_errors[i]}) > α = {mixings[i+1]} ({krr_errors[i+1]})")
            if(kpca_errors[i] < kpca_errors[i+1]-tolerance):
                print(f"KPCA error at α = {mixing} ({kpca_errors[i]}) < α = {mixings[i+1]} ({kpca_errors[i+1]})")

        return False


    # test precomputed kernel

    K = gaussian_kernel(X, X, gamma=1.0)
    n_mixing = 6
    mixings = np.linspace(0,1,n_mixing)
    krr_errors = np.nan * np.zeros(n_mixing)
    kpca_errors = np.nan * np.zeros(n_mixing)
    for i, mixing in enumerate(mixings):
        kpcovr = KernelPCovR(mixing = mixing, kernel='precomputed',
                      n_components=8,
                      tol=1e-12,
                      krr_params = dict(alpha=1,)
                    )
        kpcovr.fit(K, Y)

        T = kpcovr.transform(K)
        # Xr = kpcovr.inverse_transform(T)
        Yp = kpcovr.predict(K)

        krr_errors[i] = np.linalg.norm(Y-Yp)**2.0 / np.linalg.norm(Y)**2.0
        kpca_errors[i] = np.linalg.norm(K-np.matmul(T,T.T))**2.0 / np.linalg.norm(K)**2.0

        try:
            assert not np.isnan(krr_errors[i]) and not np.isnan(kpca_errors[i])
            print(f"Passed α = {round(mixing,4)}", end='\r')
        except AssertionError:
            return False

    try:
        assert all(krr_errors[i]-tolerance <= krr_errors[i+1] and kpca_errors[i] >= kpca_errors[i+1]-tolerance for i in range(len(krr_errors)-1))
    except AssertionError:
        for i, mixing in enumerate(mixings[:-1]):
            if(krr_errors[i]-tolerance > krr_errors[i+1]):
                print(f"KRR error at α = {mixing} ({krr_errors[i]}) > α = {mixings[i+1]} ({krr_errors[i+1]})")
            if(kpca_errors[i] < kpca_errors[i+1]-tolerance):
                print(f"KPCA error at α = {mixing} ({kpca_errors[i]}) < α = {mixings[i+1]} ({kpca_errors[i+1]})")

        return False

    # test that inverse transform does not work with precomputed kernel
    try:
        kpcovr = KernelPCovR(mixing = mixing, kernel='precomputed',
                      n_components=8,
                      tol=1e-12,
                      fit_inverse_transform = True,
                      krr_params = dict(mixing=1e-2,)
                    )
    except ValueError:
        pass

    return True
if __name__ == "__main__":
    passing = run_tests()
    if(passing):
        print("Everything passed.")
    else:
        print('Errors occurred.')
