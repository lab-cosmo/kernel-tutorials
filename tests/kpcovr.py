import unittest
import _base

from utilities.sklearn_covr.kpcovr import KernelPCovR as KPCovR
from utilities.kernels import center_kernel, gaussian_kernel

#
# class KPCovRTest(unittest.TestCase):
#     def __init__(self, kernel = None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         self.data = np.load('./tests/CSD-test.npz')
#         self.X = self.data["X"]
#         self.Y = self.data["Y"]
#         if(kernel is not None):
#             print(kernel)
#             self.K = kernel(self.X, self.X)
#         else:
#             self.K = None
#
#         self.error_tol = 1E-3
#         self.rounding = -int(round(np.log10(self.error_tol)))
#
#         n_mixing = 6
#         self.krr_errors = np.nan * np.zeros(n_mixing)
#         self.kpca_errors = np.nan * np.zeros(n_mixing)
#         self.alphas = np.linspace(0, 1, n_mixing)
#
#     def setUp(self):
#         self.startTime = time.time()
#
#     def tearDown(self):
#         t = time.time() - self.startTime
#         print('%s: %.3f' % (self.id(), t))
#
#     def rel_error(self, A, B):
#         return np.linalg.norm(A-B)**2.0 / np.linalg.norm(A)**2.0
#
#     def run_kpcovr(self, mixing, n_components=2, regularization=1E-6, full_eig=False, tol=1E-12):
#         kpcovr_params = dict(mixing=mixing,
#                               n_components=n_components,
#                               regularization=regularization,
#                               full_eig=full_eig,
#                               krr_params=dict(alpha=1e-6,),
#                               tol=tol)
#         if(self.K is not None):
#             kpcovr_params['kernel'] = 'precomputed'
#
#         pcovr = KPCovR(**kpcovr_params)
#         pcovr.fit(self.X, self.Y, K=K)
#         Yp = pcovr.predict(self.X, K=K)
#         T = pcovr.transform(self.X, K=K)
#         Xr = pcovr.inverse_transform(T)
#         return Yp, T, Xr
#
#     def add_errors(self):
#         for i, mixing in enumerate(self.alphas):
#             if(np.isnan(self.krr_errors[i]) or np.isnan(self.kpca_errors[i])):
#                 Yp, _, Xr = self.run_pcovr(mixing=mixing)
#                 self.krr_errors[i] = self.rel_error(self.Y, Yp)
#                 self.kpca_errors[i] = self.rel_error(self.X, Xr)
#
#     def add_rbf_errors(self):
#         for i, mixing in enumerate(self.alphas):
#             if(np.isnan(self.krr_rbf_errors[i]) or np.isnan(self.kpca_rbf_errors[i])):
#                 Yp, _, Xr = self.run_pcovr(mixing=mixing, K=self.K)
#                 self.krr_rbf_errors[i] = self.rel_error(self.Y, Yp)
#                 self.kpca_rbf_errors[i] = self.rel_error(self.X, Xr)
#
#     # Checks that the KPCovR will not transform before fitting
#     def test_nonfitted_failure(self):
#         # Test of KPCovR Fitting
#         pcovr = KPCovR(mixing=0.5,
#                       n_components=2,
#                       regularization=1e-6,
#                       tol=1e-12)
#         kpcovr_params = dict(mixing=mixing,
#                               n_components=n_components,
#                               regularization=regularization,
#                               full_eig=full_eig,
#                               krr_params=dict(alpha=1e-6,),
#                               tol=tol)
#         if(self.K is not None):
#             kpcovr_params['kernel'] = 'precomputed'
#
#         with self.assertRaises(exceptions.NotFittedError):
#             _ = pcovr.transform(self.X)

kpcovr_params = dict(alpha=1E-8,
                    fit_inverse_transform=True, eigen_solver='auto',
                    )
class KPCovRTestX(_base.PCovRTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = lambda mixing, **kwargs: KPCovR(mixing=mixing,
                                                     **kpcovr_params,
                                                     **kwargs)

class KPCovRTestRBF(_base.PCovRTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kernel = lambda X: center_kernel(gaussian_kernel(X, X, gamma=0.5))
        self.model = lambda mixing, **kwargs: KPCovR(mixing=mixing,
                                                     kernel=kernel,
                                                     **kpcovr_params,
                                                     **kwargs)
        self.K = kernel(self.X)

if __name__ == "__main__":
    unittest.main()
