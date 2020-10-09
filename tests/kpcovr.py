import unittest
import _base

from utilities.sklearn_covr.kpcovr import KernelPCovR as KPCovR
from utilities.kernels import center_kernel, gaussian_kernel

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
        def kernel(X): return center_kernel(gaussian_kernel(X, X, gamma=0.5))
        self.model = lambda mixing, **kwargs: KPCovR(mixing=mixing,
                                                     kernel=kernel,
                                                     **kpcovr_params,
                                                     **kwargs)
        self.K = kernel(self.X)


if __name__ == "__main__":
    unittest.main()
