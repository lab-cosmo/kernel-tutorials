import unittest
from utilities.sklearn_covr.pcovr import PCovR
import _base


class PCovRTest(_base.PCovRTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = lambda mixing, **kwargs: PCovR(
            mixing, full_eig=False, regularization=1E-8, **kwargs)


if __name__ == "__main__":
    unittest.main()
