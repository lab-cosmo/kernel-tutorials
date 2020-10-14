import numpy as np
from sklearn import exceptions
from sklearn.utils.validation import check_X_y
import unittest
import time


class PCovRTestBase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = np.load('./tests/CSD-test.npz')
        self.X = self.data["X"]
        self.Y = self.data["Y"]
        self.K = None
        self.model = None

        self.error_tol = 1E-3
        self.rounding = -int(round(np.log10(self.error_tol)))

        n_mixing = 11
        self.lr_errors = np.nan * np.zeros(n_mixing)
        self.pca_errors = np.nan * np.zeros(n_mixing)
        self.alphas = np.linspace(0, 1, n_mixing)

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    def rel_error(self, A, B):
        return np.linalg.norm(A-B)**2.0 / np.linalg.norm(A)**2.0

    def run_pcovr(self, mixing):
        pcovr = self.model(mixing=mixing,
                           n_components=2,
                           tol=1E-12)
        if(self.K is None):
            pcovr.fit(self.X, self.Y)
            Yp = pcovr.predict(self.X)
            T = pcovr.transform(self.X)
        else:
            pcovr.fit(self.X, self.Y, K=self.K)
            Yp = pcovr.predict(self.X, K=self.K)
            T = pcovr.transform(self.X, K=self.K)
        Xr = pcovr.inverse_transform(T)
        return Yp, T, Xr

    def add_errors(self):
        for i, mixing in enumerate(self.alphas):
            if(np.isnan(self.lr_errors[i]) or np.isnan(self.pca_errors[i])):
                Yp, _, Xr = self.run_pcovr(mixing=mixing)
                self.lr_errors[i] = self.rel_error(self.Y, Yp)
                self.pca_errors[i] = self.rel_error(self.X, Xr)

    # Basic Test of model LR Errors, that None return np.nan

    def test_lr_errors(self):
        self.add_errors()

        for i, mixing in enumerate(self.alphas):
            with self.subTest(error=self.lr_errors[i]):
                self.assertFalse(np.isnan(self.lr_errors[i]))

    # Basic Test of model PCA Errors, that None return np.nan
    def test_pca_errors(self):
        self.add_errors()

        for i, mixing in enumerate(self.alphas):
            with self.subTest(error=self.pca_errors[i]):
                self.assertFalse(np.isnan(self.pca_errors[i]))

    # Test that model LR Errors are monotonic with alpha
    def test_lr_monotonicity(self):
        self.add_errors()

        for i, _ in enumerate(self.alphas[:-1]):
            with self.subTest(i=i):
                lr1 = round(self.lr_errors[i], self.rounding)
                lr2 = round(self.lr_errors[i+1], self.rounding)
                self.assertTrue(lr1 <= lr2,
                                msg=f'LR Error Non-Monotonic\n {lr1} >  {lr2}'
                                )

    # Test that model PCA Errors are monotonic with alpha
    def test_pca_monotonicity(self):
        self.add_errors()

        for i, a in enumerate(self.alphas[:-1]):
            with self.subTest(i=i):
                pca1 = round(self.pca_errors[i], self.rounding)
                pca2 = round(self.pca_errors[i+1], self.rounding)
                self.assertTrue(pca1 >= pca2,
                                msg=f'PCA Error Non-Monotonic\n {pca1} < {pca2}'
                                )

    # Checks that the model will not transform before fitting
    def test_nonfitted_failure(self):
        model = self.model(mixing=0.5,
                           n_components=2,
                           tol=1E-12)
        with self.assertRaises(exceptions.NotFittedError):
            if(self.K is not None):
                _ = model.transform(self.X, K=self.K)
            else:
                _ = model.transform(self.X)

    def test_T_shape(self):
        _, T, _ = self.run_pcovr(mixing=0.5)
        self.assertTrue(check_X_y(self.X, T, multi_output=True))
