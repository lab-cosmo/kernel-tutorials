import numpy as np
import sklearn as sk
from sklearn.base import TransformerMixin, BaseEstimator

class NormalizeScaler(TransformerMixin, BaseEstimator):
    """ A more flexible scaler for features and data, that mimics 
    sklearn.preprocessing.StandardScaler but makes the mean norm of the rows
    equal to one. """
    
    
    def __init__(self, *, with_mean=True, with_norm=True, per_feature=False):
        """ Initialize NormalizeScaler. Defines whether it will subtract mean
        (with_mean=True), apply normalization (with_norm=True) and whether it will
        normalize each feature separately (per_feature=True). """
        
        self.__with_mean = with_mean
        self.__with_norm = with_norm
        self.__per_feature = per_feature
        self.n_samples_seen_ = 0
        
    def fit(self, X, y=None):
        """ Compute mean and scaling to be applied for subsequent normalization. """
        
        
        self.n_samples_seen_, self.n_features_ = X.shape
        self.mean_ = np.zeros(self.n_features_)        
        
        if self.__with_mean:
            self.mean_ = X.mean(axis=0)
            centred_X = X - self.mean_
        else:
            centred_X = X
        
        self.scale_ = 1.0
        if self.__with_norm:
            var = (centred_X**2).mean(axis=0)
            
            if self.__per_feature:
                if np.any(var==0):
                    raise ValueError("Cannot normalize a feature with zero variance")
                self.scale_ = np.sqrt(1.0/(self.n_features_*var))                
            else:
                self.scale_ = 1.0/np.sqrt(var.sum())
                
        return self

    def transform(self, X, y=None):
        """ Normalize a vector based on previously computed mean and scaling. """
        
        if self.n_samples_seen_ == 0 :
            raise sk.exceptions.NotFittedError("This "+type(self).__name__+" instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        return self.scale_*(X-self.mean_)

class KernelNormalizer(TransformerMixin, BaseEstimator):
    """ A more flexible kernel centerer that can work for both full and sparse 
    kernel frameworks, and that can normalize, center, or fully standardize the kernel. """
    
    def __init__(self, *, with_mean=True, with_norm=True, rcond=None):
        """ Initialize KernelNormalizer. Defines whether it will subtract mean
        (with_mean=True), apply normalization (with_norm=True). The LSTSQ eigenvector cutoff
        applied to invert the active-set kernel in the sparse case is given by rcond. """
        
        self.__with_mean = with_mean
        self.__with_norm = with_norm
        self.__rcond = rcond
        self.__sparse_kernel = False
        self.n_samples_seen_ = 0
        self.n_active_ = 0

    def fit(self, K, y=None, K_active=None):
        """ Compute mean and scaling to be applied to kernels. If K_active is given,
        then it works under the assumption we are dealing with sparse, Nystroem approximation
        kernels. K is then the n_train x n_active train set matrix. 
        Otherwise, K should be the n_train x n_train full train set matrix. """        
        
        if K_active is not None:
            self.__sparse_kernel = True
            self.n_active_ = K_active.shape[0]
            if K_active.shape[0] != K_active.shape[1]:
                raise ValueError("K_active should be the square, active set kernel. ")
                
        self.n_samples_seen_ = K.shape[0]
        if self.__sparse_kernel:
            if K.shape[1] != self.n_active_:
                raise ValueError("For sparse kernels, K should be train-active set kernel. ")
        elif K.shape[0] != K.shape[1]:
            raise ValueError("K should be the square, train set kernel. ")
        
        self.K_train_mean_ = 0
        
        K_diag = K.diagonal().copy()
        if self.__with_mean:
            self.K_train_row_mean_ = K.mean(axis=0)
            self.K_train_mean_ = K.mean()
            if not self.__sparse_kernel:
                K_diag +=  - self.K_train_row_mean_*2 + self.K_train_mean_        
                    
        self.scale_ = 1.0
        if self.__with_norm:
            if self.__sparse_kernel:
                var = np.sqrt( 
                        np.trace(K@np.linalg.lstsq(K_active, K.T, self.__rcond)[0])/
                        self.n_samples_seen_
                        )
            else:
                var = K_diag.mean()            
            self.scale_ = 1.0/var 
                
        return self

    def transform(self, K_test, y=None, K_test_train = None):
        """ Normalize a test kernel based on previously computed mean and scaling. 
        Should be called in different ways depending on the situation. 
        If you have full kernel learning, and the test kernel is between a test set and 
        the train set just call with
        transform(K_test)   # K_test is n_test x n_train
        If you have a kernel between two arbitrary sets A and B, you need to provide
        transform(K_AB, K_test_train(K_Atrain, K_Btrain)
        If you have a sparse kernel framework, then you need to provide the kernel matrix
        between test and active points
        transform(K_test)   # K_test is n_test x n_active        
        """
        
        if self.n_samples_seen_ == 0 :
            raise sk.exceptions.NotFittedError("This "+type(self).__name__+" instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        
        K_std = K_test.copy()
        if self.__sparse_kernel:
            if K_test.shape[1] != self.n_active_:
                raise ValueError("Test kernel must have n_active as second dimension")
            if self.__with_mean:
                K_std -= self.K_train_row_mean_
        else:
            if K_test_train is None:
                if K_test.shape[1] != self.n_samples_seen_:
                    raise ValueError("Test kernel must have n_train as second dimension")
                if self.__with_mean:
                    K_std += self.K_train_mean_ - np.add.outer(K_test.mean(axis=1), self.K_train_row_mean_)          
            else:
                K_rows, K_cols = K_test_train
                K_std += self.K_train_mean_ - np.add.outer( K_rows.mean(axis=1), K_cols.mean(axis=0) )

        if self.__with_norm:
            K_std *= self.scale_
            
        return K_std
