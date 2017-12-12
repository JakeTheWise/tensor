import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound, nonecheck

# tucker decomposition model
cdef class tdcmp:
  cdef public:
    np.ndarray S, calcs, M, B, slices_lf, slices_b
    float mu
    str calc_S, calc_pred, calc_loop
  
  def __cinit__(self, tuple shape, tuple ranks):
    np.random.seed(42)
    self.S = np.random.triangular(-.1,0,.1,size=ranks).astype('f') # S is the small central tensor
    self.M = np.array([np.random.triangular(-.1,0,.1,size=i).astype('f') for i in zip(shape,ranks)]) # M holds the latent factor matrices for each mode
    self.B = np.array([np.random.triangular(-1,0,1,size=i).astype('f') for i in shape]) # B holds the bias vectors for each mode
    
    # precompute the Einstein Sum notations we'll need:
    cdef str c = ''.join(chr(ord('a')+i) for i in range(len(shape)))
    self.calcs = np.array([','.join([c] + [j for j in c if j != mask]) + '->' + mask for mask in c])
    self.calc_S = ','.join(i for i in c)+'->'+c
    self.calc_pred = ','.join([c]+[i for i in c])+'->'
    
  @boundscheck(False)
  @wraparound(False)
  @nonecheck(False)
  def tucker_fit(self, int[:,:] X, int[:] Y, float lr=.005, float reg=0.2):
    # high-order singular value decomposition based on Tucker decomposition
    # trains the latent factor matrices and bias vectors
    cdef float diff, count
    if not self.mu:
      # set global mean via inverse logistic transform
      self.mu = -np.log(1 / np.asarray(Y).mean() - 1)
    for x, y in zip(X, Y):
      # x is a tuple of indices
      if count % 1000 == 0:
        print('{:.2%}'.format(count/len(Y)), end="\r")
      diff = self._pred_one(x) - y # derivative of binary crossentropy loss function w.r.t y_hat
      for dim, ix in enumerate(x):
        self.M[dim][ix] -= lr * (diff * np.einsum(self.calcs[dim], self.S, *[self.slices_lf[i] for i in range(len(self.slices_lf)) if i != dim], optimize=True) + reg * self.M[dim][ix]) # update factor matrix
        self.B[dim][ix] -= lr * (diff + reg * self.B[dim][ix]) # update bias
      self.S -= lr * (diff * np.einsum(self.calc_S, *self.slices_lf, optimize=True) + reg * self.S) # update central tensor
      count += 1
    
  @boundscheck(False)
  @wraparound(False)
  @nonecheck(False)
  def predict(self, int[:,:] X):
    return np.array([self._pred_one(x) for x in X])
  
  @boundscheck(False)
  @wraparound(False)
  @nonecheck(False)
  cdef float _pred_one(self, int[:] x):
    # predicts one sample, saving the slices of the factor matrices and bias vectors to self for use in gradient updating in hosvd method
    self.slices_lf, self.slices_b = np.array([(self.M[dim][ix], self.B[dim][ix]) for dim, ix in enumerate(x)]).T
    return 1 / (1 + np.exp(-(np.einsum(self.calc_pred,self.S,*self.slices_lf, optimize=True) + sum(self.slices_b) + self.mu))) # logistic transform since we want to model probabilities
