import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: D x c array of weights
  - X: N x D array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...C-1, for C classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  data_loss = 0.0
  dW = np.zeros_like(W)
  n_sample, n_feature  = X.shape
  assert n_feature == W.shape[0]
  for i in xrange(n_sample):
    scores = np.dot(X[i],W)
    scores -= np.max(scores)
    scores = np.exp(scores)
    probability = scores / np.sum(scores)

    #accumulate the data_loss
    sample_loss = -np.log(probability[y[i]])
    data_loss += sample_loss / n_sample

    #accumulate the gradient with respect to every single data sample
    probability[y[i]] -= 1
    dW_single_sample = np.outer(X[i],probability) / n_sample
    dW += dW_single_sample

  reg_loss = 0.5 * reg * np.sum(W[:-1,:] ** 2)
  loss = data_loss + reg_loss

  return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  dW = np.zeros_like(W)
  n_sample = X.shape[0]

  assert X.shape[1] == W.shape[0]
  scores = np.dot(X,W)

  scores -= np.max(scores, axis = 1, keepdims = True)
  scores = np.exp(scores)
  probs = scores / np.sum(scores, axis = 1, keepdims = True)
  kl_loss = np.sum(-np.log(probs[range(n_sample),y])) / n_sample
  reg_loss = 0.5 * reg * np.sum(W[:-1,:] ** 2)
  loss = kl_loss + reg_loss

  dscores = probs
  dscores[range(n_sample),y] -= 1
  dscores /= n_sample
  dW = np.dot(X.T, dscores)

  return loss, dW