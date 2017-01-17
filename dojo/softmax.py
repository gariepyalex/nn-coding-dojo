import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
  
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
  
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
  
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
  
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        probabilities = np.zeros(num_classes)
        log_c = -scores.max()
        denominator = 0.0
        for j in xrange(num_classes):
            probabilities[j] = np.exp(scores[j] + log_c)
            denominator += probabilities[j]
            
        probabilities /= denominator # normalize probabilities
        loss += -np.log(probabilities[y[i]])
        
        for j in xrange(num_classes):
            if j == y[i]:
                dW[:, j] -= (1 - probabilities[j]) * X[i,:]
            else:
                dW[:, j] += probabilities[j] * X[i,:]
                

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
  
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
  
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
  
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    scores = X.dot(W)
    max_scores = scores.max(axis=1).reshape((-1,1))
    scores -= max_scores #for numeric stability
    scores_exp = np.exp(scores)
    softmaxs = scores_exp / scores_exp.sum(axis=1).reshape((-1,1))
    loss = np.sum(-np.log(softmaxs[np.arange(num_train),y]))
    
    softmaxs[np.arange(num_train), y] -= 1
    dW += X.T.dot(softmaxs)
    
    loss /= num_train
    dW /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
  
    return loss, dW

