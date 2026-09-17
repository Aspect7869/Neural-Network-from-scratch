import numpy as np

def init_params():
    # W1 connects the 784 input pixels to the 128 hidden nodes.
    # Shape: (128 rows, 784 columns)
    # We subtract 0.5 to center the random values between -0.5 and +0.5
    W1 = np.random.rand(128, 784) - 0.5
    
    # b1 is the bias for the 128 hidden nodes.
    # Shape: (128, 1)
    b1 = np.random.rand(128, 1) - 0.5
    
    # W2 connects the 128 hidden nodes to the 10 output nodes.
    # Shape: (10 rows, 128 columns)
    W2 = np.random.rand(10, 128) - 0.5
    
    # b2 is the bias for the 10 output nodes (our digits 0-9).
    # Shape: (10, 1)
    b2 = np.random.rand(10, 1) - 0.5
    
    return W1, b1, W2, b2

def ReLu(Z):
    # Basically is the max between 0 and Z, is 0 if Z is negative
    # np.maximum comapres every element with 0.
    return np.maximum(0,Z)

def softmax(Z):
    # The softmax function used for the outermost layer
    # Coverts raw output to clear probabilities that sum up to 1.
    A= np.exp(Z)/sum(np.exp(Z))
    return A

def forward_prop(w1,b1,w2,b2,X):
    # Input Layer to hidden layer
    # Here Z = W.X(transpose)_b1, so we use X.T to get the transpose of X, alligning the 784 columns and rows.
    Z1 = w1.dot(X.T)+b1
    A1=ReLu(Z1)

    # Hidden Layer to output layer
    Z2 = w2.dot(A1)+b2
    A2=softmax(Z2)

    return Z1,A1,Z2,A2
