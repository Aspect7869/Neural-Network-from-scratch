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


# Loss calculation would be done using negative log likelihood function

def loss(y, A2): 
    # Here y = ground truth labels, and A2 = NN's final predictions
    # If the prediction in A2 is close to 1 and correct, then the log will tend to 0
    # => Smaller loss
    # If the probability is close to 0 and wrong, then the log blows up to a bigger number
    # => Bigger loss
    # Multiplying the log of the prediction with Y forces the calculation of only the right number
    # THis is because Y is a mtrix of one hots, meaning the dot will 0 with every other prediction.
    n=y.shape[0] # Number of samples, 60k in this case
    ep=1e-8 # we add a small value to prevent the crash at log(0)
    L= -(1/n)* np.sum((y.T)*np.log(A2+ep)) 
    return L

def predictions(A2):
    # np.argmax looks at the 10 probabilities for an image and returns the index of highest probabiity wala number
    # we need to add axis=0 in this case, as A2 is a 10 row x 60k column matrix.
    # without axis=0, np.argmax would return the greatest number's index from the 600k numbers in A2.
    return np.argmax(A2, axis=0)

def accuracy(pred, y):
    # We now convert the one hot matrix back to simple digits, to compare it directly to our predictions.
    # axis=1 basically moves thru the matrix row by row and returns the index of the 1 present in the row.
    # Y is a 60k rows x 10 column matrix btw.
    labels=np.argmax(y,axis=1)
    acc= np.sum(pred==labels)/y.shape[0]
    return acc


if __name__ == "__main__":
    # Import your data loader from the other file
    from mnist_data import loadprepd_data
    
    # 1. Load the data
    xtrain, ytrain, xtest, ytest = loadprepd_data()
    
    # 2. Initialize random parameters
    W1, b1, W2, b2 = init_params()
    
    # 3. Do a forward pass
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, xtrain)
    
    # 4. Calculate the initial loss and accuracy
    initial_loss = loss(ytrain, A2)
    predictions = predictions(A2)
    initial_accuracy = accuracy(predictions, ytrain)
    
    print(f"\n--- Initial Network Test ---")
    print(f"Starting Loss: {initial_loss:.4f}")
    print(f"Starting Accuracy: {initial_accuracy * 100:.2f}%")