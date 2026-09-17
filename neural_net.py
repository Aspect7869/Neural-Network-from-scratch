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

def ReLu_deriv(z):
    # Returns 1 for every element>0, 0 for every element<=0, as ReLu is basically x=y for x>0.
    # We get a True for every element>0 and false for element <=0.
    # Z = Wx+b, which is the preactivation value.
    return z>0

def backprop(z1,a1,z2,a2,w1,w2,x,y):
    n=y.shape[0] # number of images
    # Output layer error :
    dz2=a2-(y.T)

    # Output layer gradient : (for w2 and b2)
    dw2=(1/n)*dz2.dot(a1.T)
    # axis=1, sums across the columns,to get 1 bias per node
    # Shape of dz2 is 10 rows X 60K columns, bias matrix has the shape of only 10 rows x 1 column
    # keeping axis=1 basically collapses 60k columns into 1 row, by summing all of them up
    # keepdims= True basically keeps the dimensions the safe, and does not collapse the entire 
    # matrix into something like a rank 1 matrix. (Ts was added by AI as a safeguard during debugging)
    db2= (1/n)*np.sum(dz2, axis=1, keepdims=True)

    # Hidden Layer error :
    # w2.T's shape is (128,10)
    # dz2 is the shape of (10, 60k)
    # result is the shape of (128, 60k)
    dz1=w2.T.dot(dz2)*ReLu_deriv(z1)

    # Hidden Layer Gradients (Derivative for w1 n b1):
    dw1=(1/n)*dz1.dot(x) 
    db1=(1/n)*np.sum(dz1,axis=1,keepdims=True)

    return dw1,db1,dw2,db2

def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, a):
    # a is the learning rate. Dictates how massive of a step we take. 
    w1=w1-a*dw1
    b1=b1-a*db1
    w2=w2-a*dw2
    b2=b2-a*db2
    return w1, b1, w2, b2

def grad_descent(x,y,a,I):
    w1,b1,w2,b2= init_params()
    for i in range(I):
        # Forward pass :
        z1,a1,z2,a2=forward_prop(w1,b1,w2,b2)

        # backward pass :
        dw1,db1,dw2,db2=backprop(z1,a1,z2,a2,w1,w2,x,y)

        # update parameters :
        w1,b1,w2,b2=update_params(w1,b1,w2,b2,dw1,db1,dw2,db2,a)

        # printing progress every 50th iteration:
        if i%50==0:
            l=loss(y,a2)
            p=predictions(a2)
            acc=accuracy(p, y)
            print(f"Iteration : {i} | Loss : {l:5f} | Accuracy : {acc*100 :2f}")
    return w1,b1,w2,b2

if __name__ == "__main__":
    from mnist_data import loadprepd_data
    xtrain, ytrain, xtest, ytest = loadprepd_data()
    W1, b1, W2, b2 = init_params() # initializing parameters to a random value
    print("\n ..................Starting Training..................") 
    W1,b1,W2,b2=grad_descent(xtrain,ytrain,a=0.01,I=50000)
    # 3. Do a forward pass
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, xtrain)
    
    # 4. Calculate the initial loss and accuracy
    initial_loss = loss(ytrain, A2)
    preds = predictions(A2)
    initial_accuracy = accuracy(predictions, ytrain)
    
    print(f"\n--- Initial Network Test ---")
    print(f"Starting Loss: {initial_loss:.4f}")
    print(f"Starting Accuracy: {initial_accuracy * 100:.2f}%")

