# Neural network built from scratch with minimal use of AI.
# By Aaryan Sonawane 

import numpy as np 

# Fetchig data form the fiies 
def load_mist_images(fpath): 
    with open(fpath, 'rb') as f: 

        # Open the file in read binary mode 
        # f.reead() - reads the entire content of the file as a strream of raw bytes 
        # np.frombuffer() - An optimized NumPy function, maps the stream of raw bytes into an 1D array 
        # np.unit8 - Tells NumPy to intterpret the bytes as 9 bit unsigned integers. ( Value from 0 to 255) 
        # offset=16 - 16 byte headers containing the meta data, so we skip it. 
        rawdata=np.frombuffer(f.read(), np.uint8, offset=16) 

        # Reshaping raw data as it is just 60k image's data in one continuous line 
        x=rawdata.reshape(-1, 784) 

        # returning the normalized array 
        return (x/255.0) 
    
def load_nmnist_labels(fpath): 
    with open(fpath, 'rb') as f: 

        # Read the files, skip the 8 bit headers 
        labels=np.frombuffer(f.read(), np.uint8, offset=8) 

        # labels is now  a 1D array with 60k numbers in it 
        # We use one hot coding to avoid the NN interpreting the higher answer as the better answer. 
        # The NN will have 10 output nodes for each digit 
        # Encoding the labels into One Hot label's array 
        num_samples=labels.shape[0] # no. of samples basically 

        # Now we create a 60k x 10 array containing all zeroes initially 
        y1hot=np.zeros((num_samples,10)) 

        # We change the zeroes from a 0 to 1 wherever neceessary 
        # We need to make only one 1 cfhange per row and convert the 0->1 only for the number 
        # that is in labels[i] as this y1hot's columns represents the 10 numbers we have. 
        for i in range(num_samples): 
            y1hot[i, labels[i]]=1.0 
        return y1hot 

def loadprepd_data(): 
    # Now we just load and read the data and store em into some variable 
    print("Loading the binary data. . . ") 
    xtrain=load_mist_images('archives/train-images.idx3-ubyte') 
    ytrain=load_nmnist_labels('archives/train-labels.idx1-ubyte') 
    xtest=load_mist_images('archives/t10k-images.idx3-ubyte') 
    ytest=load_nmnist_labels('archives/t10k-labels.idx1-ubyte') 
    print("Data loaded successful;ly") 
    print(f"X train shape :{xtrain.shape} (60k images, 784 pixels each)") 
    print(f"y train shape :{ytrain.shape} (60k labels, 10 one-shot classes)") 
    return xtrain, ytrain, xtest, ytest 

# We run ts int he main function 
if __name__=="__main__": 
    xtrain, ytrain, xtest, ytest = loadprepd_data()