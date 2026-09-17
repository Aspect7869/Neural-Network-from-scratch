# Neural Network from Scratch : MNIST Digit Recogniser

This is a repo of MLP (Multi Layer Perception) made from scratch or rather with minimal help from AI, using just mathematical libraries like NumPy and no ML libraries like PyTorch or Tensorflow.
I plan to have 2 stages for this project, and this is the **stage one** of the two part project.

## Architecture & Mathematical Engine
The network is designed to classify the 60,000 images in the MNIST handwritten digit dataset. 
* **Input Layer:** 784 nodes (representing the 28x28 flattened pixel arrays of MNIST images).
* **Hidden Layer:** 128 nodes using the **ReLU** (Rectified Linear Unit) activation function to handle non-linear patterns.
* **Output Layer:** 10 nodes (representing digits 0-9) using the **Softmax** activation function to output a clean probability distribution.
* **Loss Function:** Using negative log likelihood Loss to penalize overconfident incorrect predictions.
* **Optimisation:** Gradient Descent using the calculus for precise parameter updates across 100,000+ weights and biases.

I'll be uploading shortly the result of first training done on this.
the parameters used are as follows: 
* **Learning rate** = 0.01
* **Iterations** = 50,000
* **Training time** ~ Around 1 to 1.5 hours on standard hardware.

PS : If you try running this on your laptop, make sure the data folder (archive) is in the same directory as the code.
  
