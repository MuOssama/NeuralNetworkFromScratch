# NeuralNetworkFromScratch
Neural Network From Scratch and with keras

This document provides a comprehensive explanation of a Python code that implements a neural network with one hidden layer, trained using backpropagation. The network uses ReLU activation in the hidden layer and Sigmoid activation in the output layer for binary classification.  
In this Repo there are 2 versions:  
1- from scratch version which is documented here  
2- keras version

---

## Table of Contents
1. [Import Libraries](#import-libraries)
2. [Initialize Inputs, Weights, and Biases](#initialize-inputs-weights-and-biases)
3. [Define Activation Functions and Their Derivatives](#define-activation-functions-and-their-derivatives)
4. [Set Training Parameters](#set-training-parameters)
5. [Training Loop](#training-loop)
6. [Final Output](#final-output)
7. [Code Summary](#code-summary)

---

### 1 Import Libraries

numpy: The code uses numpy (imported as np) for efficient handling of arrays and matrix operations. numpy is crucial for performing the linear algebra operations required in neural network computations, including matrix multiplication and element-wise operations.

```python
import numpy as np 
```




  
### 2 Initialize Inputs, Weights, and Biases


```
inputs = np.array([0.2, 0.1, 0.7])  # Input vector
hidden_weights = np.array([[0.1, 0.4, 0.2, 0.3],
                           [0.4, 0.1, 0.1, 0.2],
                           [0.1, 0.2, 0.2, 0.2]])  # Weights for hidden layer
hidden_bias = np.array([0.1, 0.2, 0.3, 0.4])  # Bias for hidden layer neurons

output_weights = np.array([0.1, 0.4, 0.2, 0.3])  # Weights for output layer
output_bias = 0.1  # Bias for output neuron
```
inputs: A 3-element array representing the input vector, where each element corresponds to a feature used in making predictions.
hidden_weights: A 3x4 matrix representing the weights between the input layer and the hidden layer, with each row corresponding to an input feature and each column to a neuron in the hidden layer.
hidden_bias: A vector with four elements, each representing the bias term for one hidden layer neuron.
output_weights: A 4-element vector representing the weights connecting the hidden layer neurons to the output neuron.
output_bias: A scalar value representing the bias term for the output neuron.

### 3 Define Activation Functions and Their Derivatives  

```
def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)
    
def relu_derivative(x):
    """Derivative of ReLU function."""
    return np.where(x > 0, 1, 0)
    
def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))
    
def sigmoid_derivative(x):
    """Derivative of sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)

```
### 4 Set Training Parameters  
```
learning_rate = 0.01  # Step size for weight updates
epochs = 5  # Number of iterations over the dataset

```
### 5 Training Loop  
The training loop runs for a specified number of epochs, where each epoch involves forward and backward passes.
```
for epoch in range(epochs):
```
#### Forward Propagation
Hidden Layer Calculations: The input is multiplied by the weights and added to the biases to calculate the hidden layer’s input. ReLU is then applied to obtain the hidden layer’s output.
```
hidden_layer_input = np.dot(inputs, hidden_weights) + hidden_bias
hidden_layer_output = relu(hidden_layer_input)

```
Output Layer Calculations: The hidden layer’s output is passed to the output layer, and the sigmoid activation function is applied to obtain the final output.
```
output_layer_input = np.dot(hidden_layer_output, output_weights) + output_bias
output = sigmoid(output_layer_input)

```
Loss: Binary cross-entropy loss is computed by comparing the network’s output to the true label (y_true). This measures how close the prediction is to the target.
```
loss = - (y_true * np.log(output) + (1 - y_true) * np.log(1 - output))

```
#### Backpropagation 
Backpropagation to Output Layer  
Calculate Loss Gradient: The derivative of the loss with respect to the output (dL_doutput) is calculated first. Then, we apply the chain rule to backpropagate this error through the sigmoid function, obtaining the gradient with respect to the output layer’s input (dL_dz_out).
```
dL_doutput = - ((y_true / output) - ((1 - y_true) / (1 - output)))
doutput_dz_out = sigmoid_derivative(output_layer_input)
dL_dz_out = dL_doutput * doutput_dz_out

```
Gradient Calculation for Output Weights and Bias: The gradient of the loss with respect to the output weights and output bias is calculated.
```
dL_doutput_weights = dL_dz_out * hidden_layer_output
dL_doutput_bias = dL_dz_out

```
Backpropagation to Hidden Layer  
Backpropagate to Hidden Layer: The error from the output layer is backpropagated to the hidden layer using the output weights and ReLU derivative
```
dL_dhidden_output = dL_dz_out * output_weights
dL_dz_hidden = dL_dhidden_output * relu_derivative(hidden_layer_input)

```

Gradients for Hidden Layer Weights and Biases: The gradients for weights and biases in the hidden layer are computed.
```
dL_dhidden_weights = np.outer(inputs, dL_dz_hidden)
dL_dhidden_bias = dL_dz_hidden
```
Update Weights and Biases
Gradient Descent Step: Each weight and bias is updated by subtracting the product of the learning rate and its respective gradient, moving them in the direction that minimizes the loss.
```
output_weights -= learning_rate * dL_doutput_weights
output_bias -= learning_rate * dL_doutput_bias
hidden_weights -= learning_rate * dL_dhidden_weights
hidden_bias -= learning_rate * dL_dhidden_bias

```



### 6 Final Output
These final values represent the trained parameters of the neural network.  
```
print("Final hidden weights:", hidden_weights)
print("Final hidden biases:", hidden_bias)
print("Final output weights:", output_weights)
print("Final output bias:", output_bias)


```  
  
### 7 Code Summary
This neural network code implements a forward pass, loss calculation, backpropagation, and gradient descent optimization to train a model with one hidden layer on a binary classification task. The network can make predictions by running a forward pass with the trained weights and biases.

Each component, from input processing to weight updates, works together to enable the network to learn from data and minimize classification errors.
