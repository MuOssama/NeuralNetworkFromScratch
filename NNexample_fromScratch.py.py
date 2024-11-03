import numpy as np

# Initialize inputs, weights, and biases for hidden and output layers
inputs = np.array([0.2, 0.1, 0.7])
hidden_weights = np.array([[0.1, 0.4, 0.2, 0.3],
                           [0.4, 0.1, 0.1, 0.2],
                           [0.1, 0.2, 0.2, 0.2]])
hidden_bias = np.array([0.1, 0.2, 0.3, 0.4])

output_weights = np.array([0.1, 0.4, 0.2, 0.3])
output_bias = 0.1

# Define activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Define learning rate and number of epochs
learning_rate = 0.01
epochs = 5

# Assume the target output (label)
y_true = 1  # example target

for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(inputs, hidden_weights) + hidden_bias
    hidden_layer_output = relu(hidden_layer_input)
    print(f'hidden layer output: {hidden_layer_output}')

    output_layer_input = np.dot(hidden_layer_output, output_weights) + output_bias
    output = sigmoid(output_layer_input)

    # Calculate the Loss
    loss = - (y_true * np.log(output) + (1 - y_true) * np.log(1 - output))
    print(f'Loss: {loss}')

    # Step 1: Calculate the Loss Gradient with respect to output
    dL_doutput = - ((y_true / output) - ((1 - y_true) / (1 - output)))
    print(f'Loss Gradient with respect to outputLoss: {dL_doutput}')

    # Step 2: Backpropagate to Output Layer
    doutput_dz_out = sigmoid_derivative(output_layer_input)
    print(f'sigmoid_derivative of Output Layer: {doutput_dz_out}')

    dL_dz_out = dL_doutput * doutput_dz_out
    print(f'Backpropagate to Output Layer: {dL_dz_out}')

    # Gradients for output weights and bias
    dL_doutput_weights = dL_dz_out * hidden_layer_output
    dL_doutput_bias = dL_dz_out

    # Step 3: Backpropagate to Hidden Layer
    print(f'Backpropagate to Hidden Layer')

    dL_dhidden_output = dL_dz_out * output_weights
    dL_dz_hidden = dL_dhidden_output * relu_derivative(hidden_layer_input)
    print(f'Relu derivative: {relu_derivative(hidden_layer_input)}')
    print(f'dL/dz_hidden: {dL_dz_hidden}')
     
    # Gradients for hidden weights and biases
    dL_dhidden_weights = np.outer(inputs, dL_dz_hidden)
    dL_dhidden_bias = dL_dz_hidden
    print("ssss")
    print(dL_dhidden_bias)
    # Update weights and biases using gradient descent
    output_weights -= learning_rate * dL_doutput_weights
    output_bias -= learning_rate * dL_doutput_bias
    hidden_weights -= learning_rate * dL_dhidden_weights
    hidden_bias -= learning_rate * dL_dhidden_bias  # sum for bias update
    print("updated hidden weights:", hidden_weights)
    print("updated hidden biases:", hidden_bias)
    print("updated output weights:", output_weights)
    print("updated output bias:", output_bias)
    print(f'Epoch {epoch}, Loss: {loss}')
    print("\n\n\n")

# Final weights and biases after training
print("Final hidden weights:", hidden_weights)
print("Final hidden biases:", hidden_bias)
print("Final output weights:", output_weights)
print("Final output bias:", output_bias)
while True:
    pass