import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import Callback

# Sample data (inputs and target output)
X = np.array([[0.2, 0.1, 0.7]])  # Input data
y = np.array([[1]])  # Target output

# Define a custom callback to print weights and biases
class WeightsBiasesLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}:")
        for layer in self.model.layers:
            weights, biases = layer.get_weights()  # Get the weights and biases
            print(f"Layer {layer.name} weights:\n{weights}")
            print(f"Layer {layer.name} biases:\n{biases}")

# Define the model
model = Sequential()
model.add(Dense(4, input_dim=3, activation='relu', name='hidden_layer'))  # Hidden layer with 4 neurons, ReLU activation
model.add(Dense(1, activation='sigmoid', name='output_layer'))            # Output layer with 1 neuron, sigmoid activation

# Compile the model
model.compile(optimizer=SGD(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

# Initialize weights and biases
hidden_weights = np.array([[0.1, 0.4, 0.2, 0.3],
                            [0.4, 0.1, 0.1, 0.2],
                            [0.1, 0.2, 0.2, 0.2]])
hidden_bias = np.array([0.1, 0.2, 0.3, 0.4])

output_weights = np.array([[0.1], [0.4], [0.2], [0.3]])  # Reshape to (4, 1) for the output layer
output_bias = np.array([0.1])

# Set initial weights and biases for the hidden layer
model.layers[0].set_weights([hidden_weights, hidden_bias])

# Set initial weights and biases for the output layer
model.layers[1].set_weights([output_weights, output_bias])

# Train the model on the data with the custom callback
model.fit(X, y, epochs=10, verbose=1, callbacks=[WeightsBiasesLogger()])  # Train for 10 epochs

# Make a prediction
predicted_output = model.predict(X)
print("Predicted Output:", predicted_output)
