import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Generate some simple sequential data
# For example, a sine wave
def generate_data(seq_length, num_sequences):
    X = []
    y = []
    for _ in range(num_sequences):
        start = np.random.rand()
        sequence = np.sin(np.linspace(start, start + np.pi, seq_length))
        X.append(sequence[:-1])
        y.append(sequence[-1])
    return np.array(X), np.array(y)

# Parameters
seq_length = 50
num_sequences = 1000

# Generate data
X, y = generate_data(seq_length, num_sequences)

# Reshape data for RNN input
X = X.reshape((X.shape[0], X.shape[1], 1))

# Define the RNN model
model = Sequential()
model.add(SimpleRNN(50, activation='relu', input_shape=(seq_length-1, 1)))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=200, verbose=1)

