from tensorflow.keras.layers import SimpleRNN

# Define the RNN model
def create_rnn(input_shape, num_classes):
    model = Sequential([
        SimpleRNN(32, input_shape=input_shape),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Example usage
input_shape = (None, 100)  # Example input shape for sequences of length 100
num_classes = 10  # Example number of classes
rnn_model = create_rnn(input_shape, num_classes)
rnn_model.summary()
