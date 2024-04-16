# Define the ANN model
def create_ann(input_shape, num_classes):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Example usage
input_shape = (28, 28)  # Example input shape for flattened MNIST images
num_classes = 10  # Example number of classes for MNIST
ann_model = create_ann(input_shape, num_classes)
ann_model.summary()
