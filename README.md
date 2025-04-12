# cnn-cifar10-classification
A Convolutional Neural Network (CNN) model for image classification on the CIFAR-10 dataset using TensorFlow and Keras.
This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to perform image classification on the CIFAR-10 dataset, which consists of 60,000 32x32 color images across 10 different classes. The model architecture includes multiple convolutional layers with ReLU activations and 'same' padding, followed by max-pooling layers to reduce spatial dimensions, and a fully connected dense layer for final classification. The dataset is preprocessed by normalizing the pixel values and converting the labels into one-hot encoded format. The model is trained over 20 epochs with a batch size of 32, using Adam as the optimizer and categorical crossentropy as the loss function. Training and validation accuracy and loss are also visualized using Matplotlib. After training, the model achieves a test accuracy of 74.34%, making it a solid baseline for CIFAR-10 image classification tasks.

import tensorflow as tf  # Import TensorFlow for building and training neural networks
from tensorflow.keras.datasets import cifar10  # CIFAR-10 dataset: 60,000 32x32 color images in 10 classes
from tensorflow.keras.models import Sequential  # Sequential model: linear stack of layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # Layers used in CNN
from tensorflow.keras.utils import to_categorical  # Convert labels to one-hot encoding
import matplotlib.pyplot as plt  # For plotting training and validation results

# Load the dataset and split into training and testing sets
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()  # Load CIFAR-10 dataset

# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0  # Scale image data to improve training performance

# Convert labels to one-hot encoded vectors
Y_train = to_categorical(Y_train, 10)  # 10 classes for CIFAR-10
Y_test = to_categorical(Y_test, 10)

# Build the CNN model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),  # First convolution layer
    Conv2D(64, (3, 3), activation='relu', padding='same'),  # Second convolution layer
    MaxPooling2D((2, 2)),  # Reduce feature map size by half

    Conv2D(128, (3, 3), activation='relu', padding='same'),  # Third convolution layer
    Conv2D(128, (3, 3), activation='relu', padding='same'),  # Fourth convolution layer
    MaxPooling2D((2, 2)),  # Another pooling layer to reduce dimensions

    Conv2D(128, (3, 3), activation='relu', padding='same'),  # Fifth convolution layer

    Flatten(),  # Flatten the 3D output to 1D for the dense layers
    Dense(128, activation='relu'),  # Fully connected layer with 128 neurons
    Dense(10, activation='softmax')  # Output layer with 10 neurons for classification
])

# Compile the model
model.compile(optimizer='Adam',  # Adam optimizer for adaptive learning rate
              loss='categorical_crossentropy',  # Suitable loss for multi-class classification
              metrics=['accuracy'])  # Track accuracy during training

# Train the model
history = model.fit(X_train, Y_train,
                    epochs=20,  # Train for 20 full passes through the dataset
                    batch_size=32,  # Process 32 samples at a time
                    validation_data=(X_test, Y_test))  # Use test data for validation

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f'Test accuracy:{test_accuracy * 100:2f}%')  # Print final accuracy on test set

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


import numpy as np
index = 400
prediction = model.predict(X_test[index].reshape(1, 32, 32, 3))
predicted_class = np.argmax(prediction)
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
print(f"Predicted class: {class_names[predicted_class]}")
plt.imshow(X_test[index])
plt.show()




#CNN  
#Convolutional Neural Network  
#Image Classification  
#CIFAR-10  
#Deep_Learning  
#TensorFlow  
#Keras  
#Computer_Vision  
#Python  
#Image Recognition  
#Supervised_Learning  
#Multi-class_Classification  
#Machine_Learning  
#Data_Preprocessing  
#Neural_Networks  
#Model_Evaluation  
