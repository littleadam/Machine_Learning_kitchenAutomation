import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

# Load the data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test Accuracy:', test_acc)

# Function to predict if the object is clean or not clean
def predict_cleanliness(image):
    predictions = model.predict(np.array([image]))
    predicted_class = np.argmax(predictions[0])
    if predicted_class == 0:
        return 'Clean'
    else:
        return 'Not Clean'

# Use OpenCV to capture an image of the object
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

# Resize the image to 32x32
frame = cv2.resize(frame, (32, 32))

# Predict if the object is clean or not clean
print('The object is:', predict_cleanliness(frame))
