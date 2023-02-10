import numpy as np
import cv2
from sklearn.cluster import KMeans

# Load the training data and labels
train_data = np.load('vegetable_training_data.npy')
train_labels = np.load('vegetable_training_labels.npy')

# Initialize the K-Means clustering model
model = KMeans(n_clusters=len(np.unique(train_labels)))

# Fit the model to the training data
model.fit(train_data)

# Initialize the camera
camera = cv2.VideoCapture(0)

while True:
    # Capture an image from the camera
    ret, image = camera.read()

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to the image to segment the background from the foreground
    ret, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if len(contours) > 0:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Check if the largest contour is large enough to be a vegetable
        if cv2.contourArea(largest_contour) > 1000:
            # Draw a bounding box around the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop the image to just the region of interest (the largest contour)
            roi = image[y:y + h, x:x + w]

            # Reshape the ROI into a 1D feature vector
            features = roi.reshape(1, -1)

            # Use the trained model to make a prediction
            prediction = model.predict(features)

            # Print the prediction
            print("This is a(n) {}".format(np.unique(train_labels)[prediction[0]]))

    # Display the image
    cv2.imshow('Vegetable Detection', image)

    # Check if the 'q' key was pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
camera.release()

# Close all windows
cv2.destroyAllWindows()
