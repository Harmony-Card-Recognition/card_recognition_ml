import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from preformat_data.file_format import get_images

# Define your image paths and labels
image_paths, labels = get_images()

# Load and preprocess images
images = []  # list of images
for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))  # resize to standard size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    image = image / 255.0  # normalize pixel values
    images.append(image)

# Convert to numpy arrays and reshape for TensorFlow
images = np.array(images).reshape(-1, 100, 100, 1)
labels = np.array(labels)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

# Create model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # adjust the number here to match the number of your classes
])

# Compile and train model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy}')

# Save model
model.save('card_recognition_model.h5')