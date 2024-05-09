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






mnist = tf.keras.datasets.fashion_mnist

# gives two sets of lists:
# training, and testing
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# plt.imshow(training_images[0])
# print(training_labels[0])
# print(training_images[0])

# normalizing the data from a range from 0 to 1 to make it easier to train (b&w images)
training_images  = training_images / 255.0
test_images = test_images / 255.0


# defining the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), # creates a 1D array of the image
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),  # adds a layer of neurons # Relu: "If X>0 return X, else return 0"
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)]) # picks the biggest value (the one that has the highest probability)


# compile the model
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

# trains the model
model.fit(training_images, training_labels, epochs=100)

# save the model
tf.keras.Model.save(model, './models/' + modelName + '.keras')

# test the model
model.evaluate(test_images, test_labels)
