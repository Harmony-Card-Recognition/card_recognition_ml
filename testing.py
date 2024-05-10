import tensorflow as tf
import json
import numpy as np
from sklearn.model_selection import train_test_split

from data_formatting import from_json_to_trainingset

# PROCESSING THE JSON TO USEABLE TRAINING SETS
# Load the data
with open('./.data/small_dataset.json', 'r') as f:
    data = json.load(f)

# Get the length
if isinstance(data, list):
    length = len(data)
elif isinstance(data, dict):
    length = len(data['images'])  # replace 'images' with the key you're interested in
else:
    length = "Unknown"

# Convert to numpy arrays and reshape as needed
# This assumes each data point is a flattened image
images = np.array(data['images'])
labels = np.array(data['labels'])

# Reshape the images if they are flattened
images = images.reshape(-1, 480, 680, 3)

# Normalize the images
images = images / 255.0

# Split the data into training and test sets
training_images, test_images, training_labels, test_labels = train_test_split(images, labels, test_size=0.2)





# 
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.9):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

# DEFINE THE MODEL
# # Fully connected Network
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(512, activation=tf.nn.relu),
#   tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

# Convolutional Neural Network (CNN)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(480, 680, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])


# TRAIN THE MODEL
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])


# POST TRAINING ANALYSIS    
model.evaluate(test_images, test_labels)
model.save('./models/model.h5')
model.summary()
