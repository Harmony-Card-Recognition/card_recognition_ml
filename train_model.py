import tensorflow as tf

mnist = tf.keras.datasets.fashion_mnist

# gives two sets of lists:
# training, and testing
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()


# normalizing the data from a range from 0 to 1 to make it easier to train
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
model.fit(training_images, training_labels, epochs=10)

# save the model
# tf.keras.Model.save(model, './models/' + modelName + '.keras')

# test the model
model.evaluate(test_images, test_labels)
