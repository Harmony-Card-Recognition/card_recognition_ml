import os
import shutil
from time import time

import keras
from keras import callbacks, layers, models
import numpy as np

from helper import get_current_time, get_elapsed_time
from json_processing import format_json, get_datasets, get_json_length


def train_CNN_model(
    model_name,
    training_images,
    testing_images,
    training_labels,
    testing_labels,
    unique_printings,
    callbacks,
    verbose=True,
    epochs=1000,
):
    """Help: Create and train a CNN model for the provided model_data"""

    if verbose:
        print(f"Initializing {model_name} on {get_current_time()} ...")
        model_start_time = time()

    # if the folder already exists, delete it so we can start fresh
    if os.path.exists(f"{model_name}.model"):
        shutil.rmtree(f"{model_name}.model")

    model = models.Sequential()
    model.add(
        layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=training_images.shape[1:]
        )
    )
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(unique_printings, activation="softmax"))

    # Define the optimizer
    optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

    # compile the model
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    if verbose:
        print("Network compiled, fitting data now ... \n")
    # fit the model to the provided data
    model.fit(
        training_images,
        training_labels,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(testing_images, testing_labels),
    )

    if verbose:
        print("\nModel fit, elvaluating accuracy and saving locally now ... \n")
    # evaluate the model

    loss, accuracy = model.evaluate(testing_images, testing_labels)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")

    # save it locally for future reuse
    model.save(f"{model_name}.keras")

    if verbose:
        print(
            f"\nModel evaluated & saved locally at '{model_name}.model' on {get_current_time()} after {get_elapsed_time(model_start_time)}!\n"
        )

    return model

def test_model(model, testing_images, testing_labels):
    for i in range(len(testing_images)):
        img = testing_images[i]
        label = testing_labels[i]
        img = np.array([img])
        result = model.predict(img)
        prediction = np.argmax(result)
        confidence = result[0][prediction]
        print(f"Prediction: {prediction} | Actual: {label} | Confidence: {np.round(confidence*100,4)}")


# =======================================================


class AccuracyThresholdCallback(callbacks.Callback):
    def __init__(self, threshold):
        super(AccuracyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get("val_accuracy")
        if val_accuracy >= self.threshold:
            self.model.stop_training = True


# =======================================================


# here, you would format the raw json data that Trent has, and then make a formatted json file
raw_json_filepath = './.data/deckdrafterprod.MTGCard.json'
formatted_json_filepath = format_json(raw_json_filepath, -1)
# formatted_json_filepath = ".data/deckdrafterprod.MTGCard_small(50).json"

train_imgs, test_imgs, train_lbs, test_lbs = get_datasets(formatted_json_filepath)
model_name = "harmony_1.1.0"
unique_printings = get_json_length(formatted_json_filepath)

# Create a callback that stops training when accuracy reaches 98%
accuracy_threshold_callback = AccuracyThresholdCallback(threshold=0.95)

# =======================================================

model = train_CNN_model(
    model_name,
    train_imgs,
    test_imgs,
    train_lbs,
    test_lbs,
    unique_printings,
    callbacks=[accuracy_threshold_callback],
)


# model = models.load_model(f'{model_name}.keras')
# model.fit(train_imgs, train_lbs, epochs=epochs)