import os
import time as time
from contextlib import redirect_stdout

from PIL import UnidentifiedImageError, Image
import keras
from keras import callbacks, layers, models, optimizers, preprocessing, utils
import numpy as np
import pandas as pd
import requests
from sklearn.calibration import LabelEncoder

from helper import get_current_time, get_elapsed_time
from json_processing import format_json, get_datasets, get_json_length, get_train_only_dataset
import os

from PIL import Image



def train_CNN_model(
    model_filepath,
    train_image_dir,
    ## test_image_dir,
    train_labels_csv,
    ## test_labels_csv,
    unique_printings,
    callbacks,
    verbose=True,
    epochs=1000,
    batch_size=32,
):
    """Help: Create and train a CNN model for the provided model_data"""
    model_start_time = time.time()


    img = Image.open(f'{train_image_dir}/0.png')
    img_width, img_height = img.size

    # Load the labels from the CSV files
    if verbose: print('Loading the labels from the CSV files ...')
    training_labels = pd.read_csv(train_labels_csv)['label'].values
    ## testing_labels = pd.read_csv(test_labels_csv)['label'].values
    
    # Create tf.data.Dataset for the training and testing images
    if verbose: print('Creating the training and testing datasets ...')
    train_dataset = preprocessing.image_dataset_from_directory(
        os.path.normpath(train_image_dir),
        labels=training_labels.tolist(),
        image_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=True
    )

    # # test_dataset = preprocessing.image_dataset_from_directory(
    # #     os.path.normpath(test_image_dir),
    # #     labels=testing_labels.tolist(),
    # #     image_size=(img_width, img_height),
    # #     batch_size=batch_size,
    # #     shuffle=False
    # # )

    # Define the model
    if verbose: print('Defining the model ...')
    model = models.Sequential()
    model.add(layers.Input(shape=(img_width, img_height, 3)))  # Use an Input layer to specify the input shape
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation="relu")) # added another dropout layer
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(unique_printings, activation="softmax"))

    # Define the optimizer
    optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)

    # Compile the model
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    if verbose: print("Network compiled, fitting data now ... \n")

    # Fit the model using the datasets
    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=train_dataset,
        callbacks=callbacks,
        verbose=verbose
    )

    # evaluate the model
    if verbose:
        print("\nModel fit, evaluating accuracy and saving locally now ... \n")
    loss, accuracy = model.evaluate(train_dataset)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")

    # save it locally for future reuse
    model.save(f"{model_filepath}model.keras")

    if verbose:
        print(
            f"\nModel evaluated & saved locally at '{model_filepath}.keras' on {get_current_time()} after {get_elapsed_time(model_start_time)}!\n"
        )

    return model




# =======================================================

# can stop at val_accuracy, or actual accuracy
class AccuracyThresholdCallback(callbacks.Callback):
    def __init__(self, threshold):
        super(AccuracyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get("accuracy")
        if accuracy >= self.threshold:
            self.model.stop_training = True


# =======================================================
if __name__ == "__main__":
    data = './.data/'
    model_name = 'harmony_1.4.0'
    model_input_size = -1   # -1 to get all of the objects in the json
    model_filepath = f'{data}{model_name}/'

    # # IF THIS IS THE FIRST TIME MAKING A PARTICULAR MODEL
    # raw_json_filepath = f'{data}/deckdrafterprod.MTGCard.json'
    # formatted_json_filepath = format_json(raw_json_filepath, model_input_size, 'small')
    # # train_image_dir, test_image_dir, train_labels_csv, test_labels_csv = get_datasets(formatted_json_filepath, model_filepath)
    # train_image_dir, train_labels_csv = get_train_only_dataset(formatted_json_filepath, model_filepath)


    # IF YOU ALREADY PROCESSED THE JSON AND HAVE THE IMAGES AND LABELS
    formatted_json_filepath = f'{data}/deckdrafterprod.MTGCard_small({model_input_size}).json'
    train_image_dir = f'{model_filepath}train_images'
    # test_image_dir = f'{model_filepath}test_images'
    train_labels_csv = f'{model_filepath}train_labels.csv'
    # test_labels_csv = f'{model_filepath}test_labels.csv'
    
    # Create a callback that stops training when accuracy reaches 98%
    accuracy_threshold_callback = AccuracyThresholdCallback(threshold=0.99)

    unique_printings = pd.read_csv(train_labels_csv)['label'].nunique()

    checkpoint_filepath = f'{model_filepath}model_checkpoint.keras'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    # =======================================================

    model = train_CNN_model(
        model_filepath,
        train_image_dir,
        # # test_image_dir,
        train_labels_csv,
        # # test_labels_csv,
        unique_printings,
        [accuracy_threshold_callback, checkpoint_callback],
        verbose=True,
        epochs=10000000000000,
        batch_size=100,
    )



