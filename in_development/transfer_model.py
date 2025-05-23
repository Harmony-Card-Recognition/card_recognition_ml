# ===========================================================================================================
import tensorflow as tf

import os
import time as time
import pandas as pd
import tensorflow as tf

from PIL import Image
from keras import callbacks, layers, models, optimizers, applications
from helper.callbacks import CsvLoggerCallback, ValidationAccuracyThresholdCallback

from ..helper.image_processing import load_image
from ..helper.helper import get_current_time, get_elapsed_time
from ..helper.json_processing import format_json, get_datasets

def create_dataset(csv_file, image_dir, img_width, img_height, batch_size):
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Get the filenames and labels from the CSV file
    filenames = df['filename'].tolist()
    labels = df['label'].tolist()

    # Join the filenames with the directory path
    image_paths = [os.path.join(image_dir, f) for f in filenames]

    image_paths = tf.convert_to_tensor(image_paths)
    labels = tf.convert_to_tensor(labels)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda image, label: (load_image(image, img_width, img_height), label))
    dataset = dataset.batch(batch_size)
    return dataset

def train_new_CNN_model(
    model_filepath,
    train_image_dir,
    test_image_dir,
    train_labels_csv,
    test_labels_csv,
    unique_printings,
    callbacks,
    verbose=True,
    epochs=1000,
):
    """Help: Create and train a CNN model for the provided model_data"""
    model_start_time = time.time()

    img = Image.open(f'{train_image_dir}/0.png')
    img_width, img_height = img.size

    # Define the model
    if verbose: print('Defining the model ...')
    model = models.Sequential()
    model.add(layers.Input(shape=(img_width, img_height, 3)))
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(128, (3, 3)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(256, (3, 3)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(256, (3, 3)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(512, (3, 3)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Flatten())
    model.add(layers.Dense(2048))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(unique_printings, activation="softmax"))

    # Define the optimizer
    optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)

    # Compile the model
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    if verbose: print("Network compiled, fitting data now ... \n")

    return fit_model(
        model,
        model_filepath,
        train_image_dir,
        test_image_dir,
        train_labels_csv,
        test_labels_csv,
        callbacks,
        model_start_time=model_start_time,
        verbose=verbose,
        epochs=epochs,
    )

def fit_model(
    model,
    model_filepath,
    train_image_dir,
    test_image_dir,
    train_labels_csv,
    test_labels_csv,
    callbacks,
    model_start_time,
    verbose=True,
    epochs=1000,
):
    img = Image.open(f'{train_image_dir}/0.png')
    img_width, img_height = img.size

    # Load the labels from the CSV files
    if verbose: print('Loading the labels from the CSV files ...')
    training_labels = pd.read_csv(train_labels_csv)['label'].values
    testing_labels = pd.read_csv(test_labels_csv)['label'].values
    
    # Create tf.data.Dataset for the training and testing images
    if verbose: print('Creating the training and testing datasets ...')
    train_dataset = create_dataset(os.path.normpath(f'{model_filepath}/train_labels.csv'), os.path.normpath(train_image_dir), img_width, img_height, batch_size=32)
    test_dataset = create_dataset(os.path.normpath(f'{model_filepath}/test_labels.csv'), os.path.normpath(test_image_dir), img_width, img_height, batch_size=32)
    
    # Fit the model using the datasets
    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        callbacks=callbacks,
        verbose=verbose
    )

    # evaluate the model
    if verbose:
        print("\nModel fit, evaluating accuracy and saving locally now ... \n")
    loss, accuracy = model.evaluate(test_dataset)
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
if __name__ == "__main__":
    # ACTION:
        # 0 : you want to make a new model from scratch
            # downloads the images
            # processes the json
            # augments the images

        # 1 : you already have a folder with a model and you want to KEEP ON TRAINING IT
            # references the alredy compiled training and testing data
            # loads a checkpoint model
            # continues to fit the model

        # 2 : purely testing a pre-exsisting model
            # references the already compiled testing data
            # loads a checkpoint (or the end model)
            # tests the model 

    action = 0


    data = './.data/'
    model_name = 'harmony_0.0.0'
    # NOTE: this count tries to grab x number of cards (or unique classes)
    # however, there could be some image url's that are invalid (in which case this image is skipped)
    # and there could be card faces to one unique class (in which case there is another card with the same id added)
    inital_json_grab = 10   # -1 to get all of the objects in the json
    model_filepath = f'{data}{model_name}/'

    # =======================================
    # CALLBACKS

    accuracy_threshold_callback = ValidationAccuracyThresholdCallback(threshold=0.95)

    checkpoint_filepath = f'{model_filepath}model_checkpoint.keras'
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        save_best_only=False
    )

    csv_logger_callback = CsvLoggerCallback(f'{model_filepath}csv_logs.csv')

    # =======================================

        # print some preliminary information here
        # so that the log can log it lol

    if action == 0:
        # make new model FROM SCRATCH
        print('MAKING NEW MODEL FROM SCRATCH')

        raw_json_filepath = f'{data}/deckdrafterprod.MTGCard.json'
        formatted_json_filepath = format_json(raw_json_filepath, inital_json_grab, 'large')
        train_image_dir, test_image_dir, train_labels_csv, test_labels_csv, unique_classes= get_datasets(formatted_json_filepath, model_filepath)
        # train_image_dir, train_labels_csv = get_train_only_dataset(formatted_json_filepath, model_filepath)


        
        model = train_new_CNN_model(
            model_filepath,
            train_image_dir,
            test_image_dir,
            train_labels_csv,
            test_labels_csv,
            unique_classes,
            [accuracy_threshold_callback, checkpoint_callback, csv_logger_callback],
            verbose=True,
            epochs=10000000000000,
        )

    elif action == 1:
        # KEEP TRAINING EXSISTING MODEL
        print('CONTINUING TRAINING ON EXSISTING MODEL')

        formatted_json_filepath = f'{data}/deckdrafterprod.MTGCard_small({inital_json_grab}).json'
        train_image_dir = f'{model_filepath}train_images'
        test_image_dir = f'{model_filepath}test_images'
        train_labels_csv = f'{model_filepath}train_labels.csv'
        test_labels_csv = f'{model_filepath}test_labels.csv'

        # unique_printings = pd.read_csv(train_labels_csv)['label'].nunique()
        # print(f'Unique Classes: {unique_printings}')

        loaded_model = models.load_model(checkpoint_filepath)
        model = fit_model(
            loaded_model,
            model_filepath,
            train_image_dir,
            test_image_dir,
            train_labels_csv,
            test_labels_csv,
            model_start_time=time.time()
            [accuracy_threshold_callback, checkpoint_callback, csv_logger_callback],
            verbose=True,
            epochs=10000000000000,
        )
    
    elif action == 2:
        # Add code for testing a pre-existing model
        pass
    else:
        print("Invalid action value. Please choose a valid action.")

# ===========================================================================================================


def create_transfer_learning_model(input_shape, num_classes):
    # Load the pre-trained model
    base_model = applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create a new model on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Modify the train_new_CNN_model function to use the transfer learning model
def train_new_CNN_model_transfer_learning(
    model_filepath,
    train_image_dir,
    test_image_dir,
    train_labels_csv,
    test_labels_csv,
    unique_printings,
    callbacks,
    verbose=True,
    epochs=1000,
):
    model_start_time = time.time()

    img = Image.open(f'{train_image_dir}/0.png')
    img_width, img_height = img.size
    input_shape = (img_width, img_height, 3)

    # Use the create_transfer_learning_model function to get the model
    if verbose: print('Defining the model with transfer learning...')
    model = create_transfer_learning_model(input_shape, unique_printings)

    # The rest of the function remains the same...