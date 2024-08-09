import os, sys
PROJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJ_PATH)


import time 
import pandas as pd
import tensorflow as tf

from tensorflow.keras import callbacks, layers, models, optimizers, mixed_precision # type: ignore

from PIL import Image

from helper.image_processing import get_tensor_from_dir
from helper.helper import get_current_time, get_elapsed_time
from helper.model_specs import post_save_model_specs 


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
    dataset = dataset.map(lambda image, label: (get_tensor_from_dir(image, img_width, img_height), label))
    dataset = dataset.batch(batch_size)
    return dataset

def compile_model(
    unique_classes,
    img_width, 
    img_height, 
    learning_rate, 
    beta_1, 
    beta_2, 
    metrics,
    loss,
    verbose=True,
):
    # Define the model
    if verbose: print('Defining the model ...')
    model = models.Sequential()

    model.add(layers.InputLayer(shape=(img_width, img_height, 3)))
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
    model.add(layers.Dense(unique_classes, activation='softmax'))

    # Define the optimizer
    optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

def fit_model(
    model,
    img_width, 
    img_height, 
    fp,
    callbacks,
    verbose=True,
    batch_size=32,
    epochs=1000,
):
    # FITTING THE DATA 
    if verbose: print('Network compiled, fitting data now ... \n')
    if verbose: print('Creating the training and testing datasets ...')

    train_dataset = create_dataset(fp["TRAIN_LABELS"], fp["TRAIN_IMAGES"], img_width, img_height, batch_size=batch_size)
    test_dataset = create_dataset(fp["TEST_LABELS"], fp["TEST_IMAGES"], img_width, img_height, batch_size=batch_size)

    st = time.time() 
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
        print('\nModel fit, evaluating accuracy and saving locally now ... \n')
    loss, accuracy = model.evaluate(test_dataset)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    training_time = get_elapsed_time(st)
    post_save_model_specs(
        fp=fp,
        training_time=training_time,
        loss=loss,
        accuracy=accuracy,
        model=model,
    )

    # save it locally for future reuse
    model.save(os.path.join(fp["MODEL"], 'model.keras'))

    if verbose:
        print(f'\nModel evaluated & saved locally at {fp["MODEL"]}.keras on {get_current_time()} after {training_time}!\n')

    return model