import os, sys
PROJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJ_PATH)


import time 
import json
import pandas as pd
import tensorflow as tf

from tensorflow.keras import callbacks, layers, models, optimizers, mixed_precision # type: ignore

from PIL import Image
# from keras import callbacks, layers, models, optimizers, mixed_precision

from helper.callbacks import CsvLoggerCallback, ValidationAccuracyThresholdCallback
from helper.image_processing import get_tensor_from_dir, get_img_dim
from helper.helper import get_current_time, get_elapsed_time
from helper.json_processing import format_json, get_datasets
from helper.model_specs import pre_save_model_specs, post_save_model_specs

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

    # save some specs of the model that is being trained
    specs_filepath = os.path.join(model_filepath, 'model_specs.txt')
    pre_save_model_specs(
        specs_filepath=specs_filepath,
        model_name=model_name,
        image_size=image_size,
        inital_json_grab=inital_json_grab,
        unique_classes=unique_classes,
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        metrics=metrics,
        loss=loss,
        img_width=img_width,
        img_height=img_height,
    )
    return model

def fit_model(
    model,
    model_filepath,
    img_width, 
    img_height, 
    callbacks,
    verbose=True,
    epochs=1000,
):
    # FITTING THE DATA 
    if verbose: print('Network compiled, fitting data now ... \n')
    if verbose: print('Creating the training and testing datasets ...')
    train_dataset = create_dataset(os.path.normpath(f'{model_filepath}/train_labels.csv'), os.path.normpath(f'{model_filepath}/train_images/'), img_width, img_height, batch_size=32)
    test_dataset = create_dataset(os.path.normpath(f'{model_filepath}/test_labels.csv'), os.path.normpath(f'{model_filepath}/test_images/'), img_width, img_height, batch_size=32)

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


    specs_filepath = os.path.join(model_filepath, 'model_specs.txt')
    training_time = get_elapsed_time(st)
    post_save_model_specs(
        specs_filepath=specs_filepath,
        training_time=training_time,
        loss=loss,
        accuracy=accuracy,
        model=model,
    )

    # save it locally for future reuse
    model.save(os.path.join(model_filepath, 'model.keras'))


    if verbose:
        print(f'\nModel evaluated & saved locally at {model_filepath}.keras on {get_current_time()} after {training_time}!\n')

    return model

# =======================================================
if __name__ == '__main__':
    st = time.time()
    # ACTION:
        # 0 : you want to make a new model from scratch
            # downloads the images
            # processes the json
            # augments the images

        # 1 : you already have a folder with a model and you want to KEEP ON TRAINING IT
            # references the alredy compiled training and testing data
            # loads a checkpoint model
            # continues to fit the model

    action = 0
    model_name = 'LORCANA_0.1.0'
    image_size = 'large'
    inital_json_grab =  -1 # -1 to get all of the objects in the json
    large_json_name = 'deckdrafterprod.LorcanaCard' # without the '.json'
    img_width, img_height = 450, 650 
    learning_rate = 0.0001
    beta_1 = 0.9
    beta_2 = 0.999
    metrics = ['accuracy']
    loss = 'sparse_categorical_crossentropy'

    
    data = os.path.join(PROJ_PATH, '.data/cnn')
    model_filepath = os.path.join(data, model_name)
    if not os.path.exists(model_filepath):
        os.makedirs(model_filepath)
        
    formatted_json_filepath = os.path.join(model_filepath, f'{large_json_name}({inital_json_grab}).json')
    
    metadata_filepath = os.path.join(model_filepath, '.metadata.json') 
    with open(metadata_filepath, 'w') as file:
        json.dump([{'img_width': img_width, 'img_height': img_height}], file, indent=4)

    # =======================================
    # CALLBACKS
    # defines when the model will stop training
    accuracy_threshold_callback = ValidationAccuracyThresholdCallback(threshold=0.98)
    # saves a snapshot of the model while it is training
    # note: there may be a huge performance difference if we chose to not include this callback... something to keep in mind
    checkpoint_filepath = os.path.join(model_filepath, 'model_checkpoint.keras')
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        save_best_only=False
    )
    # logs the epoch, accuracy, and loss for a training session
    # note: removing this would also probably result in a performance increase
    csv_logger_callback = CsvLoggerCallback(os.path.join(model_filepath, 'csv_logs.csv'))  

    # =======================================

    if action == 0:
        # make new model FROM SCRATCH
        print('MAKING NEW MODEL FROM SCRATCH')
        
        raw_json_filepath = os.path.join(data, '..', f'{large_json_name}.json')

        format_json(raw_json_filepath, formatted_json_filepath, inital_json_grab, image_size)
        train_image_dir, test_image_dir, train_labels_csv, test_labels_csv, unique_classes = get_datasets(formatted_json_filepath, model_filepath)



        model = compile_model(
            unique_classes=unique_classes,
            img_width=img_width,
            img_height=img_height,
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            metrics=metrics,
            loss=loss,
            verbose=True,
        )
        model = fit_model(
            model=model,
            model_filepath=model_filepath,
            img_width=img_width,
            img_height=img_height,
            callbacks=[accuracy_threshold_callback, checkpoint_callback, csv_logger_callback],
            verbose=True,
            epochs=10000000000000,
        )

    elif action == 1:
        print('CONTINUING TRAINING ON EXSISTING MODEL') 
        # this takes in the last checkpoint
        # if the model crashed, then there is no 'final' model to start training again
        # if you did want to load this based on the final model from the previous training session, use the following
        # os.path.join(model_filepath, 'model.keras')
        model = models.load_model(checkpoint_filepath)

        train_image_dir, test_image_dir, train_labels_csv, test_labels_csv, unique_classes = get_datasets(formatted_json_filepath, model_filepath)

        # start the training process for the model 
        model = fit_model(
            model=model,
            model_filepath=model_filepath,
            img_width=img_width,
            img_height=img_height,
            callbacks=[accuracy_threshold_callback, checkpoint_callback, csv_logger_callback],
            verbose=True,
            epochs=10000000000000,
        )
    else:
        print('Invalid action value. Please choose a valid action.')
    print(f'TOTAL TIME: {get_elapsed_time(st)}')
# ===========================================================================================================
