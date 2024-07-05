import os, sys
PROJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJ_PATH)


import time as time
import pandas as pd
import tensorflow as tf

from PIL import Image
from keras import callbacks, layers, models, optimizers
from helper.callbacks import CsvLoggerCallback, ValidationAccuracyThresholdCallback

from helper.image_processing import get_tensor_from_dir, get_img_dim
from helper.helper import get_current_time, get_elapsed_time
from helper.json_processing import format_json, get_datasets





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

def train_new_CNN_model(
    image_size,
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

    # img = Image.open(f'{train_image_dir}/0.png') # img.size
    img_width, img_height = get_img_dim(image_size)

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
    model.add(layers.Dense(unique_printings, activation='softmax'))

    # Define the optimizer
    optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)

    # Compile the model
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    if verbose: print('Network compiled, fitting data now ... \n')

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
        print('\nModel fit, evaluating accuracy and saving locally now ... \n')
    loss, accuracy = model.evaluate(test_dataset)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    # save it locally for future reuse
    model.save(os.path.join(model_filepath, 'model.keras'))

    if verbose:
        print(f'\nModel evaluated & saved locally at {model_filepath}.keras on {get_current_time()} after {get_elapsed_time(model_start_time)}!\n')

    return model

# =======================================================
if __name__ == '__main__':
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
    model_name = 'harmony_cnn_0.0.0'
    image_size = 'small'
    inital_json_grab = 10   # -1 to get all of the objects in the json

    if len(sys.argv) == 1:
        print('\nrunning with DEFAULT args\n')
    elif len(sys.argv) == 5:
        print('\nrunning with CUSTOM args\n')
        action = int(sys.argv[1])
        model_name = 'harmony_cnn_' + str(sys.argv[2])
        image_size = str(sys.argv[3])
        inital_json_grab = int(sys.argv[4])
    else:
        print('\n\nPLEASE CHECK ARGUMENTS')
        print('python(3) path_to_cnn/cnn.py [action] [model] [size] [count]')
        print('python(3) path_to_cnn/cnn.py 0 0.0.8 small 3')
        sys.exit()
        
    hello = 'hello'
    data = os.path.join(PROJ_PATH, '.data/cnn')
    model_filepath = os.path.join(data, model_name)
    os.makedirs(model_filepath)

    # =======================================
    # CALLBACKS

    accuracy_threshold_callback = ValidationAccuracyThresholdCallback(threshold=0.95)

    checkpoint_filepath = os.path.join(model_filepath, 'model_checkpoint.keras')
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        save_best_only=False
    )

    csv_logger_callback = CsvLoggerCallback(os.path.join(model_filepath, 'csv_logs.csv'))

    # =======================================

        # print some preliminary information here
        # so that the log can log it lol

    if action == 0:
        # make new model FROM SCRATCH
        print('MAKING NEW MODEL FROM SCRATCH')
        
        raw_json_filepath = os.path.join(data, '..', 'deckdrafterprod.MTGCard.json')
        formatted_json_filepath = os.path.join(model_filepath, f'deckdrafterprod.MTGCard_small({inital_json_grab}).json')

        format_json(raw_json_filepath, formatted_json_filepath, inital_json_grab, image_size)
        train_image_dir, test_image_dir, train_labels_csv, test_labels_csv, unique_classes= get_datasets(formatted_json_filepath, model_filepath)

        
        model = train_new_CNN_model(
            image_size,
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
        # # KEEP TRAINING EXSISTING MODEL
        # print('CONTINUING TRAINING ON EXSISTING MODEL')

        # formatted_json_filepath = f'{data}/deckdrafterprod.MTGCard_small({inital_json_grab}).json'
        # train_image_dir = f'{model_filepath}train_images'
        # test_image_dir = f'{model_filepath}test_images'
        # train_labels_csv = f'{model_filepath}train_labels.csv'
        # test_labels_csv = f'{model_filepath}test_labels.csv'

        # # unique_printings = pd.read_csv(train_labels_csv)['label'].nunique()
        # # print(f'Unique Classes: {unique_printings}')

        # loaded_model = models.load_model(checkpoint_filepath)
        # model = fit_model(
        #     loaded_model,
        #     model_filepath,
        #     train_image_dir,
        #     test_image_dir,
        #     train_labels_csv,
        #     test_labels_csv,
        #     model_start_time=time.time()
        #     [accuracy_threshold_callback, checkpoint_callback, csv_logger_callback],
        #     verbose=True,
        #     
        # epochs=10000000000000,
        # )
        pass
    else:
        print('Invalid action value. Please choose a valid action.')

# ===========================================================================================================
