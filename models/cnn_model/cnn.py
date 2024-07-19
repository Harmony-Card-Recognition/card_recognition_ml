import os, sys
PROJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJ_PATH)


import time as time
import pandas as pd
import tensorflow as tf

from PIL import Image
from keras import callbacks, layers, models, optimizers, mixed_precision
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

def train_model(
    image_size,
    model_filepath,
    unique_classes,
    callbacks,
    verbose=True,
    epochs=1000,
):
    model_start_time = time.time()

    # img = Image.open(f'{train_image_dir}/0.png') # img.size
    img_width, img_height = get_img_dim(image_size)
    img = Image.open(f'{train_image_dir}/0.png')
    img_width, img_height = img.size

    img_width, img_height = 450, 650 

    # Define the model
    if verbose: print('Defining the model ...')
    # model = models.Sequential()
    # model.add(layers.Input(shape=(img_width, img_height, 3)))
    # model.add(layers.Conv2D(48, (3, 3)))  # Adjusted number of filters
    # model.add(layers.LeakyReLU(negative_slope=0.01))
    # model.add(layers.MaxPooling2D(2, 2))

    # model.add(layers.Conv2D(96, (3, 3)))  # Adjusted number of filters
    # model.add(layers.LeakyReLU(negative_slope=0.01))
    # model.add(layers.MaxPooling2D(2, 2))

    # # Retained this layer but adjusted filters
    # model.add(layers.Conv2D(192, (3, 3)))
    # model.add(layers.LeakyReLU(negative_slope=0.01))
    # model.add(layers.MaxPooling2D(2, 2))

    # # Removed one 256 filter layer to balance size and complexity
    # model.add(layers.Conv2D(384, (3, 3)))  # Adjusted number of filters
    # model.add(layers.LeakyReLU(negative_slope=0.01))
    # model.add(layers.MaxPooling2D(2, 2))

    # # Adjusted the dense layer size to balance the model
    # model.add(layers.Flatten())
    # model.add(layers.Dense(1024))  # Adjusted size
    # model.add(layers.LeakyReLU(negative_slope=0.01))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(unique_classes, activation='softmax'))
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
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
    learning_rate = 0.0001
    beta_1 = 0.9
    beta_2 = 0.999
    optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

    # Compile the model
    loss = 'sparse_categorical_crossentropy'
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # save some specs of the model that is being trained
    specs_filepath = os.path.join(model_filepath, 'model_specs.txt')
    with open(specs_filepath, 'w') as f:
        f.write(f'Model Name: {model_name}\n')
        f.write(f'Image Size: {image_size}\n')
        f.write(f'Initial JSON Grab: {inital_json_grab}\n')
        f.write(f'Unique Classes: {unique_classes}\n')
        f.write('\n')
        f.write(f'Learning Rate: {learning_rate}\n')
        f.write(f'Beta 1: {beta_1}\n') 
        f.write(f'Beta 2: {beta_2}\n') 
        f.write(f'Loss: {loss}\n') 
        f.write(f'metrics: {metrics}\n')
        f.write(f'Preprocessed Image Dimensions (wxh): {img_width}x{img_height}')
        f.write('\n')

    # FITTING THE DATA 
    if verbose: print('Network compiled, fitting data now ... \n')
    if verbose: print('Creating the training and testing datasets ...')
    train_dataset = create_dataset(os.path.normpath(f'{model_filepath}/train_labels.csv'), os.path.normpath(f'{model_filepath}/train_images/'), img_width, img_height, batch_size=32)
    test_dataset = create_dataset(os.path.normpath(f'{model_filepath}/test_labels.csv'), os.path.normpath(f'{model_filepath}/test_images/'), img_width, img_height, batch_size=32)
    
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

    # save some more model specs
    with open(specs_filepath, 'a') as f:
        f.write(f'Training Time: {get_elapsed_time(model_start_time)}\n')
        f.write(f'Loss: {loss}\n')
        f.write(f'Accuracy: {accuracy}\n')
        f.write('\n')
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    if verbose:
        print(f'\nModel evaluated & saved locally at {model_filepath}.keras on {get_current_time()} after {get_elapsed_time(model_start_time)}!\n')

    return model
    

# GPU OPTIMIZATION
def enable_gpu():
    # Enable mixed precision
    mp = mixed_precision.set_global_policy('mixed_float16')

    # Configure TensorFlow to use GPU efficiently
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print('GPU ERROR')
            print(e)


# =======================================================
if __name__ == '__main__':
    st = time.time()
    # enable_gpu()
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
    model_name = 'harmony_cnn_MTG_0.0.21'
    image_size = 'normal'
    inital_json_grab =  10 # -1 to get all of the objects in the json
    large_json_name = 'deckdrafterprod.MTGCard' # without the '.json'
        
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


    # =======================================

    if action == 0:
        # make new model FROM SCRATCH
        print('MAKING NEW MODEL FROM SCRATCH')
        
        raw_json_filepath = os.path.join(data, '..', f'{large_json_name}.json')
        formatted_json_filepath = os.path.join(model_filepath, f'{large_json_name}({inital_json_grab}).json')

        format_json(raw_json_filepath, formatted_json_filepath, inital_json_grab, image_size)
        train_image_dir, test_image_dir, train_labels_csv, test_labels_csv, unique_classes= get_datasets(formatted_json_filepath, model_filepath)

        
        model = train_model(
            image_size=image_size,
            model_filepath=model_filepath,
            unique_classes=unique_classes,
            callbacks=[accuracy_threshold_callback, checkpoint_callback, csv_logger_callback],
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

        # # unique_classes= pd.read_csv(train_labels_csv)['label'].nunique()
        # # print(f'Unique Classes: {unique_classes}')

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
    print(f'TOTAL TIME: {get_elapsed_time(st)}')
# ===========================================================================================================
