import os, sys
PROJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJ_PATH)


import json
import argparse

from tensorflow.keras import callbacks, layers, models, optimizers, mixed_precision # type: ignore

from PIL import Image
# from keras import callbacks, layers, models, optimizers, mixed_precision

from helper.callbacks import CsvLoggerCallback, ValidationAccuracyThresholdCallback, ClearMemory
from helper.image_processing import get_tensor_from_dir, get_img_dim
from helper.helper import get_current_time, get_elapsed_time
from helper.json_processing import format_json, get_datasets
from helper.model_specs import pre_save_model_specs, post_save_model_specs

from cnn import compile_model, fit_model, create_dataset


def compile_argument_parser():
    parser = argparse.ArgumentParser(description = 'Creates CNN Models for Card Games', add_help=False)

    # Create a mutually exclusive group
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-C', '--create', 
        action='store_true', 
        help='Create a new model'
    )
    group.add_argument(
        '-R', '--retrain', 
        action='store_true', 
        help='Retrain an existing model'
    )
    group.add_argument(
        '-E', '--expand', 
        action='store_true', 
        help='Expand an existing model'
    )
    
    parser.add_argument(
        '-v', '--version',
        type=str,
        help='xx.xx.xxx (for older versions try x.x.x)'
    )
    parser.add_argument(
        '-c', '--cardset',
        type=str,
        help='Based on the large json: ex) LorcanaCard or MTGCard'
    )

    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='Enable verbose mode'
    )

    parser.add_argument(
        '-h','--help', 
        action='help',
        default=argparse.SUPPRESS,
        help='For those with room temperature IQ'
    )


    args = parser.parse_args()

    return args

def create_new_model(
        learning_rate,
        beta_1,
        beta_2,
        metrics,
        loss,
        model_filepath,
        img_width,
        img_height,
        callbacks,
        verbose,
        epochs,
    ):

    # NOTE THIS IS SOMETHING THAT YOU ARE PROBABLY GOING TO CHANGE LATER ON...
    data = os.path.join(PROJ_PATH, '.data/cnn', args.cardset)
    raw_json_filepath = os.path.join(data, '..', '..', f'{large_json_name}.json')

    format_json(raw_json_filepath, formatted_json_filepath, inital_json_grab, image_size)
    train_image_dir, test_image_dir, train_labels_csv, test_labels_csv, unique_classes = get_datasets(
        formatted_json_filepath, 
        model_filepath
    )

    # train_image_dir = os.path.join(model_filepath, '..', 'lorcana_images', 'train_images')
    # test_image_dir = os.path.join(model_filepath, '..', 'lorcana_images', 'test_images')
    # train_labels_csv = os.path.join(model_filepath, '..', 'lorcana_images', 'train_labels.csv')
    # test_labels_csv = os.path.join(model_filepath, '..', 'lorcana_images', 'test_labels.csv')
    # unique_classes = 993 

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

    model = compile_model(
        unique_classes=unique_classes,
        img_width=img_width,
        img_height=img_height,
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        metrics=metrics,
        loss=loss,
        verbose=verbose,
    )
    model = fit_model(
        model=model,
        model_filepath=model_filepath,
        img_width=img_width,
        img_height=img_height,
        callbacks=callbacks,
        verbose=verbose,
        epochs=epochs,
    )
 
def retrain_existing_model(
        model_filepath,
        img_width,
        img_height,
        callbacks,
        verbose,
        epochs,
    ):
    # adds to the current dataset (without adding new classifications)
    # keeps the current model

    # this is for when a client uses the model and gets new labled data
    # might as well capitalize on this and continue to train the model and make it better
    model = fit_model(
        model=model,
        model_filepath=model_filepath,
        img_width=img_width,
        img_height=img_height,
        callbacks=callbacks,
        verbose=verbose,
        epochs=epochs,
    )

def expand_existing_model():
    # adds to the current dataset (with new classifications) 
    # creates a new model

    # this is for when a client uses the model and gets new labled data that CANNOT be classified by the current model
    # you are going to have to change the structure of the model
    # this may get complicated to be honest
    # you might have to expand the hidden layers of the model as well as the output of the model
    # you should also only do this periodically
    pass

def get_callbacks(model_filepath: str): 
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
    csv_logger_callback = CsvLoggerCallback(os.path.join(model_filepath, 'training_logs.csv'))  

    # Define the ReduceLROnPlateau callback
    reduce_lr_callback = callbacks.ReduceLROnPlateau(
        monitor='val_loss',  # Metric to monitor
        factor=0.2,          # Factor by which the learning rate will be reduced
        patience=5,          # Number of epochs with no improvement after which learning rate will be reduced
        min_lr=0.00001       # Lower bound on the learning rate
    )

    clear_memory_callback = ClearMemory()

    return [accuracy_threshold_callback, checkpoint_callback, csv_logger_callback, reduce_lr_callback, clear_memory_callback] 


if __name__ == "__main__":
    # defaults that will rarely change
    image_size = 'large'
    inital_json_grab =  3# -1 to get all of the objects in the json
    img_width, img_height = 450, 650 
    learning_rate = 0.0001
    beta_1 = 0.9
    beta_2 = 0.999
    metrics = ['accuracy']
    loss = 'sparse_categorical_crossentropy'
    
    args = compile_argument_parser()

    model_name = args.cardset.upper()[:-4] + '_' + args.version
    large_json_name = 'deckdrafterprod.' + args.cardset

    # this is where the models and data are being stored
    data = os.path.join(PROJ_PATH, '.data/cnn', args.cardset)
    model_filepath = os.path.join(data, model_name)
    if not os.path.exists(model_filepath):
        os.makedirs(model_filepath)

    formatted_json_filepath = os.path.join(model_filepath, f'{large_json_name}({inital_json_grab}).json')

    callbacks = get_callbacks(model_filepath=model_filepath)

    if args.verbose:
        print('Verbose mode enabled')

    if args.create: 
        print(f'Creating a new model from scratch')
        create_new_model(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            metrics=metrics,
            loss=loss,
            model_filepath=model_filepath,
            img_width=img_width,
            img_height=img_height,
            callbacks=callbacks,
            verbose=args.verbose,
            epochs=10000000000000,
        )
    elif args.retrain:
        print(f'Continuing to train a prexsisting model')
        retrain_existing_model(
            model_filepath=model_filepath,
            img_width=img_width,
            img_height=img_height,
            callbacks=callbacks,
            verbose=args.verbose,
            epochs=10000000000000,
        )
    elif args.expand:
        print(f'Expanding the current model to hold more classes')
        expand_existing_model()

    print('yay, you did something and the main script executed')
    

    
        


    # =======================================

    # elif action == 1:
    #     print('CONTINUING TRAINING ON EXSISTING MODEL') 
    #     # this takes in the last checkpoint
    #     # if the model crashed, then there is no 'final' model to start training again
    #     # if you did want to load this based on the final model from the previous training session, use the following
    #     # os.path.join(model_filepath, 'model.keras')
    #     model = models.load_model(checkpoint_filepath)

    #     train_image_dir, test_image_dir, train_labels_csv, test_labels_csv, unique_classes = get_datasets(formatted_json_filepath, model_filepath)

    #     # start the training process for the model 
    #     model = fit_model(
    #         model=model,
    #         model_filepath=model_filepath,
    #         img_width=img_width,
    #         img_height=img_height,
    #         callbacks=[accuracy_threshold_callback, checkpoint_callback, csv_logger_callback],
    #         verbose=True,
    #         epochs=10000000000000,
    #     )