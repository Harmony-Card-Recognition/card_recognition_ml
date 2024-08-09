import os, sys

PROJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJ_PATH)


import argparse
import datetime

from tensorflow.keras import callbacks, layers, models, optimizers, mixed_precision  # type: ignore

from helper.callbacks import (
    CsvLoggerCallback,
    ValidationAccuracyThresholdCallback,
    ClearMemory,
)
from helper.json_processing import format_json

# from helper.data import populate_images_and_labels
from helper.data import (
    populate_datafolder_from_original,
    populate_original_from_formatted_json,
    flush_original_data,
)
from helper.model_specs import pre_save_model_specs

from cnn import compile_model, fit_model
from filepaths import get_filepaths

def compile_argument_parser():
    parser = argparse.ArgumentParser(
        description="Creates CNN Models for Card Games", add_help=False
    )

    # Create a mutually exclusive group
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-C", "--create", action="store_true", help="Create a new model")
    group.add_argument(
        "-R", "--retrain", action="store_true", help="Retrain an existing model"
    )
    group.add_argument(
        "-E", "--expand", action="store_true", help="Expand an existing model"
    )

    parser.add_argument(
        "-v", "--version", type=str, help="xx.xx.xxx (for older versions try x.x.x)"
    )
    parser.add_argument(
        "-c",
        "--cardset",
        type=str,
        help="Based on the large json: ex) LorcanaCard or MTGCard",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")

    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="For those with room temperature IQ",
    )

    args = parser.parse_args()

    return args


def get_callbacks(fp):
    # defines when the model will stop training
    accuracy_threshold_callback = ValidationAccuracyThresholdCallback(threshold=0.98)

    # saves a snapshot of the model while it is training
    # note: there may be a huge performance difference if we chose to not include this callback... something to keep in mind
    checkpoint_filepath = os.path.join(fp["MODEL"], "checkpoint.keras")
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, save_weights_only=False, save_best_only=False
    )

    # logs the epoch, accuracy, and loss for a training session
    # note: removing this would also probably result in a performance increase
    csv_logger_callback = CsvLoggerCallback(
        os.path.join(fp["MODEL"], "training_logs.csv")
    )

    # Define the ReduceLROnPlateau callback
    reduce_lr_callback = callbacks.ReduceLROnPlateau(
        monitor="val_loss",  # Metric to monitor
        factor=0.2,  # Factor by which the learning rate will be reduced
        patience=5,  # Number of epochs with no improvement after which learning rate will be reduced
        min_lr=0.00001,  # Lower bound on the learning rate
    )

    clear_memory_callback = ClearMemory()

    return [accuracy_threshold_callback, checkpoint_callback, csv_logger_callback]


# different actions one can take
def create_new_model(
    learning_rate,
    beta_1,
    beta_2,
    metrics,
    loss,
    fp, 
    img_width,
    img_height,
    unique_classes,
    callbacks,
    verbose,
    epochs,
):
    # save some specs of the model that is being trained
    pre_save_model_specs(
        fp=fp, 
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
        img_width=img_width,
        img_height=img_height,
        fp=fp, 
        callbacks=callbacks,
        verbose=verbose,
        epochs=epochs,
    )

def retrain_existing_model(
    fp, 
    img_width,
    img_height,
    callbacks,
    verbose,
    epochs,
):
    # this is for when a client uses the model and gets new labled data
    # might as well capitalize on this and continue to train the model and make it better
    model = models.load_model(fp["KERAS_MODEL"])
    model = fit_model(
        model=model,
        img_width=img_width,
        img_height=img_height,
        fp=fp,
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


if __name__ == "__main__":
    # ===========================================
    # defines parameters (both defaults that don't change, and variables and flags)
    # defaults
    image_size = "large"
    inital_json_grab = 3  # -1 to get all of the objects in the json
    img_width, img_height = 100, 100  # 450, 650
    learning_rate = 0.0001
    beta_1 = 0.9
    beta_2 = 0.999
    metrics = ["accuracy"]
    loss = "sparse_categorical_crossentropy"

    # flags
    args = compile_argument_parser()

    # ===========================================
    # define names and filepaths
    # NAMES
    if args.version:
        version = args.version
    else:
        version = version = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    print(f"Model Version: {version}")

    model_name = args.cardset.upper()[:-4] + "_" + version
    large_json_name = "deckdrafterprod." + args.cardset

    # FILEPATHS
    fp = get_filepaths(args.cardset, model_name, large_json_name, inital_json_grab)

    # ===========================================
    # callbacks for fitting the model
    callbacks = get_callbacks(fp=fp)

    # ===========================================
    if args.create:
        print(f"Creating a new model from scratch")
        format_json(
            fp["RAW_JSON"], fp["FORMATTED_JSON"], inital_json_grab, image_size
        )
        populate_original_from_formatted_json(
            fp=fp,
            verbose=args.verbose,
        )
        unique_classes = populate_datafolder_from_original(
            fp=fp,
            verbose=args.verbose,
        )
        create_new_model(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            metrics=metrics,
            loss=loss,
            fp=fp,
            img_width=img_width,
            img_height=img_height,
            unique_classes=unique_classes,
            callbacks=callbacks,
            verbose=args.verbose,
            epochs=10000000000000,
        )
    elif args.retrain:
        print(f"Continuing to train a prexsisting model")

        unique_classes = populate_datafolder_from_original(
            fp=fp,
            verbose=args.verbose,
        )
        retrain_existing_model(
            fp=fp,
            img_width=img_width,
            img_height=img_height,
            callbacks=callbacks,
            verbose=args.verbose,
            epochs=10000000000000,
        )
    elif args.expand:
        print(f"Expanding the current model to hold more classes")
        expand_existing_model()


    # ensure that the original data folder is wiped for the next use (hopefully the training)
    # flush_original_data(fp=fp)

    print(f"Model Version: {version}")



# for the first time that a model is compiled, it should append the entire small json and its images to the original filepaths
# convert the json to images that will populate the original image folder and original csv file
# given and "original" image folder and csv file, augment the data into training and testing data
#   maybe at the same time, populate that augmented data to the training and testing filepath
#   remove the originals from the original image folder and original csv path
#   this will be the queue that we append to when we have new data that we want the model to train on
#
# there should be something in the model specs that highlights the
# unique classes
# which classes a model is able to read from the json
#   you gotta probably add the class number to a card type as well
# this will allow for classifications to be added later on
# I am getting ahead of myself
