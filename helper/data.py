import os
import pandas as pd
import requests

from PIL import Image, UnidentifiedImageError
from io import BytesIO

from helper.image_processing import random_edit_img
from helper.helper import generate_unique_filename


def populate_images_and_labels_from_json(
    formatted_json_filepath: str,
    # model_filepath:str,
    train_labels_filepath: str,
    test_labels_filepath: str,
    train_images_filepath: str,
    test_images_filepath: str,
    verbose: bool = True,
):
    # create the data folder ONLY
    # append the data later

    """returns the train_images, test_images, train_labels, test_labels in that order"""
    # Load the JSON file into a DataFrame
    df = pd.read_json(formatted_json_filepath)

    column_names = ["label", "filename", "_id"]

    # Initialize lists to store the labels
    training_labels = []
    testing_labels = []

    testing_csv_filenames = []
    training_csv_filenames = []

    training_csv_ids = []
    testing_csv_ids = []

    # Create directories to store the training and testing images
    if verbose:
        print("Creating image directories ...")
    os.makedirs(train_images_filepath, exist_ok=True)
    os.makedirs(test_images_filepath, exist_ok=True)

    # Download the images and save them to files
    if verbose:
        print("Populating images ...")
    json_length = len(df)

    # when retraining the model, and there are more classes, you will need to change this
    # the unique index should reflect the last value in the csv -1 or something
    unique_index = (
        -1
    )  # I want this to start at 0, but the first thing the for loop does is add something, so this basically changes to 0
    previous_id = None
    for i, row in df.iterrows():
        if previous_id != row["_id"]:
            previous_id = row["_id"]
            unique_index += 1

        if verbose:
            print(f'Processing {row["_id"]} - image {i}/{json_length-1} ...')
        try:
            response = requests.get(row["image"])
            img = Image.open(BytesIO(response.content))
        except UnidentifiedImageError:
            if verbose:
                print(f'Error: UnidentifiedImageError for {row["_id"]}')
            continue
        # I DON'T THINK THAT THERE SHOULD BE OVERLAPPING IMAGES IN THE TESTING AND TRAINGING DATASETS
        # ONLY THE TESTING DATASET SHOULD HAVE THE ORIGINAL IMAGES TO PREVENT OVERFITTING
        # # Save the original image to a file, using the index as the filename
        # img_path = os.path.join(train_image_dir, f'{i}.png')
        # img.save(img_path)

        # # Add the label to the training labels list
        # training_labels.append(unique_index)
        # training_csv_filenames.append(f'{i}.png')
        # training_csv_ids.append(f'{row["_id"]}')

        # Create distorted versions of the image for training
        for j in range(15):
            distorted_img = random_edit_img(img)
            # distorted_img_path = os.path.join(train_image_dir, f'{i}_distorted({j}).png')
            distorted_img_filename = generate_unique_filename(
                train_images_filepath, f"{i}_distorted", "png"
            )
            distorted_img_path = os.path.join(
                train_images_filepath, distorted_img_filename
            )
            distorted_img.save(distorted_img_path)

            training_labels.append(unique_index)
            training_csv_filenames.append(distorted_img_filename)
            training_csv_ids.append(f'{row["_id"]}')

        # Save the original image to a file, using the index as the filename
        img_path = os.path.join(test_images_filepath, f"{i}.png")
        img.save(img_path)

        # Add the label to the training labels list
        testing_labels.append(unique_index)
        testing_csv_filenames.append(f"{i}.png")
        testing_csv_ids.append(f'{row["_id"]}')

        for j in range(2):
            distorted_img = random_edit_img(img)
            # distorted_img_path = os.path.join(test_image_dir, f'{i}_distorted({j}).png')
            distorted_img_filename = generate_unique_filename(
                test_images_filepath, f"{i}_distorted", "png"
            )
            distorted_img_path = os.path.join(
                test_images_filepath, distorted_img_filename
            )
            distorted_img.save(distorted_img_path)

            testing_labels.append(unique_index)
            testing_csv_filenames.append(distorted_img_filename)
            testing_csv_ids.append(f'{row["_id"]}')

    # Save the labels to CSV files
    if verbose:
        print("Saving labels as CSV ...")
    # Create a dictionary with specified column names
    data = {
        column_names[0]: testing_labels,
        column_names[1]: testing_csv_filenames,
        column_names[2]: testing_csv_ids,
    }
    test_labels_df = pd.DataFrame(data)
    data = {
        column_names[0]: training_labels,
        column_names[1]: training_csv_filenames,
        column_names[2]: training_csv_ids,
    }
    train_labels_df = pd.DataFrame(data)

    # test_labels_df = pd.DataFrame(testing_csv_filenames, columns=['label'])
    train_labels_df.to_csv(train_labels_filepath, index=False)
    test_labels_df.to_csv(test_labels_filepath, index=False)

    if verbose:
        print("Finished creating the datasets!")

    print(f"\nUNIQUE CLASSES: {unique_index + 1}")

    # The function now returns the paths to the image directories and the labels CSV files
    return unique_index + 1






# before
# formatted_json -> augment data -> populate training and testing image folders and label csv

# new 
# NEW STEP: (seperate function) populate original_folder from the formatted json
# original_folder (imagesfolder and label csv) -> augment data -> populate training and testing image folders and label csv




