import os
import shutil
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
import json
import random

from image_processing import random_edit_img




def get_datasets(json_filepath, model_filepath, verbose=True):
    """returns the train_images, test_images, train_labels, test_labels in that order"""
    # Load the JSON file into a DataFrame
    df = pd.read_json(json_filepath)

    # Initialize lists to store the labels
    training_labels = []
    testing_labels = []

    # Create directories to store the training and testing images
    if verbose: print('Creating image directories ...')
    train_image_dir = os.path.join(model_filepath, 'train_images')
    test_image_dir = os.path.join(model_filepath, 'test_images')
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)

    # Download the images and save them to files
    if verbose: print('Populating images ...')
    json_length = len(df)
    for index, row in df.iterrows():
        if verbose: print(f'\nProcessing {row["_id"]} - image{index}/{json_length} ...') 
        try:
            response = requests.get(row["image"])
            img = Image.open(BytesIO(response.content))
        except UnidentifiedImageError:
            if verbose: print(f'Error: UnidentifiedImageError for {row["_id"]}')
            continue

        # Save the original image to a file, using the index as the filename
        img_path = os.path.join(train_image_dir, f'{index}.png')
        img.save(img_path)

        # Add the label to the training labels list
        training_labels.append(row["_id"])

        # Create distorted versions of the image for training
        for i in range(4):
            distorted_img = random_edit_img(img)
            distorted_img_path = os.path.join(train_image_dir, f'{index}_distorted({i}).png')
            distorted_img.save(distorted_img_path)
            training_labels.append(row["_id"])
            # if verbose: print(f' distorted ({i})')

        # Save the original image to the testing images directory
        # WE CAN TAKE THIS OUT LATER IF WE WANT!!!!!!!!!!!!!!!!!!!!!
        # if verbose: print('Saving to testing directory ...')
        test_img_path = os.path.join(test_image_dir, f'{index}.png')
        img.save(test_img_path)
        testing_labels.append(row["_id"])

    # Convert the labels into numpy arrays and encode them as integers
    if verbose: print('Hashing labels to le.fit integers ...')
    le = LabelEncoder()
    training_labels = le.fit_transform(training_labels)
    testing_labels = le.fit_transform(testing_labels)
    
    # Save the labels to CSV files
    if verbose: print('Saving labels as CSV ...')
    train_labels_df = pd.DataFrame(training_labels, columns=['label'])
    test_labels_df = pd.DataFrame(testing_labels, columns=['label'])
    train_labels_df.to_csv(os.path.join(model_filepath, 'train_labels.csv'), index=False)
    test_labels_df.to_csv(os.path.join(model_filepath, 'test_labels.csv'), index=False)

    if verbose: print('Finished creating the datasets!')
    # The function now returns the paths to the image directories and the labels CSV files
    return train_image_dir, test_image_dir, os.path.join(model_filepath, 'train_labels.csv'), os.path.join(model_filepath, 'test_labels.csv')

def get_train_only_dataset(json_filepath, model_filepath, verbose=True):
    """returns the train_images, test_images, train_labels, test_labels in that order"""
    # Load the JSON file into a DataFrame
    df = pd.read_json(json_filepath)

    # Initialize lists to store the labels
    training_labels = []

    # Create directories to store the training and testing images
    if verbose: print('Creating image directories ...')
    train_image_dir = os.path.join(model_filepath, 'train_images')
    os.makedirs(train_image_dir, exist_ok=True)

    # Download the images and save them to files
    if verbose: print('Populating images ...')
    json_length = len(df)
    for index, row in df.iterrows():
        if verbose: print(f'\nProcessing {row["_id"]} - image{index}/{json_length} ...') 
        try:
            response = requests.get(row["image"])
            img = Image.open(BytesIO(response.content))
        except UnidentifiedImageError:
            if verbose: print(f'Error: UnidentifiedImageError for {row["_id"]}')
            continue

        # Save the original image to a file, using the index as the filename
        img_path = os.path.join(train_image_dir, f'{index}.png')
        img.save(img_path)

        # Add the label to the training labels list
        training_labels.append(row["_id"])


    # Convert the labels into numpy arrays and encode them as integers
    if verbose: print('Hashing labels to le.fit integers ...')
    le = LabelEncoder()
    training_labels = le.fit_transform(training_labels)
    
    # Save the labels to CSV files
    if verbose: print('Saving labels as CSV ...')
    train_labels_df = pd.DataFrame(training_labels, columns=['label'])
    train_labels_df.to_csv(os.path.join(model_filepath, 'train_labels.csv'), index=False)

    if verbose: print('Finished creating the datasets!')
    # The function now returns the paths to the image directories and the labels CSV files
    return train_image_dir, os.path.join(model_filepath, 'train_labels.csv')



# ==================================================
# HELPERS


def get_json_length(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return len(data)


def create_smaller_json(filepath, image_size=-1, verbose=True):
    if verbose: print(f"Copying {image_size} Objects ...\n")
    # Create a new file path for the smaller JSON file
    new_filepath = filepath.replace(".json", f"_small({image_size}).json")

    # Load the entire JSON file
    with open(filepath, "r", encoding="utf-8") as original_file:
        data = json.load(original_file)

    # get the specified # of data from the dataset
    if image_size == -1:
        if verbose: print(
            f'Copying ALL objects from "{filepath}" to "{new_filepath}" ...'
        )
        small_data = data
    else:
        if verbose: print(
            f'Copying {image_size} objects from "{filepath}" to "{new_filepath}" ...'
        )
        small_data = data[:image_size]

    # Write the small data to the new JSON file
    with open(new_filepath, "w", encoding="utf-8") as new_file:
        json.dump(small_data, new_file, indent=4)

    # Copy the original file's permissions to the new file
    shutil.copymode(filepath, new_filepath)

    if verbose: print("\nFinished Copying!")

    # Return the path of the new JSON file
    return new_filepath


def filter_attributes_json(filepath, attributes=["_id", "image_uris", "card_faces"], verbose=True):
    if verbose: print(f"Filtering {filepath} with only {attributes} ...")
    # Open the original JSON file and load the data
    with open(filepath, "r", encoding="utf-8") as original_file:
        data = json.load(original_file)

    # Filter the objects to only include the specified attributes
    filtered_data = [
        {attr: obj[attr] for attr in attributes if attr in obj} for obj in data
    ]

    # Write the filtered data back to the original JSON file
    with open(filepath, "w", encoding="utf-8") as original_file:
        json.dump(filtered_data, original_file, indent=4)

    if verbose: print("\nFinished filtering!")


def format_image_attributes(filepath, image_size="normal", verbose=True):
    # unique_ids = 0
    if verbose: print(f'Formatting {filepath} with {image_size} image size')

    # Load the JSON file
    with open(filepath, "r") as f:
        data = json.load(f)

    # new json object for duplicate card faces
    # list of dictionaries with two values ('_id' and image url)
    new_data = []

    # Add the attribute to each dictionary
    for json_object in data:
        if ("image_uris" in json_object): # use the first images that we see (these would probably be the best)
            new_face = {}
            new_face['_id'] = json_object["_id"] 
            new_face['image'] = json_object["image_uris"][image_size]
            new_data.append(new_face)
            # unique_ids += 1

        elif "card_faces" in json_object:
            # for each image that there is in the 'card_faces', create a new json object
            new_face_objects = []
            for face in json_object["card_faces"]:
                if "image_uris" in face:
                    # the face that we are currently on (or else this iteration will result in an object
                    # without an image)
                    
                    if new_face_objects.count == 0:
                        json_object['image'] = face["image_uris"][image_size]
                    else:
                        new_face = {}
                        new_face['_id'] = json_object["_id"] 
                        new_face['image'] = face["image_uris"][image_size]

                        new_face_objects.append(new_face)
            
            if new_face_objects.count != 0:
                new_data.extend(new_face_objects)
                # unique_ids += 1


                item_id = json_object["_id"]
                if verbose: print(f'({item_id}) DUPLICATE card faces added ...')
                

            else:
                if verbose: print(f'NO IMAGES FOUND IN CARDFACES [skipped] ...')

        else:
            # if there is no image found for the object, just skip it for now, and print a message
            item_id = json_object["_id"]
            if verbose: print(f"({item_id}) NO IMAGES FOUND [skipped] ...")
            


    # Write the modified data back to the JSON file
    with open(filepath, "w") as f:
        json.dump(new_data, f, indent=4)

    if verbose: print('Finished formatting!')

# ==================================================


def format_json(raw_json_filepath, small_json_size, image_size="normal", verbose=True):
    # create a smaller dataset (ideally with all of the images)
    if verbose: print('\n--- CREATING SEPERATE JSON ---')
    new_filepath = create_smaller_json(raw_json_filepath, small_json_size)

    # for each object in the json file, remove the everything but the '_id', 'image_uris', 'card_faces' attributes
    if verbose: print('\n--- FILTERINg JSON ---')
    filter_attributes_json(new_filepath)

    # convert the 'image_uris' and 'card_faces' to a universal 'image'
    if verbose: print('\n--- FORMATTING JSON ATTRIBUTES ---')
    format_image_attributes(new_filepath, image_size)

    if verbose: print('\n--- JSON FULLY FORMATTED ---')
    return new_filepath
