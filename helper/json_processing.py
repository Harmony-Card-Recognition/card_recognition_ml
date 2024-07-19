import os
import shutil
import pandas as pd
import requests
import json
import hashlib

from PIL import Image, UnidentifiedImageError
from io import BytesIO
from copy import deepcopy

from helper.image_processing import random_edit_img
from sklearn.preprocessing import LabelEncoder


def get_datasets(json_filepath:str, model_filepath:str, verbose:bool=True):
    """returns the train_images, test_images, train_labels, test_labels in that order"""
    # Load the JSON file into a DataFrame
    df = pd.read_json(json_filepath)

    column_names = ['label', 'filename', "_id"]

    # Initialize lists to store the labels
    training_labels = []
    testing_labels = []

    testing_csv_filenames = []
    training_csv_filenames = []

    training_csv_ids = []
    testing_csv_ids = []

    # Create directories to store the training and testing images
    if verbose: print('Creating image directories ...')
    train_image_dir = os.path.join(model_filepath, 'train_images')
    test_image_dir = os.path.join(model_filepath, 'test_images')
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)

    # Download the images and save them to files
    if verbose: print('Populating images ...')
    json_length = len(df)

    unique_index = -1  # I want this to start at 0, but the first thing the for loop does is add something, so this basically changes to 0
    previous_id = None
    for i, row in df.iterrows():
        if previous_id != row["_id"]:
            previous_id = row["_id"]
            unique_index += 1

        if verbose: print(f'Processing {row["_id"]} - image {i}/{json_length-1} ...') 
        try:
            response = requests.get(row["image"])
            img = Image.open(BytesIO(response.content))
        except UnidentifiedImageError:
            if verbose: print(f'Error: UnidentifiedImageError for {row["_id"]}')
            continue

        # Save the original image to a file, using the index as the filename
        img_path = os.path.join(train_image_dir, f'{i}.png')
        img.save(img_path)

        # Add the label to the training labels list
        training_labels.append(unique_index)
        training_csv_filenames.append(f'{i}.png')
        training_csv_ids.append(f'{row["_id"]}')

        # Create distorted versions of the image for training
        for j in range(4):
            distorted_img = random_edit_img(img)
            distorted_img_path = os.path.join(train_image_dir, f'{i}_distorted({j}).png')
            distorted_img.save(distorted_img_path)

            training_labels.append(unique_index)
            training_csv_filenames.append(f'{i}.png')
            training_csv_ids.append(f'{row["_id"]}')


        # Save the original image to a file, using the index as the filename
        img_path = os.path.join(test_image_dir, f'{i}.png')
        img.save(img_path)

        # Add the label to the training labels list
        testing_labels.append(unique_index)
        testing_csv_filenames.append(f'{i}.png')
        testing_csv_ids.append(f'{row["_id"]}')

        # create a dataset for testing (or validation)
        # for i in range(1):
        #     distorted_img = random_edit_img(img)
        #     distorted_img_path = os.path.join(test_image_dir, f'{index}_distorted({i}).png')
        #     distorted_img.save(distorted_img_path)
        #     testing_labels.append(index)

        
    
    # Save the labels to CSV files
    if verbose: print('Saving labels as CSV ...')
    # Create a dictionary with specified column names
    data = {column_names[0]: testing_labels, column_names[1]: testing_csv_filenames, column_names[2]: testing_csv_ids}
    test_labels_df = pd.DataFrame(data)
    data = {column_names[0]: training_labels, column_names[1]: training_csv_filenames, column_names[2]: training_csv_ids}
    train_labels_df = pd.DataFrame(data)

    # test_labels_df = pd.DataFrame(testing_csv_filenames, columns=['label'])
    train_labels_df.to_csv(os.path.join(model_filepath, 'train_labels.csv'), index=False)
    test_labels_df.to_csv(os.path.join(model_filepath, 'test_labels.csv'), index=False)

    if verbose: print('Finished creating the datasets!')

    print(f'\nUNIQUE CLASSES: {unique_index + 1}')

        # The function now returns the paths to the image directories and the labels CSV files
    return train_image_dir, test_image_dir, os.path.join(model_filepath, 'train_labels.csv'), os.path.join(model_filepath, 'test_labels.csv'), unique_index+1

# ==================================================
# HELPERS

# def get_json_length(filepath):
#     with open(filepath, "r") as f:
#         data = json.load(f)
#     return len(data)


def create_smaller_json(json_filepath:str, new_filepath:str, image_count:int, verbose:bool=True):
    if verbose: print(f"Copying {image_count} Objects ...\n")

    # # Create a new file path for the smaller JSON file
    # d, f = json_filepath.rsplit('/', 1)
    # f = f.replace(".json", f"_small({image_count}).json")

    # new_filepath = os.path.join(model_filepath, f)

    # Load the entire JSON file
    with open(json_filepath, "r", encoding="utf-8") as original_file:
        data = json.load(original_file)

    # get the specified # of data from the dataset
    if image_count == -1:
        if verbose: print(
            f'Copying ALL objects from "{json_filepath}" to "{new_filepath}" ...'
        )
        small_data = data
    else:
        if verbose: print(
            f'Copying {image_count} objects from "{json_filepath}" to "{new_filepath}" ...'
        )
        small_data = data[:image_count]

    # Write the small data to the new JSON file
    with open(new_filepath, "w", encoding="utf-8") as new_file:
        json.dump(small_data, new_file, indent=4)

    # Copy the original file's permissions to the new file
    shutil.copymode(json_filepath, new_filepath)

    if verbose: print("\nFinished Copying!")

def filter_attributes_json(json_filepath:str, attributes:list[str], verbose:bool=True):
    if verbose: print(f"Filtering {json_filepath} with only {attributes} ...")
    # Open the original JSON file and load the data
    with open(json_filepath, "r", encoding="utf-8") as original_file:
        data = json.load(original_file)
    

    # Filter the objects to only include the specified attributes
    filtered_data = []

    for obj in data:
        new_obj = {}
        for attr in attributes:
            if attr in obj:
                new_obj[attr] = obj[attr]

        if 'card_faces' in obj:
            for face in obj['card_faces']:
                face_obj = deepcopy(new_obj)
                for attr in attributes:
                    if attr in face:
                        face_obj[attr] = face[attr]
                filtered_data.append(face_obj)
            continue 
        filtered_data.append(new_obj)


    # Write the filtered data back to the original JSON file
    with open(json_filepath, "w", encoding="utf-8") as original_file:
        json.dump(filtered_data, original_file, indent=4)

    if verbose: print("Finished filtering!")

def format_image_attributes(json_filepath:str, image_size:str, image_attribute_label:str, verbose:bool=True):
    if verbose: print(f'Formatting {json_filepath} with {image_size} image size')

    # Load the JSON file
    with open(json_filepath, "r") as f:
        data = json.load(f)

    filtered_data = []

    # Add the attribute to each dictionary
    for json_object in data:
        if image_attribute_label in json_object: # use the first images that we see (these would probably be the best)
            json_object['image'] = json_object[image_attribute_label][image_size]
            # delete the attribute that the json object has of the old image data
            json_object
            filtered_data.append(json_object)
 
        else:
            # if there is no image found for the object, just skip it for now, and print a message
            if verbose: print(f"({json_object['_id']}) NO IMAGES FOUND [removed from json] ...")

    # Write the modified data back to the JSON file
    with open(json_filepath, "w") as f:
        json.dump(filtered_data, f, indent=4)

    if verbose: print('Finished formatting!')




def encode_alphanumeric_to_int(json_filepath:str):
    # Load the JSON file
    with open(json_filepath, "r") as f:
        data = json.load(f)
    
    # Assign the index of each object in the JSON file as the 'encoded' attribute
    for index, json_object in enumerate(data):
        if "_id" in json_object:
            # Use the index as the 'encoded' value
            json_object["encoded"] = str(index)
    
    # Write the modified data back to the JSON file
    with open(json_filepath, "w") as f:
        json.dump(data, f, indent=4)

# ==================================================


def format_json(raw_json_filepath:str, new_filepath:str, image_count:int, image_size:str, attributes:list[str] = ['_id', 'images'], verbose:bool=True):
    if verbose: print('\n--- CREATING SEPERATE JSON ---')
    create_smaller_json(json_filepath=raw_json_filepath, new_filepath=new_filepath, image_count=image_count)

    if verbose: print('\n--- FILTERING JSON ---')
    filter_attributes_json(json_filepath=new_filepath, attributes=attributes)

    if verbose: print('\n--- FORMATTING JSON ATTRIBUTES ---')
    format_image_attributes(json_filepath=new_filepath, image_size=image_size, image_attribute_label='images')

    if verbose: print('\n--- JSON FULLY FORMATTED ---\n')
