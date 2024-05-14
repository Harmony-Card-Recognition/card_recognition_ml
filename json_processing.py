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


def get_datasets(json_filepath, verbose=True):
    """returns the train_images, test_images, train_labels, test_labels in that order"""
    # Load the JSON file into a DataFrame
    df = pd.read_json(json_filepath)

    # Initialize lists to store the image data and labels
    training_images = []
    training_labels = []

    testing_images = []
    testing_labels = []

    # Download the images and convert them into numpy arrays
    for _, row in df.iterrows():
        try:
            response = requests.get(row["image"])
            img = Image.open(BytesIO(response.content))
        except UnidentifiedImageError:
            if verbose: print(f'Error: UnidentifiedImageError for {row["_id"]} [skipped]')
            continue
        img_array = np.array(img)
        training_images.append(img_array)
        training_labels.append(row["_id"])

        for _ in range(5):
            distorted_img = random_edit_img(img)
            distorted_img_array = np.array(distorted_img)
            training_images.append(distorted_img_array)
            training_labels.append(
                row["_id"]
            )  # The label for the distorted image is the same as the original image

        # create n different UNSEEN variants of the 'perfect' cards for testing
        for _ in range(2):
            distorted_img = random_edit_img(img)
            distorted_img_array = np.array(distorted_img)
            testing_images.append(distorted_img_array)
            testing_labels.append(
                row["_id"]
            )  # The label for the distorted image is the same as the original image

    # Convert TRAINING LISTS into numpy arrays
    training_images = np.array(training_images)
    # labels = np.array(labels) # this gave an issue because the labels must be ingegers... not strings
    le = LabelEncoder()  # Initialize the label encoder
    training_labels = le.fit_transform(
        training_labels
    )  # Convert the labels into numpy arrays and encode them as integers

    # Convert TESTING LISTS into numpy arrays
    testing_images = np.array(training_images)
    # labels = np.array(labels) # this gave an issue because the labels must be ingegers... not strings
    le = LabelEncoder()  # Initialize the label encoder
    testing_labels = le.fit_transform(
        training_labels
    )  # Convert the labels into numpy arrays and encode them as integers

    # # Split the data into training and testing sets
    # # this is wrong becuase it can seperate all variations of the same card to the testing set, so the model doesn't have a label/bucket for that card
    # train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2)

    return training_images, testing_images, training_labels, testing_labels


# ==================================================
# HELPERS


def get_json_length(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return len(data)


def create_smaller_json(filepath, image_size=-1, rand=False, verbose=True):
    if verbose: print(f"Copying {image_size} Objects ...")
    # Create a new file path for the smaller JSON file
    new_filepath = filepath.replace(".json", f"_small({image_size}).json")

    # Load the entire JSON file
    with open(filepath, "r", encoding="utf-8") as original_file:
        data = json.load(original_file)

    # If random is True, select 'image_size' random objects from the data
    if rand and image_size == -1:
        print(
            f'Copying the FIRST {image_size} objects from "{filepath}" to "{new_filepath}" ...'
        )
        small_data = random.sample(data, image_size)
    else:
        print(
            f'Copying {image_size} RANDOM (unique) objects from "{filepath}" to "{new_filepath}" ...'
        )
        small_data = data[:image_size]

    # Write the small data to the new JSON file
    with open(new_filepath, "w", encoding="utf-8") as new_file:
        json.dump(small_data, new_file, indent=4)

    # Copy the original file's permissions to the new file
    shutil.copymode(filepath, new_filepath)

    if verbose: print("Finished Copying!")

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

    if verbose: print("Finished filtering!")


def format_image_attributes(filepath, image_size="normal", verbose=True):
    if verbose: print(f'Formatting {filepath} with {image_size} image size')

    # Load the JSON file
    with open(filepath, "r") as f:
        data = json.load(f)

    # new json object for duplicate card faces
    # list of dictionaries with two values ('_id' and image url)
    duplicate_face_objects = []

    # Add the attribute to each dictionary
    for json_object in data:
        if (
            "image_uris" in json_object
        ): # use the first images that we see (these would probably be the best)
            image_url = json_object["image_uris"][image_size]
            del json_object["image_uris"]

            # I don't know if this will cause issues, but take these out if we already have an image
            if "card_faces" in json_object:
                del json_object["card_faces"]

            json_object["image"] = image_url

        elif "card_faces" in json_object:
            # for each image that there is in the 'card_faces', create a new json object
            new_face_objects = []
            for i, face in enumerate(json_object["card_faces"]):
                if "image_uris" in face:
                    # the face that we are currently on (or else this iteration will result in an object
                    # without an image)
                    if i == 0:
                        json_object['image'] = face["image_uris"][image_size]
                    else:
                        new_face = {}
                        new_face['_id'] = json_object["_id"] 
                        new_face['image'] = face["image_uris"][image_size]

                        new_face_objects.append(new_face)
            
            if new_face_objects.count != 0:
                duplicate_face_objects.extend(new_face_objects)

                item_id = json_object["_id"]
                if verbose: print(f'({item_id}) DUPLICATE card faces added ...')

                del json_object["card_faces"]
            else:
                if verbose: print(f'NO IMAGES FOUND IN CARDFACES [skipped] ...')

        else:
            # if there is no image found for the object, just skip it for now, and print a message
            item_id = json_object["_id"]
            if verbose: print(f"({item_id}) NO IMAGES FOUND [skipped] ...")

    # append the duplicate card faces to the end of the json file
    data.extend(duplicate_face_objects)

    # Write the modified data back to the JSON file
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

    if verbose: print('Finished formatting!')


def get_random_test_datasets(filepath, count=10):
    # Load the JSON file
    with open(filepath, "r") as f:
        data = json.load(f)

    random_test_data = random.sample(data, count)
    testing_images = []
    testing_labels = []

    for obj in random_test_data:
        response = requests.get(obj["image"])
        img = Image.open(BytesIO(response.content))
        img_array = np.array(img)
        testing_images.append(img_array)
        testing_labels.append(obj["_id"])

    testing_images = np.array(testing_images)
    le = LabelEncoder()
    testing_labels = le.fit_transform(testing_labels)

    return testing_images, testing_labels


# ==================================================


def format_json(raw_json_filepath, small_json_size, verbose=True):
    # create a smaller dataset (ideally with all of the images)
    if verbose: print('\n--- CREATING SEPERATE JSON ---')
    new_filepath = create_smaller_json(raw_json_filepath, small_json_size)

    # for each object in the json file, remove the everything but the '_id', 'image_uris', 'card_faces' attributes
    if verbose: print('\n--- FILTERINg JSON ---')
    filter_attributes_json(new_filepath)

    # convert the 'image_uris' and 'card_faces' to a universal 'image'
    if verbose: print('\n--- FORMATTING JSON ATTRIBUTES ---')
    format_image_attributes(new_filepath)

    if verbose: print('\n--- JSON FULLY FORMATTED ---')
    return new_filepath
