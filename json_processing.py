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


def get_datasets(json_filepath):
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
            print(f'Error: UnidentifiedImageError for {row["_id"]}')
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


def create_smaller_json(filepath, image_size=-1, rand=False):
    print(f"Copying {image_size} Objects ...\n")
    # Create a new file path for the smaller JSON file
    new_filepath = filepath.replace(".json", f"_small({image_size}).json")

    # Load the entire JSON file
    with open(filepath, "r", encoding="utf-8") as original_file:
        data = json.load(original_file)

    # If random is True, select 'image_size' random objects from the data
    if rand and image_size == -1:
        print(
            f'Copying the FIRST {image_size} objects from "{filepath}" to "{new_filepath}" ...\n'
        )
        small_data = random.sample(data, image_size)
    else:
        print(
            f'Copying {image_size} RANDOM (unique) objects from "{filepath}" to "{new_filepath}" ...\n'
        )
        small_data = data[:image_size]

    # Write the small data to the new JSON file
    with open(new_filepath, "w", encoding="utf-8") as new_file:
        json.dump(small_data, new_file, indent=4)

    # Copy the original file's permissions to the new file
    shutil.copymode(filepath, new_filepath)

    print("\nFinished Copying!")

    # Return the path of the new JSON file
    return new_filepath


def filter_attributes_json(filepath, attributes=["_id", "image_uris", "card_faces"]):
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

    print("\nFinished filtering!")


def format_image_attributes(filepath, image_size="normal"):
    # Load the JSON file
    with open(filepath, "r") as f:
        data = json.load(f)

    additional_faces_object = {}

    # Add the attribute to each dictionary
    for item in data:
        if (
            "image_uris" in item
        ):  # Replace 'attribute_to_delete' with the name of the attribute you want to delete
            image_url = item["image_uris"][image_size]
            del item["image_uris"]
            if "card_faces" in item:
                del item["card_faces"]
        elif "card_faces" in item:
            # for each image that there is in the 'card_faces', create a new json object
            # FOR NOW i am lazy, and Im just going to use the first instance of an image
            for face in item["card_faces"]:
                if "image_uris" in item:
                    image_url = face["image_uris"][image_size]
                    break

            del item["card_faces"]
        else:
            itemid = item["_id"]
            print(f"\n\n {itemid} NO IMAGES FOUND...\n\n")
            continue

        item["image"] = image_url

    # Write the modified data back to the JSON file
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


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


def format_json(raw_json_filepath, small_json_size):
    # create a smaller dataset (ideally with all of the images)
    new_filepath = create_smaller_json(raw_json_filepath, small_json_size)

    # for each object in the json file, remove the everything but the '_id', 'image_uris', 'card_faces' attributes
    filter_attributes_json(new_filepath)

    # convert the 'image_uris' and 'card_faces' to a universal 'image'
    format_image_attributes(new_filepath)

    return new_filepath
