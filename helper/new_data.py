import os, sys
PROJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJ_PATH)

import pandas as pd
import requests
import json

from PIL import Image, UnidentifiedImageError
from io import BytesIO

from helper.image_processing import random_edit_img
from helper.helper import generate_unique_filename, alphanumeric_to_int
 
def process_images(
    df,
    images_folder,
    output_dir,
    num_distortions,
    output_csv_filepath,
    verbose=False,
):
    filenames = []
    ids = []
    labels = []

    column_names = ["label", "filename", "_id"]

    for i, row in df.iterrows():
        if verbose:
            print(f'Processing {row["_id"]} - image {i}/{len(df)-1} ...')
        try:
            img_path = os.path.join(images_folder, row["filename"])
            img = Image.open(img_path)
        except UnidentifiedImageError:
            if verbose:
                print(f'Error: UnidentifiedImageError for {row["_id"]}')
            continue

        # Create distorted versions of the image
        for j in range(num_distortions):
            distorted_img = random_edit_img(img)
            distorted_img_filename = generate_unique_filename(
                output_dir, f"{i}_distorted", "png"
            )
            distorted_img_path = os.path.join(output_dir, distorted_img_filename)
            distorted_img.save(distorted_img_path)

            filenames.append(distorted_img_filename)
            ids.append(f'{row["_id"]}')
            labels.append(f'{row["label"]}')

    # Create a dictionary with specified column names
    data = {
        # lookup the correct label based on the _id, (or variable ids). this should be a unique label
        column_names[0]: labels,  
        column_names[1]: filenames,
        column_names[2]: ids,
    }
    labels_df = pd.DataFrame(data)

    # Save the labels to CSV file
    if verbose:
        print(f"Saving labels to {output_csv_filepath} ...")
    labels_df.to_csv(output_csv_filepath, index=False)

    return labels_df


def original_from_formatted_json(
    formatted_json_filepath, original_filepath, verbose=True
):
    # this should be the place that the json wiht the key value paris should be created
    # there is no need for you to worry about adding new classes... if that happens, then we should recompile the model entirely
    # this will be stored in the .data/dataset/bidict.json

    original_images_folder = os.path.join(original_filepath, "images")
    original_csv_file = os.path.join(original_filepath, "labels.csv")

    with open(formatted_json_filepath, "r") as json_file:
        data = json.load(json_file)

    df = pd.DataFrame(data)
    json_length = len(df)
    df["label"] = None

    for i, row in df.iterrows():
        if verbose:
            print(f'Processing {row["_id"]} - image {i}/{json_length-1} ...')
        try:
            response = requests.get(row["image"])
            img = Image.open(BytesIO(response.content))
            img_filename = f'{row["_id"]}.png'
            # img_filename = f'{row["_id"]}_{i}.png'  # Ensure unique filename
            img_path = os.path.join(original_images_folder, img_filename)
            img.save(img_path)
            df.at[i, "filename"] = img_filename
            df.at[i, "label"] = i 
        except UnidentifiedImageError:
            if verbose:
                print(f'Error: UnidentifiedImageError for {row["_id"]}')
            continue

    df.to_csv(original_csv_file, index=False)


def create_datafolders_from_original(
    original_filepath,
    dataset_filepath,
    verbose=False,
):
    train_images_dir=os.path.join(dataset_filepath, 'train_images')
    test_images_dir=os.path.join(dataset_filepath, 'test_images')
    train_labels_dir=os.path.join(dataset_filepath, 'train_labels.csv')
    test_labels_dir=os.path.join(dataset_filepath, 'test_labels.csv')

    images_folder = os.path.join(original_filepath, "images")
    csv_file = os.path.join(original_filepath, "labels.csv")

    df = pd.read_csv(csv_file)

    if verbose:
        print("Reading images from the original folder ...")

    process_images(
        df, images_folder, train_images_dir, 15, train_labels_dir, verbose
    )
    process_images(
        df, images_folder, test_images_dir, 5, test_labels_dir, verbose
    )

    if verbose:
        print("Finished creating the datasets!")

    print(f"\nUNIQUE CLASSES: {len(df['_id'].unique())}")

    # The function now returns the number of unique classes
    return len(df["_id"].unique())
