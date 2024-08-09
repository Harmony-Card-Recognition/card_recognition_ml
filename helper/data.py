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
 
def process_original_dataframe(
    df,
    original_images,
    output_images,
    output_labels,
    num_distortions,
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
            img_path = os.path.join(original_images, row["filename"])
            img = Image.open(img_path)
        except UnidentifiedImageError:
            if verbose:
                print(f'Error: UnidentifiedImageError for {row["_id"]}')
            continue

        # Create distorted versions of the image
        for j in range(num_distortions):
            distorted_img = random_edit_img(img)
            distorted_img_filename = generate_unique_filename(
                output_images, f"{i}_distorted", "png"
            )
            distorted_img_path = os.path.join(output_images, distorted_img_filename)
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
        print(f"Saving labels to {output_labels} ...")
    labels_df.to_csv(output_labels, mode='a', index=False, header=not os.path.exists(output_labels))

    return labels_df

def populate_original_from_formatted_json(
    fp, verbose=True
):
    with open(fp["FORMATTED_JSON"], "r") as json_file:
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
            img_path = os.path.join(fp["ORIGINAL_IMAGES"], img_filename)
            img.save(img_path)
            df.at[i, "filename"] = img_filename
            df.at[i, "label"] = i 
        except UnidentifiedImageError:
            if verbose:
                print(f'Error: UnidentifiedImageError for {row["_id"]}')
            continue

    df.to_csv(fp["ORIGINAL_LABELS"], index=False)

def populate_datafolder_from_original(
    fp,
    verbose=False,
):
    df = pd.read_csv(fp["ORIGINAL_LABELS"])

    if verbose:
        print("Reading images from the original folder ...")

    process_original_dataframe(
        df, fp["ORIGINAL_IMAGES"], fp["TRAIN_IMAGES"], fp["TRAIN_LABELS"], 15, verbose
    )
    process_original_dataframe(
        df, fp["ORIGINAL_IMAGES"], fp["TEST_IMAGES"], fp["TEST_LABELS"], 5, verbose
    )

    if verbose:
        print("Finished creating the datasets!")

    print(f"\nUNIQUE CLASSES: {len(df['_id'].unique())}")

    # The function now returns the number of unique classes
    return len(df["_id"].unique())

def flush_original_data(fp):
    # remove all of the images
    for image in os.listdir(fp["ORIGINAL_IMAGES"]):
        os.remove(os.path.join(fp["ORIGINAL_IMAGES"], image))
    # remove all of the labels
    # this may need to be changed as we want to append to the file, not overwrite it
    os.remove(fp["ORIGINAL_LABELS"])