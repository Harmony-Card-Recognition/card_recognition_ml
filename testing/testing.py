import os, sys
PROJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJ_PATH)

import tensorflow as tf
import numpy as np
import pandas as pd

from PIL import Image   

from helper.image_processing import load_image
   

def test_model_via_csv(csv_path, image_folder, model, img_width, img_height):
    df = pd.read_csv(csv_path)

    for index, row in df.iterrows():
        img_path = os.path.join(image_folder, row['filename'])
        label = row['label']
        test_model_via_index(img_path, label, model, img_width, img_height)

# def test_model_via_index(img_path, card_index, model, img_width, img_height):
#     test_card = load_image(img_path, img_width, img_height)
#     test_card = np.expand_dims(test_card, axis=0)

#     result = model.predict(test_card)
#     result_index, confidence = np.argmax(result), result[0,np.argmax(result)]

#     #display the result!
#     if result_index == card_index:
#         print(f'For card {os.path.basename(img_path)}, model predicted index {result_index} with {np.round(confidence*100,4)}% confidence.')
#     else:
#         print(f'For card {os.path.basename(img_path)}, model predicted index {result_index} with {np.round(confidence*100,4)}% confidence. (INCORRECT)')

def test_model_via_index(img_path, card_index, model, img_width, img_height, threshold=0.6):
    test_card = load_image(img_path, img_width, img_height)
    test_card = np.expand_dims(test_card, axis=0)

    result = model.predict(test_card)
    result_index, confidence = np.argmax(result), result[0,np.argmax(result)]

    # If the confidence is below the threshold, classify the image as 'unknown'
    if confidence < threshold:
        print(f'For card {os.path.basename(img_path)}, model predicted the image as unknown.')
    else:
        # Display the result!
        if result_index == card_index:
            print(f'For card {os.path.basename(img_path)}, model predicted index {result_index} with {np.round(confidence*100,4)}% confidence.')
        else:
            print(f'For card {os.path.basename(img_path)}, model predicted index {result_index} with {np.round(confidence*100,4)}% confidence. (INCORRECT)')

if __name__ == "__main__":
    model_folder = os.path.join(PROJ_PATH, '.data/harmony_0.0.7')
    model_path = os.path.join(model_folder, 'model.keras')
    image_folder = os.path.join(model_folder, 'test_images')
    labels_csv = os.path.join(model_folder, 'test_labels.csv')

    img = Image.open(f'{image_folder}/0.png')
    img_width, img_height = img.size

    model = tf.keras.models.load_model(model_path)
    test_model_via_csv(labels_csv, image_folder, model, img_width, img_height)


