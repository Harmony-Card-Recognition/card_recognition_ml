import os, sys
PROJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJ_PATH)

import json
import numpy as np
import time
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score
from dataset import enqueue, dequeue
from helper.image_processing import random_edit_img, zoom_rotate_img, blur_img, adjust_color, adjust_contrast, adjust_sharpness, get_tensor_from_image, get_image_from_uri, get_img_dim
from sklearn.pipeline import Pipeline


def load_and_preprocess_testing(model_filepath: str, image_size: str, test_copies:int) -> tuple[np.array, np.array]:
    processed_path = os.path.join(model_filepath, "processed.json")
    X_test, y_test= [], [] 

    if os.path.exists(processed_path):
        with open(processed_path, 'r') as file:
            processed_data = json.load(file)
            for item in processed_data:
                img_width, img_height = get_img_dim(image_size)
                original_image = get_image_from_uri(item['image'])

                augment_functions = [random_edit_img]
                for _ in range(test_copies):
                    for f in augment_functions: 
                        edited_image = f(original_image)  # Assuming this function distorts the image
                        tensor = get_tensor_from_image(image=edited_image, img_width=img_width, img_height=img_height)
                        X_test.append(tensor.numpy().flatten())  # Flatten the image and append to X
                        y_test.append(item['encoded'])  # Append the same ID for each distorted version


    return np.array(X_test), np.array(y_test)

def load_and_preprocess_training(model_filepath: str, image_size: str, train_copies:int) -> tuple[np.array, np.array]:
    unprocessed_path = os.path.join(model_filepath, "unprocessed.json")
    X_train, y_train = [], [] 

    if os.path.exists(unprocessed_path):
        with open(unprocessed_path, 'r') as file:
            unprocessed_data = json.load(file)

            for item in unprocessed_data: 
                img_width, img_height = get_img_dim(image_size)
                original_image = get_image_from_uri(item['image'])

                augment_functions = [random_edit_img, zoom_rotate_img, blur_img, adjust_color, adjust_contrast, adjust_sharpness]
                for _ in range(train_copies):
                    for f in augment_functions: 
                        edited_image = f(original_image)  
                        tensor = get_tensor_from_image(image=edited_image, img_width=img_width, img_height=img_height)
                        X_train.append(tensor.numpy().flatten())  
                        y_train.append(item['encoded']) 

                dequeue(0, model_filepath)

    return np.array(X_train), np.array(y_train)

def train_model(model_name:str, image_size:str, train_copies:int, pipeline:Pipeline) -> None: 
    data = os.path.join(PROJ_PATH, '.data/knn')
    model_filepath = os.path.join(data, model_name)

    # Load and preprocess the data 
    X_train, y_train = load_and_preprocess_training(
        model_filepath=model_filepath, 
        image_size=image_size, 
        train_copies=train_copies, 
    )

    pipeline.fit(X_train, y_train)



