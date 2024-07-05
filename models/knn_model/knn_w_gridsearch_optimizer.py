import os, sys
PROJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJ_PATH)

import json
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score
from dataset import enqueue, dequeue
from helper.image_processing import random_edit_img, zoom_rotate_img, blur_img, adjust_color, adjust_contrast, adjust_sharpness, get_tensor_from_image, get_image_from_uri, get_img_dim
from sklearn.pipeline import Pipeline
from joblib import dump


def load_and_preprocess_data(model_filepath: str, image_size: str, train_copies:int, test_copies:int) -> tuple[np.array, np.array]:
    unprocessed_path = os.path.join(model_filepath, "unprocessed.json")
    X_train, y_train, X_test, y_test= [], [], [], [] 

    if os.path.exists(unprocessed_path):
        with open(unprocessed_path, 'r') as file:
            processed_data = json.load(file)
            for item in processed_data:
                img_width, img_height = get_img_dim(image_size)
                original_image = get_image_from_uri(item['image'])

                # augment_functions = [random_edit_img, zoom_rotate_img, blur_img, adjust_color, adjust_contrast, adjust_sharpness]
                augment_functions = [random_edit_img]
                for _ in range(train_copies):
                    for f in augment_functions: 
                        edited_image = f(original_image)  # Assuming this function distorts the image
                        tensor = get_tensor_from_image(image=edited_image, img_width=img_width, img_height=img_height)
                        X_train.append(tensor.numpy().flatten())  # Flatten the image and append to X
                        y_train.append(item['encoded'])  # Append the same ID for each distorted version

                

                # for _ in range(test_copies):
                #     edited_image = random_edit_img(original_image)  # Assuming this function distorts the image
                #     tensor = get_tensor_from_image(image=edited_image, img_width=img_width, img_height=img_height)
                #     X_test.append(tensor.numpy().flatten())  # Flatten the image and append to X
                #     y_test.append(item['encoded'])  # Append the same ID for each distorted version
                
                dequeue(0, model_filepath)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def train_model_batch(model_name:str, image_size:str, train_copies:int, test_copies:int, batch_size:int) -> GridSearchCV:
    data = os.path.join(PROJ_PATH, '.data/knn')
    model_filepath = os.path.join(data, model_name)
    if not os.path.exists(model_filepath):
        os.mkdir(model_filepath)

    # Assume dataset creation and preprocessing is done here
    # raw_json_filepath = os.path.join(model_filepath, '..', '..', 'deckdrafterprod.MTGCard.json')
    # formatted_json_filepath = enqueue(raw_json_filepath, 10, model_filepath, image_size)

    # Load and preprocess the data
    X_train, y_train, X_test, y_test = load_and_preprocess_data(
        model_filepath=model_filepath, 
        image_size=image_size, 
        train_copies=train_copies, 
        test_copies=test_copies
    )

    # Create a pipeline that first scales the data then applies kNN
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

    # Define a grid of parameters to search over
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9],  # Experiment with different values for k
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan', 'minkowski']
    }

    # Use GridSearchCV to find the best parameters
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1)
    grid_search.fit(X_train, y_train)

    # save the best model
    best_model = grid_search.best_estimator_

    # Save the model
    model_save_path = os.path.join(model_filepath, 'knn_checkpoint.joblibl')
    dump(best_model, model_save_path)

    # # Print the best parameters
    print("Best parameters found:")
    print(grid_search.best_params_)

    return grid_search, X_test, y_test

