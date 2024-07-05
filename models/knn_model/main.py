import os, sys

import numpy as np
from sklearn.linear_model import SGDClassifier

PROJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJ_PATH)

import json, pickle, joblib

from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline

from knn import load_and_preprocess_training, load_and_preprocess_testing, train_model
from dataset import enqueue

if __name__ == "__main__":
    # Model specs
    model_name = 'harmony_knn_0.2.16'
    image_size = 'small'
    train_copies = 20
    test_copies = 1
    inital_json_grab = 100 
    batch_size = 5

    # Paths
    data = os.path.join(PROJ_PATH, '.data/knn')
    model_filepath = os.path.join(data, model_name)
    if not os.path.exists(model_filepath): os.mkdir(model_filepath)

    # create the entire dataset (just the jsons... no downloading of files) 
    enqueue(
        raw_json_filepath=os.path.join(data, '..', 'deckdrafterprod.MTGCard.json'), 
        inital_json_grab=inital_json_grab, 
        model_filepath=model_filepath, 
        image_size=image_size
    )

    # some more paths
    checkpoint_filepath = os.path.join(model_filepath, 'knn_checkpoint.pkl')
    unprocessed_filepath = os.path.join(model_filepath, "unprocessed.json")
    with open(unprocessed_filepath, 'r') as f: 
        unique_classes = len(json.load(f))
    
    # save the model specs to a readable format
    specs_filepath = os.path.join(model_filepath, 'model_specs.txt')
    with open(specs_filepath, 'w') as f:
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Image Size: {image_size}\n")
        f.write(f"Train Copies: {train_copies}\n")
        f.write(f"Test Copies: {test_copies}\n")
        f.write(f"Initial JSON Grab: {inital_json_grab}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f'Unique Classes: {unique_classes}')
        # training time
        # testing time (per card)
        # model size
        # date trained?


    
 

    # Initialize or load the pipeline
    if os.path.exists(checkpoint_filepath):
        # NOTE: this doesn't add onto the training. This overwrites the work we did, and this is why this way did not work
        with open(checkpoint_filepath, 'rb') as file:
            pipeline = pickle.load(file)  # Use pickle to load the model
        print("Loaded pre-trained model for further training") 
    else:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance', metric='manhattan'))
        ])
        print("No pre-trained model found. Created a new pipeline for the model.")



    train_model(
        model_name=model_name,
        image_size=image_size,
        train_copies=train_copies,
        pipeline=pipeline 
    )
    
    # save the model
    with open(checkpoint_filepath, 'wb') as file:
        pickle.dump(pipeline, file)
    
    # Test the model
    X_test, y_test = load_and_preprocess_testing(model_filepath=model_filepath, image_size=image_size, test_copies=test_copies)
    accuracy = pipeline.score(X_test, y_test)
    print(f"Model Overall Accuracy: {accuracy}")