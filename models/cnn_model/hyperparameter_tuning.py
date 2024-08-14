import os, sys
PROJ_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJ_PATH)

import json
import keras_tuner as kt
import cv2
import pandas as pd
import numpy as np
from tensorflow.keras import layers, models, optimizers #type: ignore

from cnn import create_dataset



def build_model(hp):
    model = models.Sequential()
    
    model.add(layers.InputLayer(shape=(450, 650, 3)))

    model.add(layers.Conv2D(hp.Int('filters_1', min_value=32, max_value=256, step=32), (3, 3)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(hp.Int('filters_2', min_value=32, max_value=256, step=32), (3, 3)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(hp.Int('filters_3', min_value=32, max_value=256, step=32), (3, 3)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(layers.Dense(units=hp.Int('unique_classes', min_value=2, max_value=10, step=1), activation='softmax'))
    
    model.compile(
        optimizer=optimizers.Adam(
            hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG'),
            beta_1=hp.Float('beta_1', min_value=0.8, max_value=0.99, step=0.01),
            beta_2=hp.Float('beta_2', min_value=0.8, max_value=0.99, step=0.01)),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    
    return model


def find_best_hyperparameters():
    # Load your dataset
    dfp = "/home/jude/harmony_org/card_recognition_ml/.data/cnn/OnePieceCard/dataset"
    train_images = os.path.join(dfp, "train_images")
    train_labels = os.path.join(dfp, "train_labels.csv")
    test_images = os.path.join(dfp, "test_images")
    test_labels = os.path.join(dfp, "test_labels.csv")

    batch_size = 32
    img_width = 450 
    img_height = 650 
    
    train_dataset = create_dataset(train_labels, train_images, img_width, img_height, batch_size) 
    test_dataset = create_dataset(test_labels, test_images, img_width, img_height, batch_size)

    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=3,
        directory='my_dir',
        project_name='cnn_tuning')
    
    tuner.search(train_dataset, epochs=10, validation_data=test_dataset, verbose=2)
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_hyperparameters = {
        'units': best_hps.get('units'),
        'learning_rate': best_hps.get('learning_rate'),
        'filters_1': best_hps.get('filters_1'),
        'filters_2': best_hps.get('filters_2'),
        'filters_3': best_hps.get('filters_3'),
        'beta_1': best_hps.get('beta_1'),
        'beta_2': best_hps.get('beta_2'),
        'img_width': best_hps.get('img_width'),
        'img_height': best_hps.get('img_height'),
        'unique_classes': best_hps.get('unique_classes')
    }

    hyperparameter_specs_filepath = os.path.join(dfp, "best_hyper.json") 
    with open(hyperparameter_specs_filepath, 'w') as f:
        json.dump(best_hyperparameters, f, indent=4)

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)

if __name__ == "__main__":
    find_best_hyperparameters()