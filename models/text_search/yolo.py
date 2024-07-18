import os, sys

PROJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJ_PATH)

from PIL import Image
from typing import List, Tuple
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import tensorflow as tf
from keras import models, layers, preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_image(image_path):
    print(f"Loading image from path: {image_path}")
    img = preprocessing.image.load_img(image_path, target_size=(224, 224))  # Adjust target_size as per your model's input
    img_array = preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Model expects batches
    return img_array

def parse_xml(xml_file):
    print(f"Parsing XML file: {xml_file}")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []
    for member in root.findall('object'):
        xmin = int(member.find('bndbox/xmin').text)
        ymin = int(member.find('bndbox/ymin').text)
        xmax = int(member.find('bndbox/xmax').text)
        ymax = int(member.find('bndbox/ymax').text)
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes

def preprocess_data(images, annotations):
    print("Starting data preprocessing")
    # Assuming images and annotations are already in the correct format
    X = np.vstack(images)  # Stack images
    Y = np.array(annotations)  # Convert annotations to a NumPy array
    print("Data preprocessing completed")
    return X, Y

def define_yolo_model():
    model = models.Sequential([
        models.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224)),
        models.MaxPooling2D(2, 2),
        models.Conv2D(64, (3, 3), activation='relu'),
        models.MaxPooling2D(2, 2),
        models.Conv2D(128, (3, 3), activation='relu'),
        models.MaxPooling2D(2, 2),
        models.Flatten(),
        models.Dense(512, activation='relu'),
        # Assuming the output format is [xmin, ymin, xmax, ymax]
        models.Dense(4, activation='sigmoid')  # 'sigmoid' because bounding box coordinates are normalized between 0 and 1
    ])
    return model

def train_model(model, dataset):
    X_train, Y_train = dataset
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, Y_train, epochs=10)  # Example parameters

def main(folder_path):
    images = []
    annotations = []
    for file in os.listdir(folder_path):
        if file.endswith('.jpg'):
            images.append(load_image(os.path.join(folder_path, file)))
        elif file.endswith('.xml'):
            annotations.append(parse_xml(os.path.join(folder_path, file)))
    
    X, Y = preprocess_data(images, annotations)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = define_yolo_model()
    train_model(model, (X_train, Y_train))
    model.evaluate(X_test, Y_test)  # Evaluate model
    predictions = model.predict(X_test)  # Inference steps

    # Save the model
    model.save('yolo_model.h5')  # Save your model

    # Add your logic to process predictions and visualize bounding boxes on images


if __name__ == '__main__':
    folder_path = '/home/jude/Work/Store Pass/card_recognition_ml/labelImg/.data/testing_augmented'
    main(folder_path)