import os, sys

PROJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJ_PATH)

from PIL import Image
from typing import Tuple
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import tensorflow as tf
from keras import models, layers, preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def parse_xml(xml_file):
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

def load_image(image_path):
    return Image.open(image_path)

def preprocess_data(images, annotations):
    # Assuming annotations are in the format [(xmin, ymin, xmax, ymax), ...] for each image
    X = []
    Y = []
    for img in images:
        img = img.resize((224, 224))
        img_array = preprocessing.image.img_to_array(img)
        X.append(img_array)
    for ann in annotations:
        # Normalize bounding box coordinates to [0, 1] by image width and height (224, 224 here)
        norm_boxes = []
        for box in ann:
            xmin, ymin, xmax, ymax = box
            norm_boxes.append([xmin/224, ymin/224, xmax/224, ymax/224])
        Y.append(norm_boxes)
    return np.array(X), np.array(Y)

def define_yolo_model():
    model = models.Sequential([
        layers.Conv2D(16, (3,3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(4)  # Assuming 4 coordinates for bounding box
    ])
    return model

def train_model(model, dataset):
    model.compile(optimizer='adam', loss='mse')
    model.fit(dataset, epochs=10)  # Example parameters

def main(folder_path):
    images = []
    annotations = []
    for file in os.listdir(folder_path):
        if file.endswith('.jpg'):
            images.append(load_image(os.path.join(folder_path, file)))
        elif file.endswith('.xml'):
            annotations.append(parse_xml(os.path.join(folder_path, file)))
    
    # Preprocess data
    X, Y = preprocess_data(images, annotations)
    
    # Split data into training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Define model
    model = define_yolo_model()
    
    # Train model
    train_model(model, (X_train, Y_train))
    
    # Evaluate model (Optional)
    model.evaluate(X_test, Y_test)
    
    # Inference steps (Optional)
    predictions = model.predict(X_test)

    # Save the model
    # model.save()


    # Process predictions to visualize bounding boxes on images
    # Display the first image in the testing set
    plt.figure(figsize=(6, 6))
    plt.imshow(X_test[0].astype('uint8'))  # Assuming X_test is normalized in [0, 1]
    plt.title('First Image in Testing Set')
    plt.axis('off')  # Hide the axis
    plt.show()


if __name__ == '__main__':
    folder_path = '/home/jude/Work/Store Pass/card_recognition_ml/labelImg/.data/augmented'
    main(folder_path)