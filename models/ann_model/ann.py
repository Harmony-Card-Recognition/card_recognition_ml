import os, sys
PROJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJ_PATH)

import pandas as pd
import numpy as np
from annoy import AnnoyIndex
from sklearn.metrics import accuracy_score
from concurrent.futures import ProcessPoolExecutor
from helper.image_processing import load_image
from helper.json_processing import format_json, get_datasets

def load_and_process_image(image_path, img_width, img_height):
    return np.array(load_image(image_path, img_width, img_height)).flatten()

def preprocess_images_for_ann_parallel(csv_file, image_dir, img_width, img_height):
    df = pd.read_csv(csv_file)
    paths = [os.path.join(image_dir, f) for f in df['filename'].tolist()]
    labels = df['label'].tolist()

    with ProcessPoolExecutor() as executor:
        images = list(executor.map(load_and_process_image, paths, [img_width]*len(paths), [img_height]*len(paths)))

    return np.array(images), labels

def train_new_ANN_model(model_filepath, train_image_dir, test_image_dir, train_labels_csv, test_labels_csv, n_trees=10, batch_size=100):
    img_width, img_height = 146, 204

    # Load and preprocess images in parallel
    train_images, train_labels = preprocess_images_for_ann_parallel(train_labels_csv, train_image_dir, img_width, img_height)
    test_images, test_labels = preprocess_images_for_ann_parallel(test_labels_csv, test_image_dir, img_width, img_height)

    # Initialize ANN index
    f = train_images.shape[1]
    t = AnnoyIndex(f, 'angular')

    # Add vectors to the Annoy index in batches
    for i in range(0, len(train_images), batch_size):
        batch = train_images[i:i+batch_size]
        for j, vector in enumerate(batch):
            t.add_item(i+j, vector)

    t.build(n_trees, n_jobs=-1)  # Utilize all available cores
    t.save(os.path.join(model_filepath, 'ann_model.ann'))

    # Load the index for querying
    u = AnnoyIndex(f, 'angular')
    u.load(os.path.join(model_filepath, 'ann_model.ann'))

    # Predict on test set by finding nearest neighbors
    predictions = [train_labels[u.get_nns_by_vector(img, 1)[0]] for img in test_images]

    # Evaluate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Test Accuracy: {accuracy}")

    return t

if __name__ == "__main__":
    action = 0

    data = os.path.join(PROJ_PATH, '.data/ann')
    model_name = 'harmony_ann_0.0.0'
    model_filepath = os.path.join(data, model_name)
    os.mkdir(model_filepath)

    if action == 0:
        print('MAKING NEW MODEL FROM SCRATCH')
        raw_json_filepath = os.path.join(data, '..', 'deckdrafterprod.MTGCard.json')
        formatted_json_filepath = format_json(raw_json_filepath, 5, model_filepath, 'small')
        train_image_dir, test_image_dir, train_labels_csv, test_labels_csv, unique_classes = get_datasets(formatted_json_filepath, model_filepath)
        
        ann_model = train_new_ANN_model(
            model_filepath,
            train_image_dir,
            test_image_dir,
            train_labels_csv,
            test_labels_csv,
            n_trees=10
        )
    else:
        print("Invalid action value. Please choose a valid action.")