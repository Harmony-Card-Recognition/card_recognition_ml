import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from ..helper.image_processing import load_image
from ..helper.json_processing import format_json, get_datasets
import numpy as np  # Ensure numpy is imported

def preprocess_images_for_knn(csv_file, image_dir, img_width, img_height):
    df = pd.read_csv(csv_file)
    filenames = df['filename'].tolist()
    labels = df['label'].tolist()
    images = [load_image(os.path.join(image_dir, f), img_width, img_height) for f in filenames]
    return images, labels



def train_new_KNN_model(model_filepath, train_image_dir, test_image_dir, train_labels_csv, test_labels_csv, n_neighbors=3):
    img_width, img_height = 146, 204,  # Example dimensions, adjust as needed

    # Load and preprocess images
    train_images, train_labels = preprocess_images_for_knn(train_labels_csv, train_image_dir, img_width, img_height)
    test_images, test_labels = preprocess_images_for_knn(test_labels_csv, test_image_dir, img_width, img_height)

    # Convert images to NumPy arrays and flatten
    train_images_flattened = [np.array(img).flatten() for img in train_images]
    test_images_flattened = [np.array(img).flatten() for img in test_images]

    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Train KNN
    knn.fit(train_images_flattened, train_labels)

    # Predict on test set
    predictions = knn.predict(test_images_flattened)

    # Evaluate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Test Accuracy: {accuracy}")

    # Save the model (optional)
    # joblib.dump(knn, f"{model_filepath}knn_model.joblib")

    return knn

# Main logic adjusted for KNN
if __name__ == "__main__":
    action = 0  # Assuming action 0 for simplicity

    data = './.data/'
    model_name = 'harmony_knn_0.0.0'
    model_filepath = f'{data}{model_name}/'

    if action == 0:
        print('MAKING NEW MODEL FROM SCRATCH')
        raw_json_filepath = f'{data}/deckdrafterprod.MTGCard.json'
        formatted_json_filepath = format_json(raw_json_filepath, 1000, 'small')
        train_image_dir, test_image_dir, train_labels_csv, test_labels_csv, unique_classes = get_datasets(formatted_json_filepath, model_filepath)
        
        knn_model = train_new_KNN_model(
            model_filepath,
            train_image_dir,
            test_image_dir,
            train_labels_csv,
            test_labels_csv,
            n_neighbors=3
        )
    else:
        print("Invalid action value. Please choose a valid action.")