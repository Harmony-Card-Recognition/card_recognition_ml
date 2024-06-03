import os
import tensorflow as tf

from PIL import Image   

from testing import test_model_via_index

if __name__ == "__main__":
    model_folder = './.data/harmony_0.0.1'
    model_path = os.path.join(model_folder, 'model.keras')
    image_folder = os.path.join(model_folder, 'test_images')
    labels_csv = os.path.join(model_folder, 'test_labels.csv')

    image_path = os.path.join('./.data/harmony_0.0.7/test_images', '400.png')
    img = Image.open(image_path)
    img_width, img_height = img.size

    model = tf.keras.models.load_model(model_path)
    test_model_via_index(image_path, 0, model, img_width, img_height)


