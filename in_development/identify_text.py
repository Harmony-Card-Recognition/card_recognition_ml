import tensorflow as tf
import keras_ocr
import matplotlib.pyplot as plt

from card_recognition_ml.helper.image_processing import load_image

def identify_text(img_path, card_width, card_height, y0, x0, cropped_width, cropped_height):
    # ===================================================
    # crop the image
    image_tensor = load_image(img_path, card_width, card_height)
    cropped_image = tf.image.crop_to_bounding_box(image_tensor, y0, x0, cropped_height, cropped_width)

    # Convert the tensor to an image
    cropped_image = tf.image.convert_image_dtype(cropped_image, tf.uint8).numpy()

    # ===================================================
    # save the image
    plt.imshow(cropped_image)
    plt.savefig('cropped_image.png')

    # ===================================================
    # load keras-ocr's text recognition model
    pipeline = keras_ocr.pipeline.Pipeline()

    # Predict the text from the image
    predictions = pipeline.recognize([cropped_image])[0]

    # Extract the text from the predictions
    text = ' '.join([word for word, box in predictions])

    return text


identify_text('/home/jude/Work/Store Pass/card_recognition_ml/modules/7.png', 146, 204, 0, 0, 100, 100)