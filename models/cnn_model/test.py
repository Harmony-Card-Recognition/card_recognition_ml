import os, sys
PROJ_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJ_PATH)

from helper.image_processing import get_tensor_from_image, get_tensor_from_dir
from helper.helper import get_elapsed_time
from tensorflow.keras import callbacks, layers, models, optimizers, mixed_precision  # type: ignore
import numpy as np
import json
import pandas as pd
import time


def predict_folder(model_path, overall_json_path, img_folder_path):
    st = time.time()

    model = models.load_model(os.path.join(model_path, 'model.keras'))
    predictions = []
    images = [img for img in os.listdir(
        img_folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # play with this to remove the need for the metadata file
    # this gives the input layer size of the model
    _, img_width, img_height, _ = model.input_shape

    for i, image_name in enumerate(images):
        img_path = os.path.join(img_folder_path, image_name)
        img_tensor = get_tensor_from_dir(
            img_path, img_width=img_width, img_height=img_height)

        # make the image tensor match the input layer of the model
        img_tensor = np.expand_dims(img_tensor, axis=0)
        prediction = model.predict(img_tensor)
        prediction, confidence = np.argmax(
            prediction), prediction[0, np.argmax(prediction)]
        print(f'Image: {image_name}')
        print(f'Prediction: {prediction}')
        print(f'Confidence: {confidence}')

        csv_path = os.path.join(model_path, 'test_labels.csv')
        card_info_df = pd.read_csv(csv_path)

        predicted_id = card_info_df[card_info_df['label']
                                    == prediction]['_id'].iloc[0]
        predicted_obj = find_object_by_id(overall_json_path, predicted_id)
        if predicted_obj is not None:
            predicted_name = predicted_obj['productUrlName']
        else:
            predicted_name = None
        # predictions[i][image_name] = {'_id': predicted_id, 'productUrlName': predicted_name, 'confidence': str(confidence)}
        predictions.append({image_name: {
                           '_id': predicted_id, 'productUrlName': predicted_name, 'confidence': str(confidence)}})

    overall_predict_time = get_elapsed_time(st)
    # overall_predict_time/len(images)
    ave_time_per_card = (time.time()-st)/len(images)

    with open(os.path.join(img_folder_path, 'info.txt'), 'a') as f:
        f.write(f'Overall Prediction Time: {overall_predict_time}\n')
        f.write(f'Averate Time Per Card: {ave_time_per_card}\n')
        f.write(f'# of Cards: {len(images)}\n')

    predictions_json_path = os.path.join(
        img_folder_path, f'{os.path.basename(model_path)}_predictions.json')
    with open(predictions_json_path, 'w') as file:
        json.dump(predictions, file, indent=4)

    print('\n')
    print(f'Overall Prediction Time: {overall_predict_time}\n')
    print(f'Averate Time Per Card: {ave_time_per_card}\n')
    print(f'# of Cards: {len(images)}\n')

def predict_image(img_folder_path, image_name, img_width, img_height, model):
        img_path = os.path.join(img_folder_path, image_name)
        img_tensor = get_tensor_from_dir(
            img_path, img_width=img_width, img_height=img_height)

        img_tensor = np.expand_dims(img_tensor, axis=0)
        prediction = model.predict(img_tensor)
        prediction, confidence = np.argmax(
            prediction), prediction[0, np.argmax(prediction)]
        
        return prediction, confidence

def predict_folder_two_link(
        overall_model_path,
        smaller_models,
        overall_json_path, 
        img_folder_path,   
): 
    # step 1:
    # the card should be passed into the first model
    # based on the output, look up the model filepath for the second model with the smaller_models dictionary

    # step 2:
    # load the second model from the filepath

    # step 3:
    # use the smaller model to identify the final card
    # 
    # step 4: 
    # look up the _id with the function
    # 
    # repeat steps 1-4 for all of the images in the img_folder_path    

    overall_model = models.load_model(os.path.join(overall_model_path, 'model.keras'))

    images = [img for img in os.listdir(
        img_folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    _, img_width, img_height, _ = overall_model.input_shape

    for image_name in images:
        sub_type, sub_type_confidence= int(predict_image(
            img_folder_path=img_folder_path,
            image_name=image_name,
            img_width=img_width,
            img_height=img_height,
            model=overall_model,
        ))
        sub_model = models.load_model(os.path.join(smaller_models[sub_type], 'model.keras'))

        final_prediction, final_prediction_confidence = predict_image(
            img_folder_path=img_folder_path,
            image_name=image_name,
            img_width=img_width,
            img_height=img_height,
            model=sub_model,
        )

        print(f'{image_name}:\nprediction: {final_prediction}\nconfidence: {sub_type_confidence * final_prediction_confidence}')

        





def find_object_by_id(overall_json_path, target_id):
    # Assuming overall_json_path is a string containing the path to a JSON file
    with open(overall_json_path, 'r') as file:
        data = json.load(file)

    for obj in data:
        if str(obj['_id']) == str(target_id):
            return obj
    return None

if __name__ == '__main__':
    smaller_models = {
        0: '/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon0Card/',
        1: '/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon1Card/',
        2: '/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon2Card/',
        3: '/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon3Card/',
        4: '/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon4Card/',
        5: '/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon5Card/',
        6: '/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon6Card/',
        7: '/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon7Card/',
        8: '/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon8Card/',
        9: '/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon9Card/',
        10:'/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon10Card/',
        11:'/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon11Card/',
    }
     
    predict_folder_two_link(
        overall_model_path='/home/jude/harmony_org/card_recognition_ml/.data/cnn/PokemonCard/',
        smaller_models=smaller_models,
        img_folder_path='/home/jude/harmony_org/scans/pokemon/card_1',
        overall_json_path='',
    )



# if __name__ == '__main__':
#     model_path = '/home/jude/Work/Store Pass/card_recognition_ml/.data/cnn/ONEPIECE_0.0.0'
#     img_folder_path = '/home/jude/Work/Store Pass/card_recognition_ml/.data/cnn/ONEPIECE_0.0.0/train_images'
#     overall_json_path = '/home/jude/Work/Store Pass/card_recognition_ml/.data/deckdrafterprod.OnePieceCard.json'

#     predict_folder(model_path=model_path,
#                    overall_json_path=overall_json_path, img_folder_path=img_folder_path)
