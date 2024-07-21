import os, sys
PROJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJ_PATH)
 
import time
import pandas as pd
import json
import numpy as np

from tensorflow.keras import callbacks, layers, models, optimizers, mixed_precision # type: ignore

from helper.helper import get_elapsed_time
from helper.image_processing import get_tensor_from_image, get_tensor_from_dir

def predict_folder(model_path, overall_json_path, img_folder_path):
    st = time.time()

    model = models.load_model(os.path.join(model_path, 'model.keras'))
    predictions = []
    images = [img for img in os.listdir(img_folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # get the img_dimensions from the  
    with open(os.path.join(model_path, '.metadata.json'), 'r') as file:
        data = json.load(file)
        img_width = data[0]['img_width']
        img_height = data[0]['img_height']

    for i, image_name in enumerate(images):
        img_path = os.path.join(img_folder_path, image_name)
        img_tensor = get_tensor_from_dir(img_path, img_width=img_width, img_height=img_height)

        img_tensor = np.expand_dims(img_tensor, axis=0) # make the image tensor match the input layer of the model
        prediction = model.predict(img_tensor)
        prediction, confidence = np.argmax(prediction), prediction[0,np.argmax(prediction)]
        print(f'Image: {image_name}')
        print(f'Prediction: {prediction}')
        print(f'Confidence: {confidence}')

        csv_path = os.path.join(model_path, 'test_labels.csv')
        card_info_df = pd.read_csv(csv_path)
        
        predicted_id = card_info_df[card_info_df['label'] == prediction]['_id'].iloc[0]
        predicted_obj = find_object_by_id(overall_json_path, predicted_id)
        if predicted_obj is not None: predicted_name = predicted_obj['productUrlName'] 
        else: predicted_name = None
        # predictions[i][image_name] = {'_id': predicted_id, 'productUrlName': predicted_name, 'confidence': str(confidence)}
        predictions.append({image_name:{'_id': predicted_id, 'productUrlName': predicted_name, 'confidence': str(confidence)}})
 
    overall_predict_time = get_elapsed_time(st) 
    ave_time_per_card = (time.time()-st)/len(images)# overall_predict_time/len(images)


    with open(os.path.join(img_folder_path, 'info.txt'), 'a') as f:
        f.write(f'Overall Prediction Time: {overall_predict_time}\n')
        f.write(f'Averate Time Per Card: {ave_time_per_card}\n')
        f.write(f'# of Cards: {len(images)}\n')


    predictions_json_path = os.path.join(img_folder_path, f'{os.path.basename(model_path)}_predictions.json')
    with open(predictions_json_path, 'w') as file:
        json.dump(predictions, file, indent=4)

    print('\n')
    print(f'Overall Prediction Time: {overall_predict_time}\n')
    print(f'Averate Time Per Card: {ave_time_per_card}\n')
    print(f'# of Cards: {len(images)}\n')



def find_object_by_id(overall_json_path, target_id):
    # Assuming overall_json_path is a string containing the path to a JSON file
    with open(overall_json_path, 'r') as file:
        data = json.load(file)

    for obj in data:
        if str(obj['_id']) == str(target_id):
            return obj
    return None



    

if __name__ == '__main__':
    model_path = '/home/jude/Work/Store Pass/card_recognition_ml/.data/cnn/ONEPIECE_0.0.0'
    img_folder_path = '/home/jude/Work/Store Pass/card_recognition_ml/.data/cnn/ONEPIECE_0.0.0/train_images'
    overall_json_path = '/home/jude/Work/Store Pass/card_recognition_ml/.data/deckdrafterprod.OnePieceCard.json'

    predict_folder(model_path=model_path, overall_json_path=overall_json_path, img_folder_path=img_folder_path)