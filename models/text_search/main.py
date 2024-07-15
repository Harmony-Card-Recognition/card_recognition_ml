import os, sys

PROJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJ_PATH)

import numpy as np
from sklearn.linear_model import SGDClassifier
from dataset import enqueue


if __name__ == "__main__":
    # Model specs
    model_name = 'harmony_text_0.2.36'
    image_size = 'large'
    inital_json_grab =  -1 
    batch_size = 5
    attributes =['_id',
                'image_uris', 
                'name', 
                # 'layout', 
                'mana_cost', 
                'type_line', 
                'oracle_text', 
                'power', 
                'toughness', 
                'artist'
                ]
    # Paths
    data = os.path.join(PROJ_PATH, '.data/text')
    model_filepath = os.path.join(data, model_name)
    if not os.path.exists(model_filepath): os.mkdir(model_filepath)

    # create the entire dataset (just the jsons... no downloading of files) 
    enqueue(
        raw_json_filepath=os.path.join(data, '..', 'deckdrafterprod.MTGCard.json'), 
        inital_json_grab=inital_json_grab, 
        model_filepath=model_filepath, 
        image_size=image_size, 
        attributes=attributes
    )

    unprocessed_filepath = os.path.join(model_filepath, "unprocessed.json")
    specs_filepath = os.path.join(model_filepath, 'model_specs.txt')
    with open(specs_filepath, 'w') as f:
        f.write(f"Model Name: {model_name}\n")
        # f.write(f"Image Size: {image_size}\n")
        f.write(f"Initial JSON Grab: {inital_json_grab}\n")
        f.write(f"Attributes in JSON: {attributes}")