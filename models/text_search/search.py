import os, sys
PROJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJ_PATH)

import json
import time

def search(json_filepath:str, attribute_value_pairs:dict, attribute_ranks:list[str]) -> list[str]:
    with open(json_filepath, 'r') as file:
        data = json.load(file)

    iterations = 0
    for i in range(len(attribute_ranks)):
        attribute = attribute_ranks[i]
        new_data = []
        for card in data:
            if card[attribute] == attribute_value_pairs[attribute]:
               new_data.append(card)
            iterations += 1
        data = new_data
        if len(new_data) <= 1: break

    # print(f'ITERATIONS: {iterations}')
    return data 




if __name__ == '__main__':
    json_filepath = '/home/jude/Work/Store Pass/card_recognition_ml/.data/text/harmony_text_0.2.34/unprocessed.json'
    attribute_value_pairs_test ={
        "_id": "GLEz0Azbwv",
        "image_uris": "https://cards.scryfall.io/small/front/0/0/00042443-4d4e-4087-b4e5-5e781e7cc5fa.jpg?1562894988",
        "name": "Wall of Vipers",
        "layout": "normal",
        "mana_cost": "{2}{B}",
        "type_line": "Creature \u2014 Snake Wall",
        "oracle_text": "Defender (This creature can't attack.)\n{3}: Destroy Wall of Vipers and target creature it's blocking. Any player may activate this ability.",
        "power": "2",
        "toughness": "4",
        "artist": "Marc Fishman"
    }
    attribute_ranks = [
        'name', 
        'oracle_text',
        'artist',
        # 'power', 
        # 'toughness', 
        # 'mana_cost'
    ]
    start = time.time()
    id_prediction = search(json_filepath=json_filepath, attribute_value_pairs=attribute_value_pairs_test, attribute_ranks=attribute_ranks)
    end = time.time()
    print(f'Prediction: {id_prediction}')
    print(f'Computation Time: {end-start}')
    # if id_prediction == attribute_value_pairs_test['_id']: print("it found it!")