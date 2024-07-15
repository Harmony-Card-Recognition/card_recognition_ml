# given a image url identify the "name", "oracle_text", and the "type_line"
# there will be 3 predefined rectangles that will be associated to those three attributes
# use pytesseract and Pillow
# this function should return a dictionary with the keys being the 3 attributes, and the values being what the model spits out

import os, sys

PROJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJ_PATH)


import requests
import pytesseract
import json
 

from PIL import Image
from io import BytesIO
from typing import Tuple

from search import search

from helper.image_processing import get_textbox_dim



def extract_text(image:Image, rectangle:Tuple[int, int]):
    cropped_image = image.crop(rectangle)
    text = pytesseract.image_to_string(cropped_image)
    return text.strip()



def search(json_filepath:str, attribute_value_pairs:dict, attribute_ranks:list[str]) -> list[str]:
    with open(json_filepath, 'r') as file:
        data = json.load(file)

    for i in range(len(attribute_ranks)):
        attribute = attribute_ranks[i]
        new_data = []
        for card in data:
            if card[attribute] == attribute_value_pairs[attribute]:
               new_data.append(card)
        data = new_data
        if len(new_data) <= 1: break

    return data 




def identify_card(image:Image, json_filepath:str) -> list[str]:
    # identifies a few '_id's that the image could be

    attribute_value_pairs = {
        'name': None, # use the extract_text method 
        'type_line': None, 
        'oracle_text': None, # use the extract_text method
        'artist': None, # use the extract_text method
        'power': None, # power and toughness could be in the same location 
        'toughness': None, 
        'mana_cost': None # use a different model to give me a string that matches the json
    }

    # YOLOv5 to find the rectangles for the attributes
    rectangles = {
        # (left_x_val, top_y_val, right_x_val, bottom_y_val)
        'name': (0, 0, 0, 0),
        'type_line': (0, 0, 0, 0),
        'oracle_text': (0, 0, 0, 0),
        'artist': (0, 0, 0, 0),
        'power/toughness': (0, 0, 0, 0),
        'mana_cost': (0, 0, 0, 0)
    }


    # extract the text for the 'name', 'type_line', 'oracle_text', 'power/toughness', and 'artist'
    # do some string processing if nessicary
    # set those values to the attribute dict

    # load a CNN that can recognize the the mana and turn it into a set of alphanumeric characters
    # do some string processing if nessicary (i.e. putting {} around them to match the json)
    mana_cnn_path = ''
    # load the cnn
    # evalueate the cnn with the image
    # set attribute_value_pairs['mana_cost'] = output

    # search the json to try and find some '_id's that match this image!
    attribute_ranks = {'name', 'type_line', 'oracle_text', 'artist', 'power', 'toughness', 'mana_cost'}
    search(json_filepath=json_filepath, attribute_value_pairs=attribute_value_pairs, attribute_ranks=attribute_ranks)




if __name__ == '__main__':
    # create the JSON ONLY ONCE
    json_filepath = '/home/jude/Work/Store Pass/card_recognition_ml/.data/text/harmony_text_0.2.36/unprocessed.json'
    image_url = 'https://cards.scryfall.io/large/front/0/0/0000579f-7b35-4ed3-b44c-db2a538066fe.jpg?156289497'
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    # NOTE: THIS IMAGE MAY NOT HAVE A GUARENTEED DIMENSION

    ids = identify_card(image=image, json_filepath=json_filepath)
    print(identify_card(image=image))
    
    

