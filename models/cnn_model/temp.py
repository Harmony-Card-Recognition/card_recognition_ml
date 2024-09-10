import os, sys
PROJ_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJ_PATH) 

import json

from helper.json_processing import format_json
from filepaths import get_filepaths


# step 1 (create a json with the "type" attribute)
# format_json("/home/jude/work/store_pass/card_recognition_ml/.data/deckdrafterprod.PokemonCard.json", 
#             "/home/jude/work/store_pass/card_recognition_ml/.data/deckdrafterprod.PokemonCard(-1).json",
#             -1 , "large")

# step 2 (relable the _ids based on the types attribute) (if there is no types attribute, then label the _id as 0)
# types_to_id = {
#     "None" : 0,
#     "Grass" : 1,
#     "Lightning" : 2, 
#     "Darkness" : 3,
#     "Fairy" : 4, 
#     "Fire" : 5, 
#     "Psychic" : 6,
#     "Metal" : 7,
#     "Dragon" : 8,
#     "Water" : 9,
#     "Fighting" : 10,
#     "Colorless" : 11,
# }

# def relabel_ids(input_filepath, output_filepath):
#     with open(input_filepath, 'r') as infile:
#         data = json.load(infile)

#     for item in data:
#         if ('types' not in item) or (item['types'] == None):
#             item['_id'] = 0
#         else: 
#             print(item['_id'])
#             item['_id'] = types_to_id[item['types'][0]] 

#     with open(output_filepath, 'w') as outfile:
#         json.dump(data, outfile, indent=4)

# input_filepath = "/home/jude/work/store_pass/card_recognition_ml/.data/deckdrafterprod.PokemonCard(-1).json"
# output_filepath = "/home/jude/work/store_pass/card_recognition_ml/.data/deckdrafterprod.PokemonCard_relabelled.json"
# relabel_ids(input_filepath, output_filepath)

# step 3 (toss this information into a different model, and name it something else without the "(-1)")

# step 4 (use a small model like model_classic_2 and train the model and peep the results!)


