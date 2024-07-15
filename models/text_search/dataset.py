import os, sys
PROJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJ_PATH)

import json
from helper.json_processing import format_json

# note the size of the card... the datasets will be different

# don't just pick a random distortions for the TRAINING set, you should universalize this so that it is consistant (pick 4-6 distortions)
# for the testing set, pick one random distortion and toss it in
def enqueue(raw_json_filepath:str, inital_json_grab: int, model_filepath:str, image_size:str, attributes:list[str]) -> str:
    new_filepath = os.path.join(model_filepath, "unprocessed.json") 
    format_json(raw_json_filepath=raw_json_filepath, new_filepath=new_filepath, image_count=inital_json_grab, image_size=image_size, attributes=attributes)
    # encode_alphanumeric_to_int(new_filepath) # Im a donut... a json is an array with indexes
    return new_filepath

def dequeue(index:int, model_filepath:str) -> bool:
    unprocessed_path = os.path.join(model_filepath, "unprocessed.json")
    processed_path = os.path.join(model_filepath, "processed.json")
    
    # Load unprocessed data
    with open(unprocessed_path, 'r') as file:
        unprocessed_data = json.load(file)
    
    # Check if there's data to process
    if not unprocessed_data:
        print("No data to process.")
        return False
    
    # Remove the first element
    element = unprocessed_data.pop(index)
    
    # Save the updated unprocessed data
    with open(unprocessed_path, 'w') as file:
        json.dump(unprocessed_data, file, indent=4)
    
    # Load processed data
    processed_data = []
    if os.path.exists(processed_path):
        with open(processed_path, 'r') as file:
            processed_data = json.load(file)
    
    # Append the removed element to processed data
    processed_data.append(element)
    
    # Save the updated processed data
    with open(processed_path, 'w') as file:
        json.dump(processed_data, file, indent=4)
    
    return True
