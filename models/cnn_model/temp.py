import os, sys
PROJ_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJ_PATH)


from helper.json_processing import format_json
from filepaths import get_filepaths

# step 1 (create a json with the "type" attribute)
# format_json("/home/jude/work/store_pass/card_recognition_ml/.data/deckdrafterprod.PokemonCard.json", 
#             "/home/jude/work/store_pass/card_recognition_ml/.data/deckdrafterprod.PokemonCard(-1).json",
#             -1 , "large")

# step 2 (relable the _ids based on the types attribute) (if there is no types attribute, then label the _id as 0)

# step 3 (toss this information into a different model, and name it something else without the "(-1)")

# step 4 (use a small model like model_classic_2 and train the model and peep the results!)


