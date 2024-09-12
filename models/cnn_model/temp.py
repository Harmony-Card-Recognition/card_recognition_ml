import os, sys
PROJ_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJ_PATH) 

import json
import csv

from helper.json_processing import format_json
from filepaths import get_filepaths


# step 1 (create a json with the "type" attribute)
# format_json("/home/jude/work/store_pass/card_recognition_ml/.data/deckdrafterprod.PokemonCard.json", 
#             "/home/jude/work/store_pass/card_recognition_ml/.data/deckdrafterprod.PokemonCard(-1).json",
#             -1 , "large")

# step 2 (relable the _ids based on the types attribute) (if there is no types attribute, then label the _id as 0)
types_to_id = {
    "None" : 0,
    "Grass" : 1,
    "Lightning" : 2, 
    "Darkness" : 3,
    "Fairy" : 4, 
    "Fire" : 5, 
    "Psychic" : 6,
    "Metal" : 7,
    "Dragon" : 8,
    "Water" : 9,
    "Fighting" : 10,
    "Colorless" : 11,
}

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




# step 5 ( from a json file that has objects with _id and type, modify the csv with label, filename, _id) with a new label)
# the new label should be based on the types_to_id dictionary. 

# Step 5: Modify the CSV with a new label based on the types_to_id dictionary
# def update_csv_with_labels(json_filepath, csv_filepath, output_csv_filepath):
#     # Load the JSON file
#     with open(json_filepath, 'r') as json_file:
#         json_data = json.load(json_file)

#     # Create a dictionary to map _id to type
#     id_to_type = {item['_id']: item['types'][0] if 'types' in item and item['types'] else 'None' for item in json_data}

#     # Load the CSV file
#     with open(csv_filepath, 'r') as csv_file:
#         csv_reader = csv.reader(csv_file)
#         headers = next(csv_reader)  # Read the header row
#         rows = list(csv_reader)

#     # Update each row with the new label
#     for row in rows:
#         _id = row[headers.index('_id')]  # Assuming _id is one of the columns in the CSV
#         type_label = id_to_type.get(_id, 'None')
#         new_label = types_to_id[type_label]
#         row[headers.index('label')] = new_label  # Overwrite the existing "label" column

#     # Write the updated CSV to a new file
#     with open(output_csv_filepath, 'w', newline='') as output_csv_file:
#         csv_writer = csv.writer(output_csv_file)
#         csv_writer.writerow(headers)  # Write the header row
#         csv_writer.writerows(rows)  # Write the data rows

# # Example usage
# json_filepath = "/home/jude/work/store_pass/card_recognition_ml/.data/deckdrafterprod.PokemonCard(-1).json"
# csv_filepath = "/home/jude/work/store_pass/card_recognition_ml/.data/test_labels.csv"
# output_csv_filepath = "/home/jude/work/store_pass/card_recognition_ml/.data/test_labels_new.csv"
# update_csv_with_labels(json_filepath, csv_filepath, output_csv_filepath)


import csv
from collections import Counter

def calculate_label_percentages(csv_filepath):
    # Read the CSV file
    with open(csv_filepath, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        labels = [row[0] for row in csv_reader]  # Assuming the label is in the first column

    # Count the occurrences of each label
    label_counts = Counter(labels)
    total_count = sum(label_counts.values())

    # Calculate the percentage of each label
    label_percentages = {label: (count / total_count) * 100 for label, count in label_counts.items()}

    return label_percentages

# Example usage
csv_filepath = "/home/jude/work/store_pass/card_recognition_ml/.data/test_labels_new.csv"
label_percentages = calculate_label_percentages(csv_filepath)

# Print the results
for label, percentage in label_percentages.items():
    print(f"Label {label}: {percentage:.2f}%")