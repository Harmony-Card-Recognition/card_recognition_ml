import json

# Load the data
with open('./.data/deckdrafterprod.MTGCard.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
# Reduce each object to only "_id" and "image_uris"
reduced_data = [{"_id": item["_id"], "image_uris": item["image_uris"]} for item in data if "_id" in item and "image_uris" in item]

# Write the reduced data to a new JSON file
with open('./.data/reduced_deckdrafterprod.MTGCard.json', 'w') as f:
    json.dump(reduced_data, f, indent=4)