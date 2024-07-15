import json
import requests
from pathlib import Path


def download_image(image_url, save_path):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raises an HTTPError if the response status code is 4XX/5XX
        with open(save_path, 'wb') as f:
            f.write(response.content)
    except requests.exceptions.HTTPError as err:
        print(f"Failed to download image. HTTP Error: {err}")



if __name__ == "__main__":
    json_file_path = '/home/jude/Work/Store Pass/card_recognition_ml/.data/cnn/harmony_cnn_0.0.0/deckdrafterprod.MTGCard_small(1000).json'
    original_path = '/home/jude/Work/Store Pass/card_recognition_ml/labelImg/.data/original/'
    
    # Ensure the original_path exists
    Path(original_path).mkdir(parents=True, exist_ok=True)
    
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    for card in data:
        image_url = card.get('image_uris')  # Adjust 'image_url' based on your JSON structure
        if image_url:
            # Use card ID or name as filename. Adjust 'id' or 'name' based on your JSON structure
            filename = card.get('_id', 'default_filename') + '.jpg'
            save_path = Path(original_path) / filename
            download_image(image_url, save_path)
            print(f"Downloaded and saved image to {save_path}")