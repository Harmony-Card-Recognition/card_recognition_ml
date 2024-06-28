import os, sys
PROJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJ_PATH)


from helper.json_processing import format_json, get_datasets

# note the size of the card... the datasets will be different
# in .data/ann/ make: (0) a KEY folder with ORIGINAL images (1) a testing folder with images, (2) a training folder with images, (3) associating csv files with the label, filename, and _id
# this function checks that the spesified card is not already in the datasets
    # if it is not, then add it
    # if it is, then skip it
    # this may be time consuming, but until I can accuratly get an idea of how the json is updated, I will not be able to know for sure
    # we can leave this out for now

# don't just pick a random distortions for the TRAINING set, you should universalize this so that it is consistant (pick 4-6 distortions)
# for the testing set, pick one random distortion and toss it in


def create_dataset(inital_json_grab, model_filepath, data_filepath, size='small'):
    raw_json_filepath = os.path.join(data_filepath, '..', 'deckdrafterprod.MTGCard.json')
        
    formatted_json_filepath = format_json(raw_json_filepath, inital_json_grab, model_filepath, size)
    train_image_dir, test_image_dir, train_labels_csv, test_labels_csv, unique_classes= get_datasets(formatted_json_filepath, model_filepath)
    # train_image_dir, train_labels_csv = get_train_only_dataset(formatted_json_filepath, model_filepath)

    # create KEY folder
    # create KEY csv
    # create TESTING folder
    # create TESTING csv
    # create TRAINING folder
    # create TRAINING csv




# we can have a queue file that has new card IDs that have not been processed. First we populate this, and then every time we run the script, 
# if there is something in the queue file, then we process that, and then take it out of that file

