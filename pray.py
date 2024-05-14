import base64
import glob
import io
import json
import pickle
import keras
import os
from PIL import Image
from urllib.request import urlopen
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import random
import os
from keras import datasets, layers, models
import shutil
from time import localtime, strftime, time
from IPython.display import display
import copy

from image_processing import random_edit_img
from helper import get_current_time, get_elapsed_time

# ======================================================================

json_filepath = './.data/deckdrafterprod.MTGCard.json'

# load json data
def load_json_data(filepath):
    data = None
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# ======================================================================



#simple function to pickle variables for later use. save a local pickle
def save_object(obj, filename, verbose=True):
    '''Help: Given an object & filepath, store the object as a pickle for later use.'''
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"File saved at {filename}")
    
#and later load the file back into a variable
def load_object(filename, verbose=True):
    '''Help: Loads something previously pickled from the provided file path.'''
    with open(filename, 'rb') as f:
        load_test = pickle.load(f)
    if verbose:
        print(f"File loaded from {filename}")
    return load_test



def prep_images_for_network(storage_path):
    '''Help: Given a folder of distorted printings, compile all images and their labels into arrays for
    neural network processing. Returns image_array, label_array'''
    
    #initialize the empty arrays
    image_array = []
    label_array = np.array([], dtype=int)

    count = 0
    mincount = 0

    for subdir, dirs, files in os.walk(storage_path):
        for file in np.sort(files):
            if file.endswith('.jpg'):
                #open the image, then convert it to an array and scale values from 0-1
                image = Image.open(f"{storage_path}/{file}")
                image = image.resize((224,312))
                image = image.convert('RGB')
                scaled_array = np.array(image)/255

                if(not storage_path.__contains__("Testing")):
                    label = int(file.split('-')[0])
                else:
                    label = int(file.split('.')[0])

                #add the data
                image_array.append(scaled_array)
                label_array = np.append(label_array, label)
                mincount += 1
                if mincount == 4:
                    count += 1
                    mincount = 0

    #convert image list to numpy array
    image_array = np.array(image_array)
    
    return image_array, label_array



# Function to get image url by multiverse_id
def get_image_url_by_multiverse_id(multiverse_id, image_size, data):
    for item in data:
        if 'multiverse_ids' in item and 'image_uris' in item and multiverse_id in item['multiverse_ids']:
            return item['image_uris'][image_size]
    return None

def generate_imgs(multiverse_id_list, storage_path, data):
    '''Help: High level function to reduce the testing & training image creation process down to a single step. 
    Provide a string list of multiverse_ids, the number of distortions desired for each printing, a general 
    storage location for the image files, and the final image size desired. Creates "poorly photographed" 
    versions of each multiverse_id printing provided. Results are named based on their index in the list.'''
    
    image_size = 'normal'
    
    images_created = 0
    printings_distorted = 0

    count = 0
    #iterate through the provided list
    for multiverse_id in multiverse_id_list:
        printings_distorted += 1
        #pull the image URL using the multiverse_id
        if (storage_path.__contains__("Testing")):
            print(multiverse_id[0])
            print(multiverse_id[1])
            image_url = get_image_url_by_multiverse_id(multiverse_id[0], image_size, data)
        else:
            image_url = get_image_url_by_multiverse_id(multiverse_id, image_size, data)
        if image_url == None:
            continue

        #get the raw image file
        img = Image.open(urlopen(image_url))

        #immediately save if it is testing image
        if (storage_path.__contains__("Testing")):
            img.save(f"{storage_path}/{multiverse_id[1]}.jpg")
        else:
            for i in range(4):
                img_copy = copy.deepcopy(img)
                #randomly choose if an image should get distorted
                if random.choice([True, True, True, False]):
                    img_copy = random_edit_img(img, distort=True, verbose=False)
                img_copy.save(f"{storage_path}/{count}-{i}.jpg")

        count+=1
            
    print(f"\n{images_created} total unique distortions saved from {printings_distorted} different printings.")
    print(f"Images stored @ '{storage_path}'\n")

def generate_img_set(image_set_name, multiverse_id_list_train, multiverse_id_list_test, data, resize=True, verbose=True):
    '''Help: Given appropriate parameters, generate num_distortions distorted image copies of each card in 
    multiverse_id_list. Then prep all the images for neural net training and save the resulting arrays.
    Returns model_data: ((training_images, training_labels), (testing_images, testing_labels))
    
    image_set_name: str, desired folder name of current image set
    multiverse_ids_list, list of ints, card printings to use
    num_distortions, number of warped copies of each card to create
    resize, boolean, if false, images keep 936,672 original resolution, otherwise resize to (224,312)
    verbose, boolean, if true print statements show function progress
    '''

    if verbose:
        print(f"Process started for {image_set_name} on {get_current_time()} ...")
        start_time = time()

    #if the folder already exists, delete it so we can start fresh
    if os.path.exists(image_set_name):
        shutil.rmtree(image_set_name)

    #now create the directory, and sub folders for image storage
    os.mkdir(image_set_name)
    os.mkdir(f'{image_set_name}/Testing')
    os.mkdir(f'{image_set_name}/Training')

    if verbose:
        print(f'Folder structure created, generating {len(multiverse_id_list_train)} training images now ...')

    #now create images for training and testing, testing will always have two images, change here if need be
    #create the training images
    training_storage_path = f"{image_set_name}/Training"

    #generate the images
    generate_imgs(multiverse_id_list_train, training_storage_path, data)

    if verbose:
        print(f"Training images finished on {get_current_time()}, now generating {len(multiverse_id_list_train)*2} testing images ...")

    #create the testing images
    testing_storage_path = f"{image_set_name}/Testing"

    #generate the images
    generate_imgs(multiverse_id_list_test, testing_storage_path, data)

    if verbose:
        print(f"All images created and saved under {image_set_name} on {get_current_time()}. \nFormatting images and labels for neural net processing now ...")

    #now open up all the image files and store contents as arrays for the neural net
    training_images, training_labels = prep_images_for_network(training_storage_path)
    testing_images, testing_labels = prep_images_for_network(testing_storage_path)

    #save the input arrays locally for later use in case we want them
    model_data = ((training_images, training_labels), (testing_images, testing_labels))
    save_object(model_data, f'{image_set_name} Arrays.p', verbose=False)

    if verbose:
        print(f"Pre processing complete on {get_current_time()} after {get_elapsed_time(start_time)}.\n\nTraining & testing data saved locally ({image_set_name} Arrays.p) and ready for neural network!\n\n")

    return model_data

# ============================================================================


def train_CNN_model(model_name, model_data, unique_printings, epochs=10, verbose=True):
    '''Help: Create and train a CNN model for the provided model_data'''
    
    #unpack the model_data variable
    ((training_images, training_labels), (testing_images, testing_labels)) = model_data

    if verbose:
        print(f'Initializing {model_name} on {get_current_time()} ...')
        model_start_time = time()

    #if the folder already exists, delete it so we can start fresh
    if os.path.exists(f'{model_name}.model'):
        shutil.rmtree(f'{model_name}.model')

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=training_images.shape[1:]))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5)) 
    model.add(layers.Dense(unique_printings, activation='softmax'))



    # Define the optimizer
    optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

    #compile the model
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    if verbose:
        print('Network compiled, fitting data now ... \n')
    #fit the model to the provided data
    model.fit(training_images, training_labels, epochs=epochs, validation_data=(testing_images, testing_labels))

    if verbose:
        print('\nModel fit, elvaluating accuracy and saving locally now ... \n')
    #evaluate the model

    loss, accuracy = model.evaluate(testing_images, testing_labels)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    #save it locally for future reuse
    model.save(f'{model_name}.keras')

    if verbose:
        print(f"\nModel evaluated & saved locally at '{model_name}.model' on {get_current_time()} after {get_elapsed_time(model_start_time)}!\n")

    return model


def test_model_via_index(image_set_name, card_index, model):
    filepath = f'{image_set_name}/Testing/{card_index}.jpg'
    test_card = Image.open(filepath)
    test_card = test_card.resize((224,312))
    test_card = test_card.convert('RGB')

    #provide the image to the model and see what comes back
    img_as_array = np.array(np.array(test_card)/255)

    eval_images = []
    eval_images.append(img_as_array)
    eval_images = np.array(eval_images)

    result = model.predict(eval_images)
    result_index, confidence = np.argmax(result), result[0,np.argmax(result)]

    correct = False
    #display the result!
    if result_index == card_index:
        #display(test_card)
        correct = True
        print(f'For card index {card_index}, model predicted index {result_index} \
with {np.round(confidence*100,4)}% confidence.')
    
    else:
        print(f'For card index {card_index}, model predicted index {result_index} \
with {np.round(confidence*100,4)}% confidence. (INCORRECT)')
        #display(test_card, Image.open(f'{image_set_name}/Testing/{result_index}-sub_index.jpg'))

#function to randomly select 100 multiverse IDs from deck.json
def random_multiverse_ids(data, times=25):
    '''Help: Randomly select 100 multiverse IDs from deck.json and return them as a list.'''
    multiverse_ids = []
    count = 0
    while count < times:
        card = random.choice(data)
        if len(card['multiverse_ids']) > 0:
            multiverse_ids.append(card['multiverse_ids'][0])
            count += 1
    return multiverse_ids

def test_model(model, test_data, test_labels):
    test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
    print('\nTest Accuracy: ', test_acc) 



# =========================================================

processed_json_data = load_json_data(json_filepath)

# #List of multiverse IDs
# multiverse_id_list1 = random_multiverse_ids(processed_json_data, 100)
# multiverse_id_list2 = []
# #pick random number between 1 - 95
# for i in range(15):
#     num = random.randint(1, 95)
#     multiverse_id_list2.append([multiverse_id_list1[num], num])
# print(multiverse_id_list2)
# # Generate image set
# model_data = generate_img_set("imagev2", multiverse_id_list1, multiverse_id_list2, processed_json_data)
# # Number of unique printings
# unique_printings = len(multiverse_id_list1)
# # Train the model
# model = train_CNN_model("modelv2", model_data, unique_printings, 25)

model_data = load_object('imagev2 Arrays.p')
model = models.load_model('modelv2.keras')
model.fit(model_data[0][0], model_data[0][1], epochs=1, validation_data=(model_data[1][0], model_data[1][1]))



test_model(model, model_data[1][0], model_data[1][1])

## model.evaluate()
# print(model_data[0][1])
# print(model_data[1][1])
# print(model.summary())
# print(multiverse_id_list1)
# print(multiverse_id_list2)


# for i in range(len(model_data[1][1])):
#     test_model_via_index("imagev2", model_data[1][1][i], model)