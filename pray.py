import base64
import glob
import io
import json
import pickle
import keras
import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
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

# =======================================================================


json_filepath = './.data/reduced_deckdrafterprod.MTGCard.json'


# =======================================================================

def current_time():
    '''Help: Returns the current time as a nice string.'''
    return strftime("%B %d, %I:%M%p", localtime())

def elapsed_time(start_time):
    '''Using seconds since epoch, determine how much time has passed since the provided float. Returns string
    with hours:minutes:seconds'''
    elapsed_seconds = time()-start_time
    h = int(elapsed_seconds/3600)
    m = int((elapsed_seconds-h*3600)/60)
    s = int((elapsed_seconds-m*60)-h*3600)
    return f'{h}hr {m}m {s}s'

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

    for subdir, dirs, files in os.walk(storage_path):
        for file in np.sort(files):
            if file.endswith('.jpg'):
                #open the image, then convert it to an array and scale values from 0-1
                image = Image.open(f"{storage_path}/{file}")
                image = image.resize((224,312))
                image = image.convert('RGB') # BW?
                scaled_array = np.array(image)/255

                #pull the _id from the filename
                label = int(file[0])

                #add the data
                image_array.append(scaled_array)
                label_array = np.append(label_array, label)

    #convert image list to numpy array
    image_array = np.array(image_array)
    
    return image_array, label_array

# Load the JSON data
with open(json_filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Function to get image url by _id
def get_image_url_by_id(_id, image_size):
    for item in data:
        if '_id' in item and 'image_uris' in item and _id in item['_id']:
            return item['image_uris'][image_size]
    return None

def generate_imgs(_id_list, storage_path):
    '''Help: High level function to reduce the testing & training image creation process down to a single step. 
    Provide a string list of _ids, the number of distortions desired for each printing, a general 
    storage location for the image files, and the final image size desired. Creates "poorly photographed" 
    versions of each _id printing provided. Results are named based on their index in the list.'''
    
    image_size = 'normal'
    
    images_created = 0
    printings_distorted = 0

    count = 0
    #iterate through the provided list
    for _id in _id_list:
        printings_distorted += 1
        #pull the image URL using the _id
        image_url = get_image_url_by_id(_id, image_size)
        if image_url == None:
            continue

        #get the raw image file
        clean_img = Image.open(urlopen(image_url))

        clean_img.save(f"{storage_path}/{count}.jpg")
        count+=1
            
    print(f"\n{images_created} total unique distortions saved from {printings_distorted} different printings.")
    print(f"Images stored @ '{storage_path}'\n")

def generate_img_set(image_set_name, _id_list_train, _id_list_test, resize=True, verbose=True):
    '''Help: Given appropriate parameters, generate num_distortions distorted image copies of each card in 
    _id_list. Then prep all the images for neural net training and save the resulting arrays.
    Returns model_data: ((training_images, training_labels), (testing_images, testing_labels))
    
    image_set_name: str, desired folder name of current image set
    _ids_list, list of ints, card printings to use
    num_distortions, number of warped copies of each card to create
    resize, boolean, if false, images keep 936,672 original resolution, otherwise resize to (224,312)
    verbose, boolean, if true print statements show function progress
    '''

    if verbose:
        print(f"Process started for {image_set_name} on {current_time()} ...")
        start_time = time()

    #if the folder already exists, delete it so we can start fresh
    if os.path.exists(image_set_name):
        shutil.rmtree(image_set_name)

    #now create the directory, and sub folders for image storage
    os.mkdir(image_set_name)
    os.mkdir(f'{image_set_name}/Testing')
    os.mkdir(f'{image_set_name}/Training')

    if verbose:
        print(f'Folder structure created, generating {len(_id_list_train)} \
training images now ...')

    #now create images for training and testing, testing will always have two images, change here if need be
    #create the training images
    training_storage_path = f"{image_set_name}/Training"

    #generate the images
    generate_imgs(_id_list_train, training_storage_path)

    if verbose:
        print(f"Training images finished on {current_time()}, now generating {len(_id_list_train)*2} \
testing images ...")

    #create the testing images
    testing_storage_path = f"{image_set_name}/Testing"

    #generate the images
    generate_imgs(_id_list_test, testing_storage_path)

    if verbose:
        print(f"All images created and saved under {image_set_name} on {current_time()}. \n\
Formatting images and labels for neural net processing now ...")

    #now open up all the image files and store contents as arrays for the neural net
    training_images, training_labels = prep_images_for_network(training_storage_path)
    testing_images, testing_labels = prep_images_for_network(testing_storage_path)

    #save the input arrays locally for later use in case we want them
    model_data = ((training_images, training_labels), (testing_images, testing_labels))
    save_object(model_data, f'{image_set_name} Arrays.p', verbose=False)

    if verbose:
        print(f"Pre processing complete on {current_time()} after {elapsed_time(start_time)}. \
\n\nTraining & testing data saved locally ({image_set_name} Arrays.p) and ready for neural network!\n\n")

    return model_data


def train_CNN_model(model_name, model_data, unique_printings, epochs=10, verbose=True):
    '''Help: Create and train a CNN model for the provided model_data'''
    
    #unpack the model_data variable
    ((training_images, training_labels), (testing_images, testing_labels)) = model_data

    if verbose:
        print(f'Initializing {model_name} on {current_time()} ...')
        model_start_time = time()

    #if the folder already exists, delete it so we can start fresh
    if os.path.exists(f'{model_name}.model'):
        shutil.rmtree(f'{model_name}.model')

    
    print(training_images)
    print(training_images.shape)
    print(training_images.shape[1:])

    # #initialize the neural network model
    # model = models.Sequential()
    # model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=training_images.shape[1:]))
    # model.add(layers.MaxPooling2D(2,2))
    # model.add(layers.Conv2D(64, (3,3), activation='relu'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dropout(0.5)) 
    # model.add(layers.Dense(unique_printings, activation='softmax'))

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
        print(f"\nModel evaluated & saved locally at '{model_name}.model' on {current_time()} \
after {elapsed_time(model_start_time)}!\n")

    return model


def test_model_via_index(image_set_name, card_index, model, training_list, test_list):
    filepath = f'{image_set_name}/Testing/{card_index}.jpg'
    test_card = Image.open(filepath)
    test_card = test_card.resize((224,312))
    test_card = test_card.convert('RGB') #BW?

    #provide the image to the model and see what comes back
    img_as_array = np.array(np.array(test_card)/255)

    eval_images = []
    eval_images.append(img_as_array)
    eval_images = np.array(eval_images)

    result = model.predict(eval_images)
    result_index, confidence = np.argmax(result), result[0,np.argmax(result)]

    correct = False
    #display the result!
    if training_list[result_index] == test_list[card_index]:
        #display(test_card)
        correct = True
        print(f'For card index {card_index}, model predicted index {result_index} \
with {np.round(confidence*100,4)}% confidence.')
    
    else:
        print(f'For card index {card_index}, model predicted index {result_index} \
with {np.round(confidence*100,4)}% confidence. (INCORRECT)')
        #display(test_card, Image.open(f'{image_set_name}/Testing/{result_index}-sub_index.jpg'))

#function to randomly select 100 IDs from json file
def random_ids():
    '''Help: Randomly select 100 IDs from json file and return them as a list.'''
    _ids = []
    count = 0
    while count < 100:
        card = random.choice(data)
        if len(card['_id']) > 0:
            _ids.append(card['_id'][0])
            count += 1
    return _ids


# List of sIDs
_id_list1 = random_ids()

#pick 2 random number between 1 - 95
num1 = random.randint(1, 95)
num2 = random.randint(1, 95)
_id_list2 = [_id_list1[num1], _id_list1[num2]]

# # Generate image set
model_data = generate_img_set("my_image_set", _id_list1, _id_list2)

# Number of unique printings
unique_printings = len(_id_list1)

# Train the model
model = train_CNN_model("my_model", model_data, unique_printings, 100)

model = models.load_model('my_model.keras')

model.fit(model_data[0][0], model_data[0][1], epochs=200, validation_data=(model_data[1][0], model_data[1][1]))

for i in range(len(_id_list2)):
    test_model_via_index("my_image_set", i, model, _id_list1, _id_list2)