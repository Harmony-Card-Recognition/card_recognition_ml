from PIL import Image

# Open the image file
img = Image.open(r'.data/harmony_1.3.2/train_images/46.png')

# Get the size of the image
width, height = img.size

print(f"The dimensions of the image are {width} x {height}")





























# import json
# import os
# import shutil

# from image_processing import random_edit_img

# train_images_dir = r'.data/harmony_1.3.2/train_images'
# train_labes_csv_dir = r'.data/harmony_1.3.2/train_labels.csv'

# # make a temp directory to hold all the augmented images
# temp_dir = 'temp_images'
# os.makedirs(temp_dir, exist_ok=True)

# # for each image in the train_images directory
# for filename in os.listdir(train_images_dir):
#     # augment it, and put it in the temporary directory
#     augmented_image1 = augment_image(os.path.join(train_images_dir, filename))
#     augmented_image2 = augment_image(os.path.join(train_images_dir, filename))
#     augmented_image3 = augment_image(os.path.join(train_images_dir, filename))
    
#     # move augmented images to the temporary directory
#     shutil.move(augmented_image1, os.path.join(temp_dir, f'augmented_1_{filename}'))
#     shutil.move(augmented_image2, os.path.join(temp_dir, f'augmented_2_{filename}'))
#     shutil.move(augmented_image3, os.path.join(temp_dir, f'augmented_3_{filename}'))

# # move all of the images into the train_images folder, and somehow update the labels
# for filename in os.listdir(temp_dir):
#     # move augmented images to the train_images folder
#     shutil.move(os.path.join(temp_dir, filename), os.path.join(train_images_dir, filename))
    
#     # update the labels (assuming the labels are stored in a CSV file)
#     update_labels(train_labes_csv_dir, filename)

# # remove the temporary directory
# shutil.rmtree(temp_dir)
