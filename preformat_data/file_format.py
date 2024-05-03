import csv
# This will take all of the dataset and convert it into something useable
# comment the output
# hopecully we can change this for the different applications that trent wants



# TRAINING
data_path = "/data/train"
csv_path = "/data/train_labels.csv"

# TESTING
# data_path = "/data/test"
# csv_path = "/data/test_labels.csv"

def get_images():
    images = []
    labels = []

    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            name = row[0] # change this depending on the format of the csv file
            label = row[3] # change this depending on the format of the csv file

            images.append(data_path + name)
            labels.append(label)
        

    return images, labels