


from tensorflow.keras import callbacks, layers, models, optimizers, mixed_precision  # type: ignore


def model_1(img_width, img_height, unique_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(img_width, img_height, 3)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(128, (3, 3)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(256, (3, 3)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(256, (3, 3)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(512, (3, 3)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Flatten())
    model.add(layers.Dense(2048))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(unique_classes, activation='softmax'))

    return model


def model_2(img_width, img_height, unique_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(img_width, img_height, 3)))

    # First Convolutional Block
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Third Convolutional Block
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Dropout(0.5))
    
    # Output Layer
    model.add(layers.Dense(unique_classes, activation='softmax'))

    return model

def model_3(img_width, img_height, unique_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(img_width, img_height, 3)))

    # First Convolutional Block
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.ELU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.ELU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Third Convolutional Block
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.ELU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    model.add(layers.ELU())
    model.add(layers.Dropout(0.5))
    
    # Output Layer
    model.add(layers.Dense(unique_classes, activation='softmax'))

    return model

def model_4(img_width, img_height, unique_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(img_width, img_height, 3)))

    # First Convolutional Block
    model.add(layers.Conv2D(16, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second Convolutional Block
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Third Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Dropout(0.5))
    
    # Output Layer
    model.add(layers.Dense(unique_classes, activation='softmax'))

    return model