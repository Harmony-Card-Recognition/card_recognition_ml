


from tensorflow.keras import callbacks, layers, models, optimizers, mixed_precision, regularizers# type: ignore


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


def model_6(img_width, img_height, unique_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(img_width, img_height, 3)))

    # First Convolutional Block
    model.add(layers.Conv2D(128, (3, 3)))  # Increased filters from 64 to 128
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Second Convolutional Block
    model.add(layers.Conv2D(256, (3, 3)))  # Increased filters from 128 to 256
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Third Convolutional Block
    model.add(layers.Conv2D(512, (3, 3)))  # Increased filters from 256 to 512
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Fourth Convolutional Block
    model.add(layers.Conv2D(512, (3, 3)))  # Increased filters from 256 to 512
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Fifth Convolutional Block
    model.add(layers.Conv2D(1024, (3, 3)))  # Increased filters from 512 to 1024
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(4096))  # Increased units from 2048 to 4096
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(unique_classes, activation='softmax'))

    return model

def model_7(img_width, img_height, unique_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(img_width, img_height, 3)))

    # First Convolutional Block
    model.add(layers.Conv2D(96, (3, 3)))  # Increased filters from 64 to 96
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Second Convolutional Block
    model.add(layers.Conv2D(192, (3, 3)))  # Increased filters from 128 to 192
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Third Convolutional Block
    model.add(layers.Conv2D(384, (3, 3)))  # Increased filters from 256 to 384
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Fourth Convolutional Block
    model.add(layers.Conv2D(384, (3, 3)))  # Increased filters from 256 to 384
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Fifth Convolutional Block
    model.add(layers.Conv2D(768, (3, 3)))  # Increased filters from 512 to 768
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(3072))  # Increased units from 2048 to 3072
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(unique_classes, activation='softmax'))

    return model


def model_8(img_width, img_height, unique_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(img_width, img_height, 3)))

    # First Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.001)))  # Added L2 regularization
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Second Convolutional Block
    model.add(layers.Conv2D(128, (3, 3), kernel_regularizer=regularizers.l2(0.001)))  # Added L2 regularization
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Third Convolutional Block
    model.add(layers.Conv2D(256, (3, 3), kernel_regularizer=regularizers.l2(0.001)))  # Added L2 regularization
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Fourth Convolutional Block
    model.add(layers.Conv2D(256, (3, 3), kernel_regularizer=regularizers.l2(0.001)))  # Added L2 regularization
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Fifth Convolutional Block
    model.add(layers.Conv2D(512, (3, 3), kernel_regularizer=regularizers.l2(0.001)))  # Added L2 regularization
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, kernel_regularizer=regularizers.l2(0.001)))  # Added L2 regularization
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Dropout(0.5))  # Increased dropout rate from 0.5 to 0.6
    model.add(layers.Dense(unique_classes, activation='softmax'))

    return model

def model_9(img_width, img_height, unique_classes):
    # resnet-like model
    def residual_block(x, filters, kernel_size=3, stride=1):
        shortcut = x
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.01)(x)
        x = layers.Conv2D(filters, kernel_size, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([shortcut, x])
        x = layers.LeakyReLU(negative_slope=0.01)(x)
        return x

    inputs = layers.Input(shape=(img_width, img_height, 3))
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    for filters in [64, 128, 256, 512]:
        x = residual_block(x, filters)
        x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(4096, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(unique_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model