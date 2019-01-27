from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.utils import plot_model
from keras import regularizers
import constants

def one_hidden(hidden_layer_units, optimizer):
    model = Sequential()
    model.add(Flatten(input_shape = (constants.IMG_WIDTH, constants.IMG_HEIGHT, 3)))
    model.add(Dense(units = hidden_layer_units, activation = 'relu'))
    model.add(Dense(units = 200, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optimizer,
                  metrics = ['accuracy', 'top_k_categorical_accuracy'])
    return model

def log_reg(optimizer):
    model = Sequential()
    model.add(Flatten(input_shape = (constants.IMG_WIDTH, constants.IMG_HEIGHT, 3)))
    model.add(Dense(units = 200, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optimizer,
                  metrics = ['accuracy', 'top_k_categorical_accuracy'])

    return model

def conv_simple(optimizer, l2):
    model = Sequential()
    reg = regularizers.l2(l2)

    # Block 1
    model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (constants.IMG_WIDTH, constants.IMG_HEIGHT, 3), kernel_regularizer = reg))
    model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_regularizer = reg))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))

    # Block 2
    model.add(Flatten())
    model.add(Dense(1024, activation = 'relu', kernel_regularizer = reg))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation = 'softmax', kernel_regularizer = reg))

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optimizer,
                  metrics = ['accuracy', 'top_k_categorical_accuracy'])
    return model

def vgg(optimizer, l2):
    model = Sequential()
    reg = regularizers.l2(l2)

    # Block 1
    model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (constants.IMG_WIDTH, constants.IMG_HEIGHT, 3), kernel_regularizer = reg))
    model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_regularizer = reg))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Block 2
    model.add(Conv2D(64, (3, 3), activation = 'relu', kernel_regularizer = reg))
    model.add(Conv2D(64, (3, 3), activation = 'relu', kernel_regularizer = reg))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Block 3
    model.add(Conv2D(128, (3, 3), activation = 'relu', kernel_regularizer = reg))
    model.add(Conv2D(128, (3, 3), activation = 'relu', kernel_regularizer = reg))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))

    # Block 4
    model.add(Flatten())
    model.add(Dense(1024, activation = 'relu', kernel_regularizer = reg))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation = 'softmax', kernel_regularizer = reg))

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optimizer,
                  metrics = ['accuracy', 'top_k_categorical_accuracy'])
    return model

def get_models():
    models = {}
    models['one_hidden'] = one_hidden(1024, 'sgd')
    models['log_reg'] = log_reg('sgd')
    models['conv_simple'] = conv_simple('sgd')
    return models

def get_models():
    models = {}
    models['one_hidden'] = one_hidden(1024, 'sgd')
    models['log_reg'] = log_reg('sgd')

    l2 = 0.0005
    models['conv_simple'] = conv_simple('sgd', l2)
    models['vgg'] = vgg('sgd', l2)

    # for model_name, model in models.items():
    #     model.summary()
    #     plot_model(model, to_file = 'model_plots/' + model_name + '.jpg', show_shapes=True)

    return models
