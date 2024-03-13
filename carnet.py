from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dropout, Dense


class Carnet:
    def __init__(self, height, width, channel):
        self.height = height
        self.width = width
        self.channel = channel
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(96, (11, 11,), padding='valid', strides=(1, 1), dilation_rate=(2, 2), input_shape=(self.height, self.width, self.channel), name='conv1'))
        model.add(Activation('relu', name='relu1'))
        model.add(MaxPooling2D((2, 2), strides=(3, 3), padding='same', name='pool1'))

        model.add(Conv2D(192, (11, 11), padding='same', name='conv2', strides=(1, 1), dilation_rate=(2, 2), ))
        model.add(Activation('relu', name='relu2'))
        model.add(MaxPooling2D((2, 2), padding='same', name='pool2'))

        model.add(Conv2D(384, (11, 11), padding='same', name='conv3', strides=(1, 1), dilation_rate=(2, 2), ))
        model.add(Activation('relu', name='relu3'))
        model.add(MaxPooling2D((2, 2), padding='same', name='pool3'))

        model.add(Flatten())
        model.add(Dropout(0.8, name='dropout6'))
        model.add(Dense(4096, activation='relu', name='fc6'))
        model.add(Dropout(0.8, name='dropout7'))
        model.add(Dense(4096, activation='relu', name='fc7'))
        model.add(Dropout(0.8, name='dropout8'))
        model.add(Dense(1, activation='sigmoid', name='predictions'))

        model.summary()
        return model