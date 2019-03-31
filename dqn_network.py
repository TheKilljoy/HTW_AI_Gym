import keras
from keras import backend as k
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Permute
from keras.layers import Conv2D, MaxPooling2D


class DqnNetwork:
    """
    Creates a dqn_network instance.
    Parameters are:
    state_shape: defines the shape of the input. e.g. (4,84,84). Careful,  this neural network works with "channel first" mode,
    therefore the number of channels or in this case the consecutive stack of frames has to be in the first place.\n
    possible_actions: the action space of the given environment. e.g. [1, 1, 1, 1] if the actionspace = 4\n
    learn_rate=0.00025\n
    rho=0.9\n
    epsilon=None\n
    """
    def __init__(self, state_shape, possible_actions, learn_rate=0.00025, rho=0.9, epsilon=None):
        self.model = Sequential()
        self.model.add(Conv2D(filters=32,
                              kernel_size=8,
                              strides=(4, 4),
                              activation="relu",
                              input_shape=state_shape,
                              data_format='channels_first'))
        self.model.add(Conv2D(filters=64,
                              kernel_size=4,
                              strides=(2, 2),
                              padding="valid",
                              activation="relu",
                              input_shape=state_shape,
                              data_format='channels_first'))
        self.model.add(Conv2D(filters=64,
                              kernel_size=3,
                              strides=(1, 1),
                              padding="valid",
                              activation="relu",
                              input_shape=state_shape,
                              data_format='channels_first'))
        self.model.add(Flatten())
        self.model.add(Dense(units=512,
                             activation='relu'))
        self.model.add(Dense(units=len(possible_actions),
                             activation='linear'))
        self.model.compile(loss=keras.losses.logcosh,
                           optimizer=keras.optimizers.RMSprop(learn_rate, rho, epsilon),
                           metrics=['accuracy'])

    def predict(self, input_state):
        return self.model.predict(input_state)

    def fit(self, inputs, targets, batch_size):
        return self.model.fit(inputs, targets, batch_size=batch_size, epochs=1, verbose=0)
    
    def set_weights(self, other_model):
        self.model.set_weights(other_model.get_weights())

    def load_weights(self, text):
        self.model.load_weights(text)
    
    def get_weights(self):
        return self.model.get_weights()

    def write_weights(self, path):
        self.model.save_weights(path)