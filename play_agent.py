from random import random, randrange, randint
import gym
import numpy as np
import tensorflow as tf
import keras
import cv2
import sys
from PIL import Image
from keras import backend as k
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Permute
from keras.layers import Conv2D, MaxPooling2D
from collections import deque
import matplotlib.pyplot as plt
import time

env = gym.make('Breakout-v0')
state_shape = (4, 84, 84)
possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())
stacked_frames = np.zeros(state_shape, dtype=np.uint8)

def create_dqnetwork():
    model = Sequential()
    model.add(Conv2D(filters=32,
                    kernel_size=8,
                    strides=(4, 4),
                    activation="relu",
                    input_shape=state_shape,
                    data_format='channels_first'))
    model.add(Conv2D(filters=64,
                    kernel_size=4,
                    strides=(2, 2),
                    padding="valid",
                    activation="relu",
                    input_shape=state_shape,
                    data_format='channels_first'))
    model.add(Conv2D(filters=64,
                    kernel_size=3,
                    strides=(1, 1),
                    padding="valid",
                    activation="relu",
                    input_shape=state_shape,
                    data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(units=512,
                    activation='relu'))
    model.add(Dense(units=len(possible_actions), #action_space = 4
                              activation='linear')) 
    model.compile(loss=keras.losses.mean_squared_error,
                optimizer=keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
                metrics=['accuracy'])
    return model

model = create_dqnetwork()
model.load_weights("./my_model_weights.h5")

def preprocess_frame(frame):
    #preprocess image
    preprocessed_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    preprocessed_image = cv2.resize(preprocessed_image, (84, 110))
    preprocessed_image = preprocessed_image[26:, :]
    #preprocessed_image = Image.fromarray(frame, 'RGB').convert('L')
    #preprocessed_image = preprocessed_image.resize((84, 84))
    #preprocessed_image = np.array(preprocessed_image)
    return preprocessed_image #.astype('uint8')

def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    if is_new_episode:
        for i in range(4):
            if(len(stacked_frames) >= 4):
                stacked_frames = np.delete(stacked_frames, 0, axis=0)
            stacked_frames = np.append(stacked_frames, np.expand_dims(frame, axis=0), axis=0)
    else:
        if(len(stacked_frames) >= 4):
            stacked_frames = np.delete(stacked_frames, 0, axis=0)    
        stacked_frames = np.append(stacked_frames, np.expand_dims(frame, axis=0), axis=0)
    return stacked_frames

for episode in range(100):
    state0 = env.reset()
    stacked_frames = stack_frames(stacked_frames, state0, True)
    state0 = stacked_frames
    reward_per_episode = 0
    for i in range(500000):
        state_input_form = np.expand_dims(state0.astype('float32'), axis=0)
        action = np.argmax(model.predict(state_input_form))
        state1, reward, done, info = env.step(action)
        stacked_frames = stack_frames(stacked_frames, state1, False)
        reward += reward
        env.render()
        time.sleep(1/30)
        #frameskipping
        for j in range(3):
            state1, reward, done, info = env.step(action)
            stacked_frames = stack_frames(stacked_frames, state1, False)
            reward += reward
            env.render()
            time.sleep(1/30)
        state1 = stacked_frames
        state0 = state1
        if (done):
            break