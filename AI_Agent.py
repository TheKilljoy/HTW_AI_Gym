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

EPISODES = 500000
MAX_STEPS = 50000
STACK_SIZE = 4
BATCH_SIZE = 32
MEMORY_SIZE = 200000 #goal 1.000.000, but currently 300.000 -> 10gb ram
PRETRAIN_LENGTH = 50000
GAMMA = 0.99
RENDER_EPISODE = False
LEARN_RATE = 0.00025
NETWORK_UPDATE = 5000
explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.000001
decay_step = 0
neural_network_active = 0
neural_network_total = 0
reward_per_episode = 0

env = gym.make('Breakout-v0')
state_shape = (4, 84, 84)
possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())
stacked_frames = np.zeros(state_shape, dtype=np.uint8) 

class Memory():
    def __init__(self, max_size):
        #self.buffer = deque(maxlen = max_size)
        self.size = max_size
        self.experience = []

    def add(self, state0, action, reward, state1, done):
        #self.buffer.append(experience) #(state0, action, reward, state1, done)
        if len(self.experience) >= self.size:
            self.experience.pop(0)
        self.experience.append({'state0': state0,
                                'action': action,
                                'reward': reward,
                                'state1': state1,
                                'done': done})
        if len(self.experience) % 100 == 0 and len(self.experience) != self.size:
            print("{0} of {1} samples accumulated".format(len(self.experience), self.size))


    def sample(self, batch_size):
        batch = []
        for i in range(batch_size):
            batch.append(self.experience[randrange(0, len(self.experience))])
        return np.asarray(batch)

    def length(self):
       return self.experience.__len__()

    def size_in_megabytes(self):
        return sys.getsizeof(self.experience) / 1024.0

def preprocess_frame(frame):
    #preprocess image
    preprocessed_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    preprocessed_image = cv2.resize(preprocessed_image, (84, 110))
    preprocessed_image = preprocessed_image[26:, :]
    return preprocessed_image

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


def predict_action(decay_step, state, actions):
    global neural_network_active
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    if explore_probability > exp_exp_tradeoff:
        choice = randint(1, len(possible_actions)) -1
        action = possible_actions[choice]
    else:
        neural_network_active += 1
        #normalize and expand dims, so that it is a batch of size 1
        state_input_form = np.expand_dims(state.astype('float64'), axis=0)
        choices = model.predict(state_input_form)
        choice = np.argmax(choices)
        action = possible_actions[choice]
    return np.argmax(action), explore_probability

def train():
    # print("Training with mini batch")
    batch = memory.sample(BATCH_SIZE)
    inputs = []
    targets = []
    loss = 0
    for dataset in batch:
        inputs.append(dataset['state0'].astype('float64'))
        training_state1 = np.expand_dims(dataset['state1'].astype('float64'), axis=0)
        training_state1_prediction = target_model.predict(training_state1)
        q_max = np.max(training_state1_prediction)
        
        t = list(model.predict(np.expand_dims(dataset['state0'].astype('float64'), axis=0))[0])
        if dataset['done']:
            t[dataset['action']] = dataset['reward']
        else:
            t[dataset['action']] = dataset['reward'] + GAMMA * q_max
        targets.append(t)

    inputs = np.asarray(inputs).squeeze()
    targets = np.asarray(targets).squeeze()
    return model.fit(inputs, targets, batch_size = BATCH_SIZE, epochs=1, verbose=0)

#create neural network
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
                optimizer=keras.optimizers.RMSprop(lr=LEARN_RATE, rho=0.95, epsilon=0.01),
                metrics=['accuracy'])
    return model

model = create_dqnetwork()
target_model = create_dqnetwork()

target_model.set_weights(model.get_weights())

memory = Memory(max_size = MEMORY_SIZE)
#init replay memory D to capacity N
for i in range(PRETRAIN_LENGTH):
    if i == 0:
        state = env.reset()
        stacked_frames = stack_frames(stacked_frames, state, True)
        state0 = stacked_frames
    action = env.action_space.sample()
    state1, reward, done, info = env.step(action)
    stacked_frames = stack_frames(stacked_frames, state1, False)
    state1 = stacked_frames

    if done:
        state = env.reset()
        stacked_frames = stack_frames(stacked_frames, state, True)
    else:
        #in state0 perform action, get reward and be in state1 where the game might be done
        for j in range(3):
            state1, reward, done, info = env.step(action)
            reward_per_episode += reward
            stacked_frames = stack_frames(stacked_frames, state1, False)

        state1 = stacked_frames
        memory.add(state0, action, reward, state1, done)
        state0 = state1 #as state1 is now the current state it becomes state0

total_loss = 0.0
#episode_x = []
#reward_per_episode_list = []
episode_counter = 0
total_counter = 0
network_counter = 0
probability = 0
f = open("rewards.txt",'a+')
for episode in range(EPISODES):
    total_counter += 1
    episode_counter += 1
    state0 = env.reset()
    stacked_frames = stack_frames(stacked_frames, state0, True)
    state0 = stacked_frames
    reward_per_episode = 0
    for i in range(MAX_STEPS):
        decay_step += 1
        total_counter += 1
        action, probability = predict_action(decay_step, state0, possible_actions)
        state1, reward, done, info = env.step(action)
        reward_per_episode += reward

        #frameskipping
        for j in range(3):
            state1, reward, done, info = env.step(action)
            reward_per_episode += reward
            stacked_frames = stack_frames(stacked_frames, state1, False)

        state1 = stacked_frames
        memory.add(state0, action, reward, state1, done)
        state0 = state1
        network_counter += 1
        history = train()
        history = history.history     

        if(network_counter >= NETWORK_UPDATE):
            print("Updated target network")
            target_model.set_weights(model.get_weights())
            network_counter = 0
        
        if RENDER_EPISODE:
            env.render()
        if done:
            f.write("Episode: {0} Reward: {1} Current Probability: {2} Loss: {3} Acc: {4} MemoryLength: {5} MemorySize: {6}mb\n".format(episode_counter, reward_per_episode, probability, history['loss'][0], history['acc'][0], memory.length(), memory.size_in_megabytes() ))
            break

    print("Episode {0} Score {1}".format(episode_counter, reward_per_episode))
    if episode_counter % 10 == 0:
        model.save_weights('my_model_weights.h5')
        f.close()
        f = open("rewards.txt", 'a+')
    total_loss = 0.0
#save final weights
model.save_weights('final_model_weights.h5')
