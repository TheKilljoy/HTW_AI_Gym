import random
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
from pynput.keyboard import Key, Listener, KeyCode
from multiprocessing import Process, Queue, Lock, Pool

MAX_STEPS = 5000000
MAX_DOING_NOTHING = 30
STACK_SIZE = 4
BATCH_SIZE = 32
MEMORY_SIZE = 300000 #goal 1.000.000, but currently 300.000 -> 10gb ram
PRETRAIN_LENGTH = 50000
GAMMA = 0.99
RENDER_EPISODE = True
LEARN_RATE = 0.00025
NETWORK_UPDATE = 40000
is_doing_nothing = True
explore_start = 1.0
explore_stop = 0.1
explore_steps = 1000000 #original 1.000.000
epsilon = explore_start
decay_rate = (explore_start - explore_stop)/explore_steps
decay_step = 0
neural_network_active = 0
neural_network_total = 0
reward_per_episode = 0

env = gym.make('Pong-v0')
state_shape = (4, 84, 84)
possible_actions = np.array(np.identity(env.action_space.n, dtype=np.uint8).tolist())
stacked_frames = np.zeros(state_shape, dtype=np.uint8)

def on_release(key):
    global RENDER_EPISODE
    if key == KeyCode.from_char('r'):
        RENDER_EPISODE = not RENDER_EPISODE

def do_multicore_compressing(frames):
    with Pool(16) as pool:
        return pool.map(rle_compress, frames)

def do_multicore_decompressing(compressed_frames):
    with Pool(4) as pool:
        return pool.map(rle_decompress, compressed_frames)

class Memory():
    def __init__(self, max_size):
        #self.buffer = deque(maxlen = max_size)
        self.size = max_size
        self.experience = []

    def add(self, state0, action, reward, state1, done):
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
            batch.append(self.experience[int(random.random() * len(self.experience)) ])
        return np.asarray(batch)

    def length(self):
       return self.experience.__len__()

    def size_in_megabytes(self):
        return sys.getsizeof(self.experience) / 1024.0

def rle_compress(frame):
    compressed_frame = []
    for y in range(84):
        count = 0
        current_number = 0
        for x in range(84):
            if x == 0:
                current_number = frame[y][x]
            if current_number == frame[y][x]:
                count += 1
            else:
                compressed_frame.append(( np.uint8(current_number), np.uint8(count) ))
                count = 1
                current_number = frame[y][x]
        compressed_frame.append(( np.uint8(current_number), np.uint8(count) ))
    return np.asarray(compressed_frame)

def rle_decompress(compressed_frame):
    frame = np.array(np.zeros((84, 84), dtype=np.uint8))
    y_index = 0
    x_index = 0
    count = 0
    for i in range(len(compressed_frame)):
        current_number = compressed_frame[i][0]
        #print("for j in range {0}".format(compressed_frame[i][1]))
        for j in range(compressed_frame[i][1]):
            x_index = count + j
            #print("frame[{0}][{1}]".format(y_index, x_index))
            frame[y_index][x_index] = np.uint8(current_number)
        count += compressed_frame[i][1]
        if(x_index == 83):
            y_index += 1
            count = 0
    return frame


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

def predict_action(state, actions):
    if random.random() < epsilon:
        choice = int(random.random() * len(possible_actions))
        action = possible_actions[choice]
    else:
        #normalize and expand dims, so that it is a batch of size 1
        state_input_form = np.expand_dims(state.astype('float32'), axis=0)
        choices = model.predict(state_input_form)
        choice = np.argmax(choices)
        action = possible_actions[choice]
    return np.argmax(action)

def train():
    # print("Training with mini batch")
    batch = memory.sample(BATCH_SIZE)
    inputs = []
    targets = []
    loss = 0
        
    for dataset in batch:
        inputs.append(dataset['state0'].astype('float32'))
        training_state1 = np.expand_dims(dataset['state1'].astype('float32'), axis=0)

        training_state1_prediction = target_model.predict(training_state1)
        q_max = np.max(training_state1_prediction)

        t = list(model.predict(np.expand_dims(dataset['state0'].astype('float32'), axis=0))[0])
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
    model.compile(loss=keras.losses.logcosh,
                optimizer=keras.optimizers.RMSprop(lr=LEARN_RATE),
                metrics=['accuracy'])
    return model

#enable key input
listener = Listener(on_release = on_release)
listener.start()

model = create_dqnetwork()
target_model = create_dqnetwork()

target_model.set_weights(model.get_weights())

frameskip_reward_accumulated = 0
memory = Memory(max_size=MEMORY_SIZE)
#init replay memory D to capacity N
while memory.length() <= PRETRAIN_LENGTH:
    if memory.length() == 0:
        state = env.reset()
        stacked_frames = stack_frames(stacked_frames, state, True)
        state0 = stacked_frames
    
    frameskip_reward_accumulated = 0 
    
    action = env.action_space.sample()
    state1, reward, done, info = env.step(action)
    frameskip_reward_accumulated += reward
    stacked_frames = stack_frames(stacked_frames, state1, False)
    #in state0 perform action, get reward and be in state1 where the game might be done
    for j in range(3):
        state1, reward, done, info = env.step(action)
        reward_per_episode += reward
        frameskip_reward_accumulated += reward
        stacked_frames = stack_frames(stacked_frames, state1, False) 
        if done:
            break
    
    state1 = stacked_frames

    if frameskip_reward_accumulated > 1.0:
        frameskip_reward_accumulated = 1
    if frameskip_reward_accumulated < -1.0:
        frameskip_reward_accumulated = -1

    memory.add(state0, action, frameskip_reward_accumulated, state1, done)
    state0 = state1 #as state1 is now the current state it becomes state0
    if done:
        reward_per_episode = 0
        state0 = env.reset()
        stacked_frames = stack_frames(stacked_frames, state0, True)
        state0 = stacked_frames

reward_per_episode = 0
frameskip_reward_accumulated = 0
total_loss = 0.0
episode_counter = 0
total_counter = 0
network_counter = 0
f = open("rewards.txt",'a+')
step = 0
doing_nothing_counter = 0
lives = 0
#init game start
state0 = env.reset()
stacked_frames = stack_frames(stacked_frames, state0, True)
state0 = stacked_frames
while step <= MAX_STEPS:
    #do regular game steps
    frameskip_reward_accumulated = 0
    action = predict_action(state0, possible_actions)
    state1, reward, done, info = env.step(action)
    stacked_frames = stack_frames(stacked_frames, state1, False)
    lives = info['ale.lives']
    reward_per_episode += reward
    frameskip_reward_accumulated += reward
    step += 1
    if RENDER_EPISODE:
        env.render()
    #frameskipping
    for j in range(3):
        state1, reward, done, info = env.step(action)
        reward_per_episode += reward
        frameskip_reward_accumulated += reward
        stacked_frames = stack_frames(stacked_frames, state1, False)
        if RENDER_EPISODE:
            env.render()
        if done:
            break
    state1 = stacked_frames
    if info['ale.lives'] < lives:
        lives = info['ale.lives']
        frameskip_reward_accumulated = -1.0
        reward_per_episode -= 1.0
        is_doing_nothing = True

    #check if it is the start of the round and the agent is doing nothing
    if action != 0:
        is_doing_nothing = False
    if is_doing_nothing and action == 0:
        doing_nothing_counter += 1
    if doing_nothing_counter > MAX_DOING_NOTHING:
        frameskip_reward_accumulated = -1.0
        print ("Neural network got punished for doing nothing")
        reward_per_episode -= 1.0
        done = True

    #frame where the agent gets a reward could be skipped, therefore clip the total reward to max 1
    if frameskip_reward_accumulated > 0:
        frameskip_reward_accumulated = 1.0
    if frameskip_reward_accumulated < 0:
        frameskip_reward_accumulated = -1
        
    memory.add(state0, action, frameskip_reward_accumulated, state1, done)
    state0 = state1
    history = train()
    history = history.history

    #update epsilon
    epsilon -= decay_rate
    if epsilon < explore_stop:
        epsilon = explore_stop

    if step % NETWORK_UPDATE == 0:
        print("Updated target network")
        target_model.set_weights(model.get_weights())

    
    if (episode_counter + 1) % 10 == 0:
        model.save_weights('my_model_weights.h5')
        f.close()
        f = open("rewards.txt", 'a+')

        #init new game
    if done:
        print("Episode {0} Score {1}".format(episode_counter, reward_per_episode))
        f.write("Episode: {0} Reward: {1} Steps: {2} Current Epsilon: {3} Loss: {4} Acc: {5}\n".format(episode_counter, reward_per_episode, step, epsilon, history['loss'][0], history['acc'][0]))
        episode_counter += 1
        state0 = env.reset()
        stacked_frames = stack_frames(stacked_frames, state0, True)
        state0 = stacked_frames
        reward_per_episode = 0
        doing_nothing_counter = 0
        is_doing_nothing = True

#save final weights
model.save_weights('final_model_weights.h5')
