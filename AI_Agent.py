import random
import gym
import cv2
import numpy as np
import tensorflow as tf
import keras
from keras import backend as k
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from collections import deque
import matplotlib.pyplot as plt

EPISODES = 1000
MAX_STEPS = 50000
STACK_SIZE = 4
BATCH_SIZE = 64
MEMORY_SIZE = 2000
PRETRAIN_LENGTH = BATCH_SIZE
GAMMA = 0.9
RENDER_EPISODE = True
explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.00005
decay_step = 0
neural_network_active = 0
neural_network_total = 0

env = gym.make('Breakout-v0')
possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())
stacked_frames = deque([np.zeros((87, 80), dtype=np.int) for i in range(STACK_SIZE)], maxlen=4)

class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size=batch_size,
                                replace=False)
        return [self.buffer[i] for i in index]

    def current_size(self):
       return self.buffer.__len__()

def preprocess_frame(frame):
    #preprocess image
    preprocessed_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    preprocessed_image = cv2.resize(preprocessed_image, (84, 110))
    preprocessed_image = preprocessed_image[15:-8, 2:-2]
    #normalize image
    preprocessed_image = preprocessed_image/255.0
    return preprocessed_image

def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    if is_new_episode:
        stacked_frames = deque([np.zeros((87, 80), dtype=np.int) for i in range(STACK_SIZE)], maxlen=4)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
        #stacked_state = np.expand_dims(stacked_state, 0)
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
        #stacked_state = np.expand_dims(stacked_state, 0)
    return stacked_state, stacked_frames



def predict_action(decay_step, state, actions):
    global neural_network_active
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    if explore_probability > exp_exp_tradeoff:
        choice = random.randint(1, len(possible_actions)) -1
        action = possible_actions[choice]
    else:
        neural_network_active += 1
        Qs = model.predict(np.expand_dims(state, 0))
        choice = np.argmax(Qs)
        action = possible_actions[choice]
    return np.argmax(action), explore_probability

def train(total_loss):
    batch = memory.sample(BATCH_SIZE)
    targets = np.zeros((BATCH_SIZE, len(possible_actions))) 
    for i in range(BATCH_SIZE):
        #0 = states
        #1 = actions
        #2 = rewards
        #3 = next_states
        #4 = dones
        targets[i] = model.predict(np.expand_dims(batch[i][0], 0))
        fut_action = model.predict(np.expand_dims(batch[i][3], 0))
        targets[i, batch[i][1]] = batch[i][2]
        if batch[i][4] == False:
            targets[i, batch[i][1]] += decay_rate * np.max(fut_action)

    states_mb = deque(maxlen=BATCH_SIZE)
    for b in batch:
        states_mb.append(b[0])
    states_mb = np.stack(states_mb, axis=0)
    loss = model.train_on_batch(states_mb, targets)
    total_loss += loss[0]
    return total_loss

#create neural network
model = Sequential()
model.add(Conv2D(input_shape=(87, 80, 4),
                 filters=32,
                 kernel_size=8,
                 strides=2,
                 padding="valid",
                 activation="elu"))
model.add(Conv2D(filters=64,
                 kernel_size=4,
                 strides=2,
                 padding="valid",
                 activation="elu"))
model.add(Conv2D(filters=128,
                 kernel_size=8,
                 strides=2,
                 padding="valid",
                 activation="elu"))
model.add(Flatten())
model.add(Dense(units=512,
                activation='elu'))
model.add(Dense(units=4, #action_space = 4
                activation='elu'))
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.RMSprop(lr=0.001),
              metrics=['accuracy'])

model.load_weights('my_model_weights.h5')

memory = Memory(max_size = MEMORY_SIZE)

#pretrain
for i in range(PRETRAIN_LENGTH):
    if i == 0:
        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    # Get the next_state, the rewards, done by taking a random action
    choice = random.randint(1, env.action_space.n) - 1
    action = possible_actions[choice]
    next_state, reward, done, info = env.step(np.argmax(action))
    #stack frames
    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

    if done:
        next_state = np.zeros(state.shape)
        memory.add((state, action, reward, next_state, done))
        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    else:
        memory.add((state, action, reward, next_state, done))
        state = next_state

total_loss = 0.0
episode_x = []
total_loss_y = []
neural_network_active_list = []
plt.xlabel('episode')
plt.ylabel('total_loss')
plt.title('ai agent learns breakout')
f = open("losses.txt",'a+')
episodeCounter = 0
for episode in range(EPISODES):
    episodeCounter += 1
    state = env.reset()
    prev_state, stacked_frames = stack_frames(stacked_frames, state, True)
    for _ in range(MAX_STEPS):
        decay_step += 1
        neural_network_total += 1
        action, probability = predict_action(decay_step, prev_state, possible_actions)
        state, reward, done, info = env.step(action) # take a random action
        stacked_state, stacked_frames = stack_frames(stacked_frames, state, False)
        memory.add((prev_state, action, reward, stacked_state, done))
        prev_state = stacked_state
        if RENDER_EPISODE:
            env.render()
        if done:
            break
        total_loss = train(total_loss)
    f.write("episode {0} total_loss: {1} with {2} explore probability\n".format(episodeCounter, total_loss, explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)))
    episode_x.append(episode)
    total_loss_y.append(total_loss)
    neural_network_active_list.append((neural_network_active / neural_network_total) * 100)
    plt.plot(episode_x, total_loss_y, color='green')
    plt.plot(episode_x, (neural_network_active_list), color='blue')
    plt.savefig('ai_agent_graph.png')
    if episodeCounter % 5 == 0:
        f.close
        f = open("losses.txt", 'a+')
        model.save_weights('my_model_weights.h5')
    total_loss = 0.0
#save final weights
model.save_weights('my_model_weights.h5')