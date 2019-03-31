import dqn_network
import util

import memory
import random
import numpy as np

class DqnAgent:
    def __init__(self, input_shape, env, MAX_STEPS, MAX_DOING_NOTHING, 
                 FRAME_SIZE, BATCH_SIZE, MEMORY_SIZE, PRETRAIN_LENGTH,
                 NETWORK_UPDATE, EXPLORE_START, EXPLORE_END, EXPLORE_STEPS,
                 GAMMA, is_training, is_rendering, is_compressing=True):
        
        self.action_space = np.array(np.identity(env.action_space.n, dtype=np.uint8).tolist())
        self.env = env
        self.is_training = is_training
        self.is_pretrain = True
        self.PRETRAIN_LENGTH = PRETRAIN_LENGTH
        self.MAX_DOING_NOTHING = MAX_DOING_NOTHING
        self.FRAME_SIZE = FRAME_SIZE
        self.NETWORK_UPDATE = NETWORK_UPDATE
        self.MAX_STEPS = MAX_STEPS
        self.EXPLORE_START = EXPLORE_START
        self.EXPLORE_END = EXPLORE_END
        self.EXPLORE_STEPS = EXPLORE_STEPS
        self.epsilon = EXPLORE_START
        self.GAMMA = GAMMA
        self.BATCH_SIZE = BATCH_SIZE
        self.is_rendering = is_rendering
        self.is_compressing = is_compressing

        self.decay_rate = (EXPLORE_START - EXPLORE_END)/EXPLORE_STEPS
        self.doing_nothing_counter = 0
        self.is_doing_nothing = True
        self.stacked_frames = np.array(np.zeros(input_shape), dtype=np.uint8)

        self.model = dqn_network.DqnNetwork(input_shape, self.action_space)
        self.target_model = dqn_network.DqnNetwork(input_shape, self.action_space)
        self.target_model.set_weights(self.target_model)
        self.memory = memory.Memory(max_size=MEMORY_SIZE, is_multiprocessing=self.is_compressing)

    def __init_new_round(self):
        self.state0 = self.env.reset()
        self.stacked_frames = util.stack_frames(self.stacked_frames, self.state0, True)
        return self.stacked_frames
    
    def __do_one_round(self, state0):
        frameskip_reward_accumulated = 0
        action = self.__predict_action(state0)
        state1, reward, done, info = self.env.step(action)
        if self.is_rendering:
            self.env.render()
        lifes = info['ale.lives']
        frameskip_reward_accumulated += reward
        self.stacked_frames = util.stack_frames(self.stacked_frames, state1, False)
        #in state0 perform action, get reward and be in state1 where the game might be done
        for j in range(3):
            state1, reward, done, info = self.env.step(action)
            if self.is_rendering:
                self.env.render()
            frameskip_reward_accumulated += reward
            self.stacked_frames = util.stack_frames(self.stacked_frames, state1, False)
            if done:
                break

        state1 = self.stacked_frames

        if info['ale.lives'] < lifes:
            lifes = info['ale.lives']
            frameskip_reward_accumulated += -1.0
            self.is_doing_nothing = True

        #check if it is the start of the round and the agent is doing nothing
        if action != 0:
            self.is_doing_nothing = False
        if self.is_doing_nothing and action == 0:
            self.doing_nothing_counter += 1
        if self.doing_nothing_counter > self.MAX_DOING_NOTHING:
            print ("Neural network got punished for doing nothing")
            frameskip_reward_accumulated = -1.0
            done = True

        frameskip_reward_accumulated = self.__cut_reward(frameskip_reward_accumulated)

        if self.is_training:
            if self.is_compressing:
                self.memory.add_with_compression((state0, action, reward, state1, done))
                self.memory.put_compressed_memory_into_memories()
            else:
                self.memory.add(state0, action, reward, state1, done)
        return state0, frameskip_reward_accumulated, done, info

    def __update_epsilon(self):
        self.epsilon -= self.decay_rate
        if self.epsilon < self.EXPLORE_END:
            self.epsilon = self.EXPLORE_END

    def __update_network(self, step):
        if step != 0 and step % self.NETWORK_UPDATE == 0:
            print("Updated target network")
            self.target_model.set_weights(self.model)
    
    def __cut_reward(self, reward):
        if reward > 0.0:
            return 1.0
        if reward < 0.0:
            return -1.0
        return 0.0

    def __predict_action(self, state):
        if random.random() < self.epsilon or not self.is_training or self.is_pretrain:
            choice = int(random.random() * len(self.action_space))
            action = self.action_space[choice]
        else:
            #normalize and expand dims, so that it is a batch of size 1
            print("NN taking action")
            state_input_form = np.expand_dims(state.astype('float32'), axis=0)
            choices = self.model.predict(state_input_form)
            choice = np.argmax(choices)
            action = self.action_space[choice]
        return np.argmax(action)

    def __pretrain(self):
        for i in range(self.PRETRAIN_LENGTH):
            while self.memory.length() <= self.PRETRAIN_LENGTH:
                if self.memory.length() == 0:
                    state0 = self.__init_new_round()
                state0, reward, done, info = self.__do_one_round(state0)

                if done:
                    state0 = self.__init_new_round()

    def __train(self):
        # print("Training with mini batch")
        batch = self.memory.sample(self.BATCH_SIZE)
        inputs = []
        targets = []
        loss = 0

        for dataset in batch:
            inputs.append(dataset['state0'].astype('float32'))
            training_state1 = np.expand_dims(dataset['state1'].astype('float32'), axis=0)

            training_state1_prediction = self.target_model.predict(training_state1)
            q_max = np.max(training_state1_prediction)

            t = list(self.model.predict(np.expand_dims(dataset['state0'].astype('float32'), axis=0))[0])
            if dataset['done']:
                t[dataset['action']] = dataset['reward']
            else:
                t[dataset['action']] = dataset['reward'] + self.GAMMA * q_max
            targets.append(t)

        inputs = np.asarray(inputs).squeeze()
        targets = np.asarray(targets).squeeze()
        return self.model.fit(inputs, targets, batch_size=self.BATCH_SIZE)

    def train(self):
        self.__pretrain()
        self.is_doing_nothing = True
        self.doing_nothing_counter = 0
        reward_per_episode = 0
        episode = 0
        f = open("log.txt", "a+")
        state0 = self.__init_new_round()
        for steps in range(self.MAX_STEPS):
            state0, reward, done, info = self.__do_one_round(state0)
            reward_per_episode += reward
            
            history = self.__train()
            history = history.history
            self.__update_epsilon()
            self.__update_network(steps)

            if((episode + 1) % 10 == 0):
                f.close()
                f = open("log.txt", "a+")

            if done:
                print("episode {0} reward {1}".format(episode, reward_per_episode))
                f.write("Episode: {0} Reward: {1} Steps: {2} Current Epsilon: {3} Loss: {4} Acc: {5}\n".format(episode, reward_per_episode, steps, self.epsilon, history['loss'][0], history['acc'][0]))
                episode += 1
                reward_per_episode = 0
                state0 = self.__init_new_round()
                

    def play(self):
        print("playing")
        self.is_training = False
        self.is_rendering = True
        self.target_model.load_weights("my_model_weights.h5")
        episode = 0
        reward_per_episode = 0
        state0 = self.__init_new_round()
        while episode < 100:
            state0, reward, done, info = self.__do_one_round(state0)
            reward_per_episode += reward
            if done:
                print("episode:{0} reward:{1}".format(episode, reward_per_episode))
                reward_per_episode = 0
                state0 = self.__init_new_round()
                episode += 1

