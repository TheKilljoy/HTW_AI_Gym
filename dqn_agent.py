import dqn_network
import util

import memory
import random
import numpy as np

class DqnAgent:
    """
    This class represents an DQN Agent, that is initialized with
    the user defined parameters or with standard parameters.
    """
    def __init__(self, input_shape, env, MAX_STEPS, MAX_DOING_NOTHING, 
                 FRAME_SIZE, BATCH_SIZE, MEMORY_SIZE, PRETRAIN_LENGTH,
                 NETWORK_UPDATE, EXPLORE_START, EXPLORE_END, EXPLORE_STEPS,
                 GAMMA, is_training, is_rendering, path_to_weights, is_compressing=False):
                 #WARNING! IT IS NOT RECOMMENDED TO USE COMPRESSION, AS IT IS REALLY SLOW
        
        self.BATCH_SIZE = BATCH_SIZE
        self.is_compressing = is_compressing
        #initialize memory asap, because new processes get an instance of the current state, therefore take a lot of memory
        self.memory = memory.Memory(max_size=MEMORY_SIZE, is_multiprocessing=self.is_compressing, batch_size=self.BATCH_SIZE)
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
        self.is_rendering = is_rendering
        
        self.decay_rate = (EXPLORE_START - EXPLORE_END)/EXPLORE_STEPS
        self.doing_nothing_counter = 0
        self.is_doing_nothing = True
        self.stacked_frames = np.zeros(input_shape, dtype=np.uint8)

        self.is_continue = False

        self.model = dqn_network.DqnNetwork(input_shape, self.action_space)
        if path_to_weights != "NONE":
            self.model.load_weights(path_to_weights)
            print("Successfully loaded weights")
            self.is_continue = True
        self.target_model = dqn_network.DqnNetwork(input_shape, self.action_space)
        self.target_model.set_weights(self.model)

    def __init_new_round(self):
        """
        Initialize a new round.
        Returns the initial state
        """
        state = self.env.reset()
        self.stacked_frames = util.stack_frames(self.stacked_frames, state, True, self.FRAME_SIZE)
        return self.stacked_frames
    
    def __do_one_round(self, state0):
        """
        Do one step in the environment depending on the given state0.
        Automatically adds experience to the memory.
        Returns the state after successfully playing.
        """
        frameskip_reward_accumulated = 0
        action = self.__predict_action(state0)
        state1, reward, done, info = self.env.step(action)
        frameskip_reward_accumulated += reward
        self.stacked_frames = util.stack_frames(self.stacked_frames, state1, False, self.FRAME_SIZE)
        if self.is_rendering:
            self.env.render()
        lifes = info['ale.lives']
        #in state0 perform action, get reward and be in state1 where the game might be done
        for j in range(3):
            state1, reward, done, info = self.env.step(action)
            frameskip_reward_accumulated += reward
            self.stacked_frames = util.stack_frames(self.stacked_frames, state1, False, self.FRAME_SIZE)
            if self.is_rendering:
                self.env.render()
            if done:
                break
        #the new state after performing 4 actions
        state1 = self.stacked_frames
        #if a life is lost punish the neural network
        if info['ale.lives'] < lifes:
            lifes = info['ale.lives']
            frameskip_reward_accumulated = -1.0
            self.is_doing_nothing = True

        #check if it is the start of the round and the agent is doing nothing
        if action != 0:
            self.is_doing_nothing = False
        if self.is_doing_nothing and action == 0:
            self.doing_nothing_counter += 1
        if self.doing_nothing_counter > self.MAX_DOING_NOTHING:
            #if the agent is doing nothing for too long punish him
            #and restart the round
            print ("Neural network got punished for doing nothing")
            frameskip_reward_accumulated = -1.0
            done = True

        frameskip_reward_accumulated = self.__cut_reward(frameskip_reward_accumulated)

        if self.is_training:
            ##### as compression is not used this part is irrelevant
            if self.is_compressing:
                self.memory.add_with_compression((state0, action, frameskip_reward_accumulated, state1, done))
                self.memory.put_compressed_memory_into_memories()
            ############################################################
            else:
                self.memory.add(state0, action, frameskip_reward_accumulated, state1, done)
        return state1, frameskip_reward_accumulated, done, info

    def __update_epsilon(self):
        """
        Updates the epsilon value
        """
        self.epsilon -= self.decay_rate
        if self.epsilon < self.EXPLORE_END:
            self.epsilon = self.EXPLORE_END

    def __update_network(self, step):
        """
        Updates the target model (the one, that makes the decisions while training)
        with the weights of the model that is trained.
        """
        if step != 0 and step % self.NETWORK_UPDATE == 0:
            print("Updated target network")
            self.target_model.set_weights(self.model)
    
    def __cut_reward(self, reward):
        """
        Cuts the reward if it is above 1.0 or below -1.0
        to 1.0 or -1.0 respectively
        """
        if reward > 0.0:
            return 1.0
        if reward < 0.0:
            return -1.0
        return 0.0

    def __predict_action(self, state):
        """
        Predicts an action for the agent for the given state.
        Depending on the current epsilon value the action can
        be chosen randomly or Q-greedy
        """

        #if the agent is playing always let the NN decide
        if not self.is_training or self.is_continue:
            state_input_form = np.expand_dims(state.astype('float32'), axis=0)
            choices = self.model.predict(state_input_form)
            choice = np.argmax(choices)
            action = self.action_space[choice]
            return np.argmax(action)            

        if random.random() < self.epsilon:
            choice = int(random.random() * len(self.action_space))
            action = self.action_space[choice]
        else:
            #normalize and expand dims, so that it is a batch of size 1
            state_input_form = np.expand_dims(state.astype('float32'), axis=0)
            choices = self.model.predict(state_input_form)
            choice = np.argmax(choices)
            action = self.action_space[choice]
        return np.argmax(action)

    def __pretrain(self):
        """
        Starts the pre-training session to fill up the memories
        of the agent.
        """
        while self.memory.length() <= self.PRETRAIN_LENGTH:
            if self.memory.length() == 0:
                state = self.__init_new_round()
            state, reward, done, info = self.__do_one_round(state)
            if done:
                state = self.__init_new_round()
                self.doing_nothing_counter = 0
                self.is_doing_nothing = True

    def __train(self):
        """
        This private method trains the NN with a batch
        It returns a the "history" object a keras NN
        returns when training it with the "fit" method
        """
        batch = self.memory.sample(self.BATCH_SIZE)
        inputs = []
        targets = []

        for dataset in batch:
            inputs.append(dataset['state0'].astype('float32'))

            training_state1 = np.expand_dims(dataset['state1'].astype('float32'), axis=0)
            training_state1_prediction = self.target_model.predict(training_state1)
            q_max = np.max(training_state1_prediction)

            t = list(self.model.predict(np.expand_dims(dataset['state0'].astype('float32'), axis=0))[0])
            # do the bellman equation with discounted reward: 
            # Reward = reward_t1 + reward_t2 * GAMMA
            # We use t[dataset['action']], because self.model.predict gives us a list of the probability of each action e.q. [0.2, -0.5, 0.54, 0.6]
            # and now we take with t[dataset['action']] the q value at position 'action' from the predicted outcome 
            # and update it with the current q value and the predicted target q value
            # (we update the "q-table")
            if dataset['done']:
                t[dataset['action']] = dataset['reward']
            else:
                t[dataset['action']] = dataset['reward'] + self.GAMMA * q_max
            targets.append(t)

        inputs = np.asarray(inputs).squeeze()
        targets = np.asarray(targets).squeeze()
        return self.model.fit(inputs, targets, batch_size=self.BATCH_SIZE)

    def train(self):
        """
        Start a new training session with the agent.
        Every 10 Episodes the current weights will be saved
        under the name "EnvName-vx_weights.h5".
        Every 10 Episodes the logfile will be updated.
        The logfile is called "EnvName-vx_log.txt"
        """
        self.__pretrain()
        self.is_continue = False
        self.is_doing_nothing = True
        self.doing_nothing_counter = 0
        reward_per_episode = 0
        episode = 0
        f = open("{0}_log.txt".format(self.env.unwrapped.spec.id), "a+")
        state = self.__init_new_round()
        for steps in range(self.MAX_STEPS):
            state, reward, done, info = self.__do_one_round(state)
            reward_per_episode += reward
            
            history = self.__train()
            history = history.history
            self.__update_epsilon()
            self.__update_network(steps)

            #save everything every 10th episode.
            if episode % 10 == 0:
                f.close()
                f = open("{0}_log.txt".format(self.env.unwrapped.spec.id), "a+")
                self.model.write_weights("{0}_weights.h5".format(self.env.unwrapped.spec.id))

            if done:
                print("episode {0} reward {1}".format(episode, reward_per_episode))
                f.write("Episode: {0} Reward: {1} Steps: {2} Current Epsilon: {3} Loss: {4} Acc: {5}\n".format(episode, reward_per_episode, steps, self.epsilon, history['loss'][0], history['acc'][0]))
                episode += 1
                reward_per_episode = 0
                self.doing_nothing_counter = 0
                self.is_doing_nothing = True
                state = self.__init_new_round()
                
    def play(self):
        """
        Let the agent play with pretrained weights.
        The Weights have to have the name "GameName-vx_weights.h5"
        """
        self.is_training = False
        self.is_rendering = True
        self.model.load_weights("{0}_weights.h5".format(self.env.unwrapped.spec.id))
        episode = 0
        reward_per_episode = 0
        state = self.__init_new_round()
        while episode < 100:
            state, reward, done, info = self.__do_one_round(state)
            reward_per_episode += reward
            if done:
                print("episode:{0} reward:{1}".format(episode, reward_per_episode))
                reward_per_episode = 0
                state = self.__init_new_round()
                episode += 1
        self.model.write_weights("{0}_weights.h5".format(self.env.unwrapped.spec.id))

