# HTW_AI_Gym
This is a project for a AI course at the HTW Berlin. I'm trying to implement an AI Agent that can play old Atari Games in Open AI Gym using a deep q neural network.
## Table of contents
1. [Introduction to the Project](#introduction-to-the-project)
2. [Explaining the Theory](#explaining-the-theory)
3. [The Journey](#the-journey)

## Introduction to the Project
#### Prerequisites
You need a linux machine, where Open AI Gym with the Atari package is installed.
You will also need Keras and Tensorflow-GPU installed, to be able to let your
Neural Network train on your graphics card. You could also use your CPU if your graphics card isn't powerful enough.
### How to use
Examples of typical usage:
```
python3 dqn_atari.py -h
```
will open a help menu with a description of all commands

```
python3 dqn_atari.py -r
```
This will start your agent train on Pong-v0 and save a log "Pong-v0_log.txt" and the weights "Pong-v0_weights.h5" every 10 episodes.
The -r command will enable rendering, so you can see what is happening.

```
python3 dqn_atari.py -env Breakout-v0 --continue Breakout-v0_weights.h5 --start 0.1 --end 0.0 -pre 300000
```
this command will load a session of Breakout-v0 with previously trained weights, do 300.000 (standard maximum of memories) pretrained memories, where the agent fully decides on the action taken and after that start at epsilon 0.1 and move down to epsilon 0.0 (probability of a random action) in 1.000.000 steps (standard configuration)

A list of available games can be found [here.](https://gym.openai.com/envs/#atari)

To generate some graphs there is a script "reward_visualizer.py". It takes 2 arguments -f [filepath] -a [number]
##### Example:
```
python3 reward_visualizer.py -f Pong-v0_log.txt -a 50
```
This will create a graphs for the Pong-v0_log.txt file with an average of the last 50 episodes.
## Explaining the Theory
### The Idea
My intention was to build a neural network that can play a game on an emulator. I got inspired by videos like [MarI/O](https://www.youtube.com/watch?v=qv6UVOQ0F44) and [Computer program that learns to play classic NES games.](https://www.youtube.com/watch?v=xOCurBYI_gY) At first I wanted to go with an evolutionoary neural network. While researching I came across something called "Deep Q Learning" and it was just what I was looking for. So that is what I will be explaining here: How does a computer learn to play games with just the raw pixels as input using DQN (Deep Q Learning).

### Q Learning
Q Learning (Q stands for quality) is a technique the mathematician [Andrey Markov](https://en.wikipedia.org/wiki/Andrey_Markov) invented. I won't go much into the math.

The Idea is simple: The game is in a state, the computer (the agent) will take an action and will get a reward for this action in that state. The result will be a new state.

!["Learning"](res/pictures/QLearning01.png)

The reward will tell the agent, if the decision was a good or a bad one. Now the agent has a new state to interact with. Over time the agent in which state to do an action, that will have a good reward. But you're not satisfied with a good reward, you want the best possible reward you can get in this specific state.
To achieve that there is something called a "Q Table". All possible states are the rows and the colums are the possible actions.
Each game will start with an empty table that will be filled through many actions, rewards and new states.

| States        | Do Nothing      | Move Right      | Move Left       |
| ------------- |:---------------:|:---------------:|:---------------:|
| State1        | 0               | 0               | 0               |
| State2        | 0               | 0               | 0               |
| State3        | 0               | 0               | 0               |
... <br>
... <br>
... <br>
Fast forward an table might look like that:
| States        | Do Nothing      | Move Right      | Move Left       |
| ------------- |:---------------:|:---------------:|:---------------:|
| State1        | 0               | 1               | -1              |
| State2        | -1              | 0.5             | 1               |
| State3        | 1               | 0.5             | -1              |
... <br>
... <br>
... <br>

Imagine you have a top down space game, where you have to maneuver your spaceship left and right to evade enemies. In the initial state doing nothing won't start the game, so it will not get a reward. Moving right will let you pick up an powerup and evade an enemy, while moving left will crush you right into an enemy. You are now in the State2, where an enemy is right in front of you. If you do nothing you will lose the game, if you move right you will evade, if you move left you will get a powerup and evade the enemy. To achieve a table like that, where each state has an optimal action you will need to update the Q table for each state for a given action and the resulting reward. First all actions have to be random, so the agent can explore the different actions for each state. Over time the agent will take less and less random actions and starts to act according to the Q-Table. To decide when the action has to be random and when it should be according to the Q-Table we introduce a variable called EPSILON. Epsilon will start at 1, meaning that 100% of actions are random and over time will decrease to a minimum like 0.1, meaning only 10% of the actions are random. This is called the "epsilon greedy strategy". Taking an action in a specific state will result in a reward. This reward has to be added to the corresponding action multiplied with a discount value called GAMMA. The discount value is to lower the learning rate of which action is good and which one is not.
Example in pesudo code: 
```
state0 -> action (Move Left) -> Result -1:

ActionSpace = Q-Table.getActionSpaceAt(state0)
Q-Table.at(state0) += (ActionSpace + (0, 0, -1) * GAMMA )
                    // (0, 0.34, -0.53) + (0, 0, -1) * 0.01
                    // (0, 0.34, -0.53) + (0, 0, -0.01)
                    // (0, 0.34, -0.54)
```

### Deep Q Learning
As a game can have millions and billions of different states a Q-Table isn't the apropriate thing to use. Thats where the Deep Neural Network comes into place. We try to create a Neural Network that will decide which action to take in which state. We effectively swap the Q-Table with a Neural Network.
As we work with pixel inputs we obviously need a convolutional neural network. Convolutional Neural Networks are used to recognize features in pictures. The structure of the neural network is the same as described in the [Mnih et al Paper 2015](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). That means:
1 convolutional input layer, 2 convolutional layers, 1 256 fully connected layer and 1 output layer, that has as many outputs as there are actions.
In Keras:
```python
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
                   optimizer=keras.optimizers.RMSprop(0.00025),
                   metrics=['accuracy'])
```
As you can see the input is a 4 x 84 x 84 picture.
As commonly done with pictures for convolutional neural network we preprocess our input. The standard input is a 210 x 160 x 3 picture.
Firstly, to reduce dimensions, take away the color. As a result we have a 210 x 160 picture. Secondly, we downscale it t0 110 x 84. Then we crop it to 84 x 84. But the Input is 4 x 84 x 84. The reason for that is, you want to give the agent a feel for direction and speed. As example pong: If you have a single picture you can't tell if the ball is moving, if you have to consecutive pictures you can tell the direction of the ball, if you have 3 pictures, you can tell how fast the ball is moving. But as in the mnih et al 2015 paper shown we take 4 consecutive pictures.

![Consecutive Frames](res/pictures/consecutiveframes.png)

To achieve those 4 frames we just stack them. As mentioned in the mnih paper, we will let the agent perform an action and repeat that action 3 times. Therefore We have 4 consecutive frames and we put them together in one array with the dimensions (4, 84, 84). The result is a state consisting of 4 consecutive frames and a next_state being the next consecutive 4 frames.


### Replay Memory
As the agent should be able to learn to handle many differen situations. For that it needs a memory. To achieve a memory we implement something called a "Replay Buffer". A memory consists of the following things: An initial state, the action taken, the received reward, the resulting state and the info if the game was done after that action. As tuple: (State0, Action, Reward, State1, Done). In the mnih paper the replay buffer has a size of 1 million memories. My approach has one big drawback: 84 x 84 x 4 bytes = 28224 bytes per state or 26.29gb for 1 million states. I tried the method described in [this](https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb) git repository. This approach uses the fact, that 4 consecutive frames are a state + 1 frame is the next_state. So you reduce the memory quite a bit. But it seems like it wasn't compatible with my architecture, so I threw it out without looking further into it. It probably had something to do that I used 4 frames for current_state and the next 4 frames for next_state, so 2 states took a total of 8 different frames. My second attempt was compressing with RLE. That worked quite well, but it was awkwardly slow, I tried to speed it up using multiprocessing in python but it wasn't successful. So I had to live with a maximum of 300.000 Memories max.


### Training
To train the agent a sample consisting of 32 random memories is taken from the memory. They are random to break correlations between consecutive memories, meaning that the agent doesn't learn to expect something if he performce a certain action in a certain state. As example: The agent plays Breakout: The first contact with ball and the bat always results in the ball moving to the right, hitting the wall, breaking one brick and landing on the left bottom corner. The agent would learn to always go to the left corner after hitting the ball, because it rather learns consecutive patterns than behaviour depending on the situation.

To train properly it is recommended to have 2 Neural Networks, one that is trained (called model in my program) and one that creates targets (called target_model in my program). The reason for that is, that the target_model will have more consistent targets. If you create targets with the model you are training you could create a feedback loop, because we try to get closer to the target, but with every iteration the target moves further away.

That covers all the theory, now to the architecture of the program.

### Sources:
* https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
* https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
* https://medium.freecodecamp.org/an-introduction-to-reinforcement-learning-4339519de419
* https://medium.freecodecamp.org/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe
* https://medium.freecodecamp.org/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8
* https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682
* https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
* https://towardsdatascience.com/tutorial-double-deep-q-learning-with-dueling-network-architectures-4c1b3fb7f756
* https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb
* https://github.com/gsurma/atari/blob/master/atari.py
* https://github.com/gsurma/atari/blob/master/game_models/ddqn_game_model.py
* https://towardsdatascience.com/atari-reinforcement-learning-in-depth-part-1-ddqn-ceaa762a546f
* https://github.com/keras-rl/keras-rl
* https://github.com/danielegrattarola/deep-q-atari
* https://github.com/ShanHaoYu/Deep-Q-Network-Breakout
* https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Deep%20Q%20Learning/Space%20Invaders/DQN%20Atari%20Space%20Invaders.ipynb
* https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Deep%20Q%20Learning/Doom/.ipynb_checkpoints/Deep%20Q%20learning%20with%20Doom-checkpoint.ipynb

## The Journey
#### Day One
Today is the 9th of march. I finally got time to start working on this project. It has been a few weeks since I read the last time
about Deep Q NNs, so I started the day with reading again through the impressively good [articles](https://medium.freecodecamp.org/an-introduction-to-reinforcement-learning-4339519de419 "First Article of this series") 
of Thomas Simonini. Right now I am at the [third](https://medium.freecodecamp.org/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8 "Starting to learn about Deep Q NN")
article again and try to understand how it all works. While reading this I'm starting to try to implement one myself into my project.

I've gone through the tutorials now and realized that I need to find a way to feed my keras neural network while it is running. It is obvious in hindsight, but I didn't think of it before. I'm quite exhausted now, so I'll find a solution tomorrow.

#### Day Two
10th of march. The way I used Keras before was just with the train method. I've found a tutorial where someone is using the train_on_batch method. That is exactly what I needed, but I need to get everything ready first and I'm still not quite confident in my understanding of this topic. So I'm still following the tutorials from Thomas Simonini. But as the Doom-Example is way over the top for my purpose I started looking at the source of this [deep q learningspace invaders](https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Deep%20Q%20Learning/Space%20Invaders/DQN%20Atari%20Space%20Invaders.ipynb) and implemented it the same way just in Keras. I'm not quite done yet. I'm now at the part, where the AI starts learning. For that I will need the train_on_batch method. But as those topics are quite complicated to grasp and my concentration is depleted I will pause for now. Maybe I'll continue later this day.

#### Day Three
12th of march. Yesterday I didn't really do anything but reading through source code and fiddling with the code. Turns out I didn't really understood how numpy arrays work, as I gave up on trying to get my training batch of states into one array. So today I took a look at those numpy arrays and came to the realization that I just have to do the same thing I already did with the frame stacks. The solution is probably horrible but it works. So Today I could finally start training my AI-Agent... I think. I didn't endure long enough to really see if something is happening, because it was training somewhat slow... so I checked if Keras is using my GPU but nope, it didn't. So I took a look into that and now I can't run my script anymore. So I've been going through tutorials on how to properly install tensorflow-gpu, cuda, nvidia drivers and so on for about 3 hours... just love it.
In the end tensorflow finally saw that cuda is indeed installed, but some kernels had different versions. So I guess I'm doing it all over again. Great.

#### Day Four
13th of march. Yesterday around midnight I managed to fix the nvidia, cuda, keras errors. For anyone who needs help with this aswell [this](https://github.com/rnreich/ubuntu-tensorflow-gpu-all-versions) is a great source for help. Protip: Scroll down first - there is a "fast route"... And the currently the correct version is 1.12.0 with up to date keras and graphics cards drivers with compute capabbalities of 6.1. So after it worked I went to bed. The next morning I wanted to let my AI Agent train while I'm at work - but after a while my computer wouldn't work anymore. I checked the ram and yep - it was fully taken. Apparently you shouldn't trust tutorials. The source of evil was the memory buffer, which the tutorial set to a size of 1 million. To get a good size with my ram I just printed the length of the memory list and watched my RAM fill up. It took around 2500 memories before my RAM was full, so I decided that 2000 memories where a good starting point. After the neural net worked I wanted to visualize some things, so I looked into drawing graphs in python. Matplotlib is exactly what you need for that. Gladly I found a bug today. As I tracked some stuff I discovered that the probability was starting at 0.99, was going down and jumped up again. Turned out I reset the step counter each episode which is not what I want, so I moved it up. Tomorrow I will let the ai agent train 1000 episodes of breakout or until I come back from work. Hopefully it will somewhat work and I just have to tweak it a little bit. We'll see. 

#### Day Five
14th of march. The Neural Network I made is garbage. As you can see [here](res/first_try/ai_agent_graph_first_try.png) it didn't quite learn to do anything. Also I realized that I need to add how much reward it is getting each episode. 
So it's back to the drawing board. But I have a better understanding of this topic. Nevertheless I will read again about it and look at repositories and learn from them.

#### Day Six
15th of march. I started reading the [minh paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) and testing a dqn agent from the [keras-rl repository](https://github.com/keras-rl/keras-rl "keras-rl") I stumpled across today. While the Agent was learning I read through the paper and wrote down some notes. The first thing I noticed was that the learning was **much** faster than the one on my neural network. Secondly they did indeed use 1 million memories, but I saved them as float32 while they did as byte8. That happens when you're not comfortable with a programming language. Also their input image was 84x84x4 (reason is, that the convolution they used needed a square array) while mine was not. I will change that in the next version. And their neural network hat different values on the filters, kernels and strides. I probably take the one from the [keras-rl repository](https://github.com/keras-rl/keras-rl "keras-rl") mentioned before. Another big difference was that they used frame skips! What they did was letting the AI Agent perform an action and then repeating this action the next 3 frames. Except for Space Invaders. That could be one reason why my neural network was quite slow. The next step from now is to look through the code of this keras-rl repo and looking at how they did things. And I probably should learn how to manipulate numpy arrays correctly, how to stack them, how to change their content from e.g. float32 to byte8 and so on.

#### Many days later
23rd of march. I spend the last days with taking a deep look into my code and really understand what is happening. And I came to the conclusion that it should work, but I probably wasn't training it long enough. I changed some hyper parameters, deleted some unnecessary code/comments. Also I got removed the normalization of the input images, because I got an extremely low total_loss after each training session. Now I got something between x - 0.00x. And then I let it train for the past three days. So in total that should be around 72 hours. Oh, and I also did double deep q learning this time, because it is quite easy to implement. It just means you have a neural network that you train and one that decides (called target network). The weights of the target network get periodically updated to match the ones of the nn that decides the actions. The reason to do it that way is for stability. [Here](https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682) is a good article explaining the reasons on why to use ddqn.
The results are following:
#### Rewards per Episode averaged by 10
!["episode_rewards"](res/second_try/iteration_rewards10.png)
#### Rewards per Episode averaged by 100
!["episode_rewards"](res/second_try/iteration_rewards100.png)
#### Unfortunately pyplot automatically plots from lowest to highest x-value, so you have to read the following graphs from right to left.
#### Rewards dependent on the probability the neural network decides the action averaged by 10
!["probability_rewards"](res/second_try/probability_rewards10.png)
#### Rewards dependent on the probability the neural network decides the action averaged by 100
!["probability_rewards"](res/second_try/probability_rewards100.png)

As you can see this time the neural network did learn *something*. It peaked when it had around 37% chance of deciding what to do. As it got more and more resposibility the rewards started fluctuating. At this time I didn't know why and I suspected the lack of memories. I "only" had 200.000 memories. I thought it might forget what it learned before and starts learning it all over again, but instead of getting better it just forgets the status quo and learns to play the game again. But after the last low I stopped the learning. I saved the weights of the last peak before and I wanted to see what the agent actually did there. And surprisingly it did quite good. It doesn't look like random actions anymore, it knew what to do. After it got some points and lost a live it just stoped doing anything, because it was satisfied with what it got. So after the 50000 steps limit a new episode started. And it learned from those many episodes of doing nothing to do... well, nothing. That is why it started fluctuating. And as the random actions got lower and lower it couldn't explore the environment anymore and it dependet on memories that got slowly overwritten by doing nothing. I think the following are steps I need to take:
- the agent needs to be punished for doing nothing for a longer period of time when a game starts
- change the structure of the program from episodes/steps to just steps, as many repositories doing ddqn do it like that. It makes it easier to adjust hyper parameters
- change the decay of exploration to a more linear and approach, so I can actually tell after how many steps the lower limit is reached.
- change some hyper paremets to fit the new approach of just counting steps
- expand the memories a little bit more (low priority, as it seems to work nevertheless)

Just a clarification: A step is one action the neural network decides, not the number of frames that have passed.

#### Many days later + 1
24th of march. I realized a big mistake I did before. When I introduced frame skipping I also introduced a bug into my neural network: When the environment granted a reward for destroying a block the reward was immediately overwritten by zero points for the next frame if it wasn't the last frame in the sequence. Therefore the agent didn't get rewards for doing good. Luckily I just realized that while re-implementing the logic. I hopefully fixed that by accumulating the rewards per sequence and then clipping it to 1. Also I've implemented a punishment for doing nothing for a certain amount of steps at the start of a round. I haven't tested it so I hope I didn't do it wrong. I also changed the calculation of the probability for the agent to take actions. Now I am able to define a certain amount of steps after which the highest probability is reached.

#### Many days later + 1
25th of march. Results of the test:
#### Breakout. The results averaged by the last 100 episodes:
!["episodes_rewards100"](res/third_try/average_100/episodes_rewards100.png)
!["steps_rewards100"](res/third_try/average_100/steps_rewards100.png)
It is learning.. something. Unfortunately it seems to get stuck. But it's doing better than before. Also I've fixed some problems with the diagram that I had in the last pictures. Also I changed the script which reads the trainings log and draws the diagrams to take in arguments (filename and average). But Unfortunately I made a mistake with the "doing nothing" thing, because the "done" flag is obviously not set when losing a life. Therefore it did nothing. To get the information about the lifes I need to get it from the "info" dictionary the env.step(action) method gives back. I get the current lifes by calling info['ale.lives'].

#### Many days later + 3
27th of march. I've been testing a lot recently. I came to the conclusion that Breakout might be somewhat hard, so I started testing pong with my neural network, because it also has losses. I tried normalizing the input again, but the loss got so low, that the weights changed so slowly that I couldn't see any progress over the night. 
#### Pong. The results averaged by the last 50 episodes:
![episodes_rewards50](res/second_pong_with_normalized_input/average_50/episodes_rewards50.png)
As you can see it didn't learn anything therefore I stopped that and turned it back to the not normalized input.
#### Pong. The results averaged by the last 50 episodes:
![episodes_rewards50](res/third_pong/average_50/episodes_rewards50.png)
Result of letting the nn train pong for like... ~30 hours. Averaged by the last 50 episodes.
As you can see it did learn something. But still it's incredibly time consuming and slow. I suspect the lack of memories, thats why I will start explicitly reading about how to implement the memory efficiently.

#### Many days later + 4
28th of march. I don't know why I didn't calculate the amount of RAM the memory takes earlier. One picture is a 84 x 84 uint8 matrix. 1 Million of those take 6.57gb of RAM. But as I those memories are a stack of 4 pictures it would take more than 26gb of RAM. Unfortunately that is a lot more than I have. So I looked again what other people did. Most of them didn't seem to bother, but some did. And I found [this](https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb) and implemented the memory as shown. Unfortunately my network didn't learn anything after testing two different games (pong and breakout) for about 10 hours each. So I'm back at where I started. But today I got an Idea. I could try to concurrently compress the pictures with RLE. It's a quite simple algorithm, but it should work pretty well as most parts of the pictures are of the same color. I'll probably try to implement it over the weekend. For that I'll have to get familiar with concurrency in python. Doing that in sequence would probably slow too much down. Speaking of slowing down. I also looked into things that could slow my program down. Something that happens quite often is generating random numbers and I tested different methods of generating them. Namely random.random(), random.randrange(), np.random.randint() and np.random.rand(). I didn't test random.randint() because I already read that it is quite slow. The testcase was 1 million times 32 random numbers. Surprisingly the fastest method was random.random, even though I needed to multiply the result with a range and cast it into an int. E.q. int(random.random() * 1000000) would give an integer between 0 and 1 million - 1 (as the range of random.random is [0, 1)).

#### Many days later + 6
30th of match. Before I implement RLE I wanted to test the network on Breakout-v0 again with keeping track of the lifes and rounds it does nothing. I implemented a punishmend for losing a life, because previously losing a life in Breakout didn't do anything at all. There was just reward for breaking blocks, that's why it was so hard to train the agent. Also I added punishment for doing nothing for 30 decisions at the start of a round (the ball only spawns if you decide to move in either direction after losing a life).
The results of ~ 30 hours of training:
![steps_rewards_50](res/fourth_try/average_50/steps_rewards50.png)
As you can see, the lowest value is now -2, because I added punishment. It is now possible to score negative rewards after a game. The agent learned faster and better than any of my previous agents. The maximum score I saw was 33 - even though he gets punished for losing a life. That was far more than the previous agents could do. 
The agent had a memory of 300.000, which was taking around 10gb of RAM. The agent had a steady increase of rewards, after 1.000.000 steps it seemed to stagnate. That was because the agent started to do nothing for several rounds. After 200.000 steps it learned that he gets punished for doing nothing. Sometimes you could see him wait for a period of time and then start moving - fearing the upcoming punishment if he didn't move. After that it seems that he reached the peak of what he is capable of, so I stopped the training. While the agent was training I already implemented an RLE compressing and decompressing algorithm. Now I had the opportunity to test if everything worked - and it did. But it consumed far more memory than I anticipated. Even more than the uncompressed version. That was because I used tuples to save compressed data and added those tuples to a list. E.g. 124, 124, 124, 124, 124 would become (124, 5) => the number 124 is written down 5 times. I researched a little bit and learned that lists have quite the overhead because they are mutable containers (e.g. you can add items at runtime). Therefore I tried converting them into numpy arrays before I returned the compressed image. And finally it worked. It is quite slow, but it consumes far less memory than before. Now I'm trying to use multiprocessing to concurrently compress and decompress data. As I can mess up a lot of code I want to push the current state before starting to implement multiprocessing. It is not hard to go back to not using compression currently.

#### Final Day
3rd of April.
Firstly, I want to correct myself. I spoke of concurrency the last time, but that is not what I need. I needed parallelism, because my problem is CPU-Bound and not IO-Bound. In Python multithreading means running multiple threads on the same cpu core, while multiprocessing means running multiple processes on several cpus cores. Unfortunately I didn't achieve a speed that was acceptable so I have to live with only 300.000 memories with 16gb RAM. Maybe one day, when I know Python better I will be able to make it work.
Two days ago I started a training session until today, unfortunately I had some little mistakes in my new architecture. As result the NN didn't learn anything. But I fixed that now. I also tested the new architecture with already trained agents and saw that it didn't work as planned. When Playing I loaded the weights to the target_model and not to the model, so I fixed that aswell. Now everything should be working hopefully. Now I will let the Pong-Agent train some more with the weights of the previously trained agent, as I don't have much time left unfortunately. I will write some documentation into the code now but I won't change any code unless I find another bug. I will now write a "how to use" and some of the theory behind DQN learning.
