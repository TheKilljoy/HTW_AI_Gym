import argparse
import dqn_agent
import gym

parser = argparse.ArgumentParser(description="Starts an DQN Agent to learn a given atari environment with pixels as input. Standard environment is Pong-v0")
parser.add_argument("--environment", "-env", default="Pong-v0", dest="env", type=str, help="chose an atari environment with pixel input")
parser.add_argument("--render", "-r", action="store_true", dest="render", help="Set  this flag to let the training session be rendered")
parser.add_argument("--steps", "-st", default=5000000, dest="steps", type=int, help="the duration in which the agent learns")
parser.add_argument("--nothing", "-n", default=30, dest="nothing", type=int, help="the maximum steps an aget may do nothing at the begining of each round")
parser.add_argument("--memory", "-m", default=1000000, dest="memory", type=int, help="the size of the memory of the agent")
parser.add_argument("--pretrain", "-pre", default=50000, dest="pretrain", type=int, help="the amount of memories the agent gathers before starting to train")
parser.add_argument("--gamma", "-g", default=0.99, dest="gamma", type=float, help="the discount factor of the bellman equation")
parser.add_argument("--update" "-u", default=50000, dest="update", type=int, help="the number of steps until the target network is updated")
parser.add_argument("--start", "-s", default=1.0, dest="start", type=float, help="exploration start. 1 = 100%% random actions")
parser.add_argument("--end", "-e", default=0.1, dest="end", type=float, help="the exploration rate doesn't go below this value")
parser.add_argument("--exploresteps", "-exs", default=1000000, dest="explore", type=int, help="the amount of steps the agent has to take until the end of the exploration rate is reached")
parser.add_argument("--play", "-p", action="store_true", dest="play", help="set this flag if you just want to let the agent play")
parser.add_argument("--framestacksize", "-fss", default=4, dest="framestacksize", type=int, help="the size of the frames stacked together to give the agent a feeling for movement")
parser.add_argument("--batch", "-b", default=32, type=int, dest="batch", help="The size of the trainingsbatch")

args = parser.parse_args()
env = gym.make(args.env)

if(args.play):
    print("Just letting the Agent play")
else:
    print( "Starting a new training session with the following configurations:\n\
            environment: {0}\n\
            memory size: {1}\n\
            duration in steps: {2}\n\
            pre training memory filling: {3}\n\
            frame stack size: {4}\n\
            explore start: {5}\n\
            explore end: {6}\n\
            explore steps: {7}\n\
            gamma: {8}\n\
            batch size: {9}\n\
            training: {10}\n\
            rendering: {11}\n\n\
            initializing neural network...\n".format(
                args.env, args.memory, args.steps,
                args.pretrain, args.framestacksize,
                args.start, args.end, args.explore,
                args.gamma, args.batch, args.play, args.render))

atari_agent = dqn_agent.DqnAgent(input_shape=(args.framestacksize, 84, 84), env=env, MAX_STEPS=args.steps,
                                 MAX_DOING_NOTHING=args.nothing, FRAME_SIZE=args.framestacksize,
                                 BATCH_SIZE=args.batch, MEMORY_SIZE=args.memory, PRETRAIN_LENGTH=args.pretrain,
                                 NETWORK_UPDATE=args.update, EXPLORE_START=args.start, EXPLORE_END=args.end,
                                 EXPLORE_STEPS=args.explore, GAMMA=args.gamma, is_training=not args.play, is_rendering=args.render)

if not args.play:
    atari_agent.train()

atari_agent.play()