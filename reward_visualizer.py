import re
import matplotlib.pyplot as plt
import argparse
import os

iteration = 0
reward = 0.0
probability = 0.0
loss = 0.0
step = 0
iterations = []
rewards = []
probabilities = []
losses = []
steps = []

avg = 100

def extract_values(line):
    values = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    return (values[0], values[1], values[2], values[3], values[4])

def sum_values(temp_it, temp_re, temp_step, temp_pr, temp_loss):
    global probability
    global reward
    global step
    global iteration
    global loss
    iteration = int(temp_it)
    step = int(temp_step)
    reward += float(temp_re)
    probability += (1.0 - float(temp_pr))
    loss += float(temp_loss)

def write_to_list(average):
    global reward
    global probability
    global step
    global iteration
    global loss
    reward /= average
    probability /= average
    loss /= average
    iterations.append(iteration)
    rewards.append(reward)
    steps.append(step)
    probabilities.append(probability)
    losses.append(loss)

def reset_values():
    global reward
    global loss
    reward = 0
    loss = 0


parser = argparse.ArgumentParser(description="Visualize the results of the ai agent. Write the path to the rewards.txt as argument. You can optionally give the average as well.")
parser.add_argument("-file", "-f", nargs=1, dest="filepath", type=str, help="the path to the rewards.txt")
parser.add_argument("-average", "-avg", "-a", nargs='?', default=100, type=float, dest="average", help="the amout of averaging e.g. reward is summed 100 times and then dividided by 100")

args = parser.parse_args()
filepath = args.filepath
avg = args.average
fobj = open(filepath[0], "r")

for line in fobj:
    if iteration % avg == 0:
        counter = 0
        write_to_list(avg)
        reset_values()
    temp_iteration, temp_reward, temp_step, temp_probability, temp_loss = extract_values(line)
    sum_values(temp_iteration, temp_reward, temp_step, temp_probability, temp_loss)

#if the folder doesn't exist, create it
if not os.path.exists("./average_{0}".format(int(avg))):
    os.makedirs("./average_{0}".format(int(avg)))
########## episodes ###############
plt.figure(figsize=(30,10))
plt.plot(iterations, rewards)
plt.grid(True)
plt.xlabel("episodes")
plt.ylabel("reward")
plt.savefig('./average_{0}/episodes_rewards{0}.png'.format(int(avg)))
plt.clf()

plt.figure(figsize=(30,10))
plt.plot(iterations, probabilities)
plt.grid(True)
plt.xlabel("episodes")
plt.ylabel("probability neural network acts")
plt.savefig('./average_{0}/episodes_probability{0}.png'.format(int(avg)))
plt.clf()

plt.figure(figsize=(30,10))
plt.plot(iterations, losses)
plt.grid(True)
plt.xlabel("episodes")
plt.ylabel("losses")
plt.savefig('./average_{0}/episodes_loss{0}.png'.format(int(avg)))
plt.clf()

plt.figure(figsize=(30,10))
plt.plot(probabilities, rewards)
plt.grid(True)
plt.xlabel("probability neural network acts")
plt.ylabel("reward")
plt.savefig('./average_{0}/probability_rewards{0}.png'.format(int(avg)))
plt.clf()
############## steps #################
plt.figure(figsize=(30,10))
plt.plot(steps, rewards)
plt.grid(True)
plt.xlabel("steps")
plt.ylabel("reward")
plt.savefig('./average_{0}/steps_rewards{0}.png'.format(int(avg)))
plt.clf()

plt.figure(figsize=(30,10))
plt.plot(steps, probabilities)
plt.grid(True)
plt.xlabel("steps")
plt.ylabel("probability neural network acts")
plt.savefig('./average_{0}/steps_probability{0}.png'.format(int(avg)))
plt.clf()

plt.figure(figsize=(30,10))
plt.plot(steps, losses)
plt.grid(True)
plt.xlabel("steps")
plt.ylabel("losses")
plt.savefig('./average_{0}/steps_loss{0}.png'.format(int(avg)))
plt.clf()
