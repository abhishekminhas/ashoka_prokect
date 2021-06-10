""" 	REINFORCE ALGORITHM, will add abseline and then A2C
		CALLING .DETACH() ON THE PREDICTIONS DELETES THE GRADIENTS, COPY THE PREDICTION FOR LOSS CALCULATION
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import random
import collections
import numpy as np
from collections import namedtuple, deque
from snake_matrix import SnakeGame
import statistics



GAMMA = 0.99
LEARNING_RATE = 0.02
NUM_EPISODES = 4
EPOCHS = 100
grid_size = 5

class PGN(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1= nn.Linear(grid_size*grid_size,64)
		self.fc2= nn.Linear(64,64)
		self.fc3= nn.Linear(64,64)
		self.fc4= nn.Linear(64,3)
	def forward(self,x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		# x= F.softmax(x, dim=1) # view(-1, 25) is important for this
		return x


env = SnakeGame(grid_size)
net = PGN()
for param in net.parameters():
	param.requires_grad = True
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

def calc_qvals(rewards):
	res = []
	sum_r = 0.0
	for r in reversed(rewards):
		sum_r *= GAMMA
		sum_r += r
		res.append(sum_r)
	return list(reversed(res))
def convert_action(x):
	actionsss_list = ['forward', 'left', 'right']
	return actionsss_list[x]


# for envs in range(NUM_EPISODES)
env.reset()
# MAIN LOOP 
reward_for_loop = 0

for _ in range(10000):
	counter = 0
	optimizer.zero_grad()
	mean_epoch_reward = []
	batch_log_probs = []
	mean_batch_reward =[]
	batch_q_vals = []
	all_states = []
	all_actions = []

	# this consists of one batch
	for episode in range(NUM_EPISODES):
		env.reset()
		episode_rewards = []
		states, actions, rewards, dones, new_states, logits, log_probs = [], [],[], [], [], [], []
		try:
			while not env.done:
				# while loop runs one episode
				old_reward = 0
				# states.append(env.state)
				state = torch.Tensor(env.state).view(-1, 5*5)
				logit = net(state) 	# somehow outputs a list of list
				log_prob = F.log_softmax(logit, dim=1) # for gradient
				logits.append(logit)
				probs = F.softmax(logit, dim=1)
				probs = probs.detach().numpy() # somehow outputs a list of list with one element
				action = np.random.choice(len(probs[0]), p=probs[0])
				# actions.append(action) # (dont since only relevant logprobs re appended)needed for selecting the relevant log_prob

				log_prob = log_prob[0][action]
				batch_log_probs.append(log_prob) 
				env.step(convert_action(action))
				actions.append(action)
				# new_states.append(env.state)
				new_reward = env.reward
				action_reward = new_reward - old_reward # calculating reward for only that particular action
				old_reward = new_reward
				rewards.append(action_reward)
				# dones.append(env.done)
		except Exception as e:
			print(e)
		
		# all_actions.extend(actions) 
		discounted_rewards = calc_qvals(rewards)
		batch_q_vals.extend(discounted_rewards)
		episode_rewards.append(new_reward)
		mean_batch_reward.append(statistics.mean(episode_rewards))
	# end of batch loop
	mean_epoch_reward.append(statistics.mean(mean_batch_reward))
	reward_for_loop = statistics.mean(mean_epoch_reward)
	# print('mean_batch_reward', mean_batch_reward)
	# #-------------------------------------------for updating gradients--------------------
	# multiplying gradients by batch_q_vals
	gradients = []
	for ele in range(len(batch_log_probs)):
		g = batch_q_vals[ele]*batch_log_probs[ele]
		gradients.append(g)
	# for mean calculation and negative multiplication
	mean = torch.Tensor([0])
	for ele in gradients:
		mean += ele
	loss= -mean/len(gradients)
	loss.backward()
	optimizer.step()
	#----------------------------------------------------------------------------------------
	print('mean_epoch_reward', mean_epoch_reward)
	counter +=1
print("mean award of 50 reached in %2d"%(counter))


