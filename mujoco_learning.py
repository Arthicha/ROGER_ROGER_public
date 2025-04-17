# ROGER: Reward-Oriented Gains via Embodied Regulation
# Arthicha Srisuchinnawong, RSS 2025
# minimalistic example code: locomotion learning with MuJoCo Locomotion Tasks
# update 17 April 2025
# (PPO code modified from https://github.com/seolhokim/Mujoco-Pytorch)

# ------------------- import modules ---------------------

# standard modules
import os, sys, time
from copy import deepcopy

# math module
import torch
import numpy as np

# parser
from configparser import ConfigParser
from argparse import ArgumentParser

# simulation
import gym
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)  # Create a window to init GLFW.

# visualization
import cv2

# custom module
from optimizer.rogerppo import PPO
from utils.utils import make_transition, Dict, RunningMeanStd

# ------------------- get arguments ---------------------
parser = ArgumentParser('parameters')
parser.add_argument('--path', type=str, default='utils', help="save path")
parser.add_argument("--env_name", type=str, default = 'Hopper-v3', help = "'Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','HumanoidStandup-v2',\
		  'InvertedDoublePendulum-v2', 'InvertedPendulum-v2' (default : Hopper-v2)")
parser.add_argument("--algo", type=str, default = 'ppo', help = 'algorithm to adjust (default : ppo)')
parser.add_argument('--train', type=bool, default=True, help="(default: True)")
parser.add_argument('--render', type=bool, default=True, help="(default: False)")
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs, (default: 1000)')
parser.add_argument("--use_cuda", type=bool, default = True, help = 'cuda usage(default : True)')
parser.add_argument("--reward_scaling", type=float, default = 0.1, help = 'reward scaling(default : 0.1)')

args = parser.parse_args()
parser = ConfigParser()
parser.read('config.ini')
agent_args = Dict(parser,args.algo)

# ------------------- config & initialization ---------------------
device = 'cuda' if (torch.cuda.is_available() and args.use_cuda) else 'cpu'

# initialize simulation	
env = gym.make(args.env_name,ctrl_cost_weight=0.0,render_mode='rgb_array') 

# get state/action dimensions
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]

# initialize state normalizer (running mean & std)
state_rms = RunningMeanStd(state_dim)

# initialize PPO
agent = PPO(device, state_dim, action_dim, agent_args)
if (torch.cuda.is_available()) and (args.use_cuda):
	agent = agent.cuda()

for n in range(args.epochs):

	# ------------------- epoch-wise setup ---------------------

	# initalize data list
	score = np.zeros((3,))
	score_list = []
	distance_list = []
	state_list = []
	firstrobot = False
	survive_time = 0

	# reset & get first state & normalize
	state_ = env.reset()
	state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)

	# ------------------- run 1 trajectory ---------------------
	for t in range(agent_args.traj_length):
		
		# render & visualize
		if (not firstrobot) and (args.render):  
			cv2.imshow('mujoco',env.render()[0][:,:,[2,1,0]])
			cv2.waitKey(1)


		# ----------------------  take action ----------------------

		# generate action & exploration
		mu,sigma = agent.get_action(torch.from_numpy(state).float().to(device))
		dist = torch.distributions.Normal(mu,torch.clamp(sigma[0],1e-4,None))
		action = dist.sample()

		# compute log_prob (for network update)
		log_prob = dist.log_prob(action).sum(-1,keepdim = True)

		# update environment/simulation
		next_state_, reward_, done, info = env.step(action.cpu().numpy())
	

		# ----------------------  compute multi-channel reward ----------------------
		
		primary_reward = reward_*args.reward_scaling
		action_penalty = -np.mean(np.power(np.clip(action.cpu().numpy(),-1,1),2))
		pitch_penalty = -np.abs(next_state_[1])
		reward = np.array([primary_reward,action_penalty,pitch_penalty])


		# ----------------------  add data to replay buffer  ----------------------

		next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
		state_list.append(state_)
		transition = make_transition(state,action.cpu().numpy(),reward,next_state,np.array([done]),log_prob.detach().cpu().numpy())
		agent.put_data(transition) 

		score += reward
		survive_time += 1

		if done: # reset needed

			# set & get first state & normalize
			state_ = env.reset()
			state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)

			# add data to score and distance list (for info visualization)
			score[1:] /= survive_time
			score_list.append(deepcopy(score))
			distance_list.append(info['x_position'])

			# initalize data list
			score = np.zeros((3,))
			firstrobot = True
			survive_time = 0

		else: # continue

			# update state
			state = next_state
			state_ = next_state_

	# train network
	agent.train_net(n)

	# update state normalization
	state_rms.update(np.vstack(state_list))

	# append score and distance only if never have appended it (not done)
	if not done:
		score[1:] /= survive_time
		score_list.append(deepcopy(score))
		distance_list.append(info['x_position'])

	# print epoch info
	print('----------------- epoch : ' +str(n)+' --------------------------')
	print("rewards :", np.mean(score_list,0))
	print("distance :", (np.sum(distance_list,0)/len(distance_list)))

		
