# ROGER: Reward-Oriented Gains via Embodied Regulation
# Arthicha Srisuchinnawong, RSS 2025
# example code: locomotion learning of a quadruped robot
# update 12 April 2025

# ------------------- import modules ---------------------
# standard modules
import sys, os
from copy import deepcopy

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array
import torch.nn.functional as F

# simulation interface (rewrite this class for your own robot)
from interface.mujocointerfaze import MujocoInterfaze

# SME-AGOL (Arthicha Srisuchinnawong, TNNLS 2025)
from network.SME import SequentialMotionExecutor
from optimizer.agol import AddedGradientOnlineLearning
from utils.utils import TorchReplay as Replay

# ROGER (Arthicha Srisuchinnawong, RSS 2025)
from optimizer.roger import ROGER

# visualization
import cv2
import matplotlib.pyplot as plt

# ------------------- config variables ---------------------
NOUTPUT = 12 # number of actuator
NREPLAY = 8 # replay length (#episode)
NTIMESTEP = 80 # timestep per episode, (default: 70-80 approx, 3-4 gait cycles)
T0 = 40 # soft start timestep (default: 0.5*NTIMESTEP)
NREWARD = 3 # reward channels (reward, constraint1, constraint2, ...)
NEPISODE = 1000 # number of episode to run
HORIZON = 20 # horizon for computing return (default: 20 for SME's w_time = 0.18, approx. 1 gait cycle)
EPS = 1e-6

FRAMEWIDTH = 400
FRAMEHEIGHT = 400
RENDER = False
CONNECTION = torch.FloatTensor(np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]])).cuda()

# ------------------- auxiliary functions ---------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def numpy(x):
	return x.cpu().detach().numpy()

def tensor(x):
	return torch.FloatTensor(x).to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

def compute_return(speed,reduce=torch.mean):
	global HORIZON
	returns_ = speed.clone()
	returns_ = returns_.unfold(1,HORIZON,1)
	returns_ = reduce(returns_,dim=-1)
	return returns_

# ------------------- setup ---------------------

# initiliaze SME network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sme = SequentialMotionExecutor('network.ini',CONNECTION)
roger = ROGER([0.2,0.2],ksigma=3)

# initialize AGOL learning algorithm
agol = AddedGradientOnlineLearning(sme.mn.W,'optimizer.ini')
agol.attach_returnfunction(compute_return) # set return function


# initialize simulation interface
robot = MujocoInterfaze(hz = 30, dt=0.001,print_step=None, render_mode="rgb_array",width=FRAMEWIDTH, height=FRAMEHEIGHT,camera_name='side')


# initialize experience replay
reward_replay = Replay(NREPLAY,shape=(NTIMESTEP,1,NREWARD))
pose_replay = Replay(NREPLAY,shape=(NTIMESTEP,1,6))
grad_replay = Replay(NREPLAY,shape= (NTIMESTEP,CONNECTION.shape[0],NOUTPUT))
weight_replay = Replay(NREPLAY,shape=(1,CONNECTION.shape[0],NOUTPUT))
bases_replay = Replay(NREPLAY,shape=(NTIMESTEP,1,CONNECTION.shape[0]))


# ------------------- start locomotion learning ---------------------
for i in range(NEPISODE):
	print('episode',i)

	# episode-wise setup
	robot.reset()
	prepose = robot.get_robot_pose()

	# apply exploration
	wnoises = agol.wnoise()
	wnoises[:,::3] *= 0.0 #(disable hip joint, optional)
	sme.explore(wnoises)
	weight_replay.add(sme.mn.Wn,convert=False)

	for t in range(NTIMESTEP+T0):

		# compute output
		output = sme.forward()

		# update environment
		gain = 1 if t >= T0 else t/T0
		action = gain*np.clip(np.array((output).detach().cpu()),-1.5,1.5)
		robot.set_robot_joint(action)
		robot.update(contact=None)

		# render
		if RENDER:
			cv2.imshow("image", robot.render())
			cv2.waitKey(1)


		# compute reward
		pose = robot.get_robot_pose()
		dx = pose[0]-prepose[0]
		dy = pose[1]-prepose[1]
		reward = np.sqrt(dx*dx+dy*dy)*np.cos(prepose[-1] - np.arctan2(dy,dx)) # forward speed
		rollpen = -np.abs(pose[3])
		pitchpen = -np.abs(pose[4])
		prepose = deepcopy(pose)


		# append experience replay
		if (t >= T0):
			sme.zero_grad()
			torch.sum(output).backward() 
			grad_replay.add(sme.mn.W.grad.abs(),convert=False)
			pose_replay.add(pose,convert=True)
			bases_replay.add(sme.get_basis(torch=True).detach(),convert=False)
			reward_replay.add([[reward,rollpen,pitchpen]],convert=True)

	
	# print episode info
	print('\tepisodic reward',torch.sum(reward_replay.data()[-1,:,:,0]).item())
	print('\tmax penalty',-torch.min(reward_replay.data()[-1,:,:,1]).item(),-torch.min(reward_replay.data()[-1,:,:,2]).item())
	
	# -------------------------------------------------------------------------------------------------

	# get data from replay
	sid = -(i + 1) if i < ((NREPLAY)-1) else 0
	rewards = reward_replay.data()[sid:].detach()
	states = bases_replay.data()[sid:].detach()
	explorations = weight_replay.data()[sid:].detach()
	relevances = grad_replay.data()[sid:].detach()

	# compute advantage
	baseline = rewards.mean(0,keepdim=True)
	delta = rewards - baseline

	# normalize reward/advantage (optional, otherwise skip)
	delta[:,:,:,0] = (delta[:,:,:,0] - delta[:,:,:,0].mean())/(EPS + delta[:,:,:,0].std()) 
	delta[:,:,:,1:] = (delta[:,:,:,1:])/(EPS + delta[:,:,:,1:].pow(2).mean([0,1,2],keepdim=True).sqrt()) 
	delta = torch.clamp(delta,-3,3)

	# compute ROGER 
	gains = roger.compute_gains(rewards, baseline)
	advantages = compute_return(torch.sum(gains*delta,-1,keepdim=True))

	# update SME
	dws, dsigma = agol.update(advantages, explorations ,sme.mn.W, relevances,lrscale=1.0,nepi=i,verbose=True)
	with torch.no_grad():
		sme.mn.W += dws
	agol.add_sigma(dsigma)


