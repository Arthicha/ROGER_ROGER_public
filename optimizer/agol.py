# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy
import configparser

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array
from torch.distributions import Normal
import torch.nn as nn

# modular network
from optim import Optim

import matplotlib.pyplot as plt

# ------------------- configuration variables ---------------------
EPS = 1e-6
ENDINGCLIP = 5 # trow away n-last timestep

# ------------------- class AddedGradientOnlineLearning ---------------------
class AddedGradientOnlineLearning(Optim):
	# -------------------- constructor -----------------------
	# (private)

	def setup(self,config):

		# initialize replay buffer
		self.__sigma = float(config["ACTOROPTIM"]["SIGMA"])
		self.__sigmas = self.zeros(self.W.shape[0],self.W.shape[1]) + self.__sigma

		self.__min_grad = float(config["ACTOROPTIM"]["MINGRAD"])
		self.__lr = float(config["ACTOROPTIM"]["LR"])
		# reset everything before use
		self.reset()

	def attach_returnfunction(self,func):
		self.compute_return = func

	

	# ------------------------- update and learning ----------------------------
	# (public)

	def weighted_average(self,x,w,enable,dim=0,eps=0):
		return torch.sum(enable*x*w,dim,keepdim=True)/(eps+torch.sum(enable*w,dim,keepdim=True))

	
	def roger(self, rewards, predicted_rewards):

		safethreshold = torch.FloatTensor(np.array([0.2,0.2])).to(self.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # 2,2
		
		ch_mu = predicted_rewards[:,:,:,1:]

		ch_sd = torch.mean((rewards[:,:,:,1:]-ch_mu).pow(2),dim=(0),keepdim=True).sqrt()
		ch_min = torch.clamp(ch_mu - 3*ch_sd,None,0.0)
		
		reweight = rewards.clone()*0
		reweight[:,:,:,1:] = torch.pow(ch_min.unsqueeze(0)/safethreshold,2) # closeness

		rv = reweight[:,:,:,1:].sum(-1,keepdim=True)
		reweight[:,:,:,1:] = reweight[:,:,:,1:]*torch.clamp(rv,0,1)/(1e-6+rv)
		reweight[:,:,:,[0]] = 1-torch.clamp(rv,0,1)

		return reweight.detach()

	
	def update(self, advantages, exp_weight_replay, weights,grad_replay,lrscale=1,
		nepi=0,verbose=False,horizon=0,weightadjustment=False):
		
		with torch.no_grad():

			# normalize advantage
			std = torch.sqrt(torch.mean(advantages.pow(2)))
			std_advantage = (advantages)/(std+EPS)
			std_advantage = torch.clamp(std_advantage,-3,3)

			
			# balance advantage
			corrected_advantage = std_advantage.clone()
			sumpos = corrected_advantage[(corrected_advantage) >= 0].abs().sum()
			sumneg = corrected_advantage[(corrected_advantage) < 0].abs().sum()
			corrected_advantage[corrected_advantage < 0] *= 0.1*torch.clamp(sumpos/sumneg,0,1)

			# compute parameter update
			exploration = (exp_weight_replay-weights)
			rels = grad_replay # or new state

			update = (rels*exploration)[:,horizon//2:corrected_advantage.shape[1]+horizon//2]* corrected_advantage
			dw = torch.mean(lrscale*self.__lr*update[:,:-ENDINGCLIP] ,dim=(0,1))
			
			dwnorm = torch.norm(dw.flatten())
			if dwnorm >=  self.__min_grad:
				print('\tclip',dwnorm.item())
				dw = dw*(self.__min_grad/dwnorm).abs()

			# compute exploration update
			dsigma = std_advantage*(rels*((torch.pow(exploration,2)-torch.pow(self.__sigmas,2))/torch.pow(self.__sigmas,1)))[:,horizon//2:corrected_advantage.shape[1]+horizon//2]
			
			dsigma = self.__lr*1e-5*torch.sum(dsigma[:,:-ENDINGCLIP],dim=[0,1])
			dsigma = torch.clamp(dsigma,-0.001,0.001)

			# apply the update
			with torch.no_grad():
				param_update = dw.detach()
				sigma_update = (torch.clamp(self.__sigmas + dsigma,0.03,self.__sigma) - self.__sigmas)

		return param_update, sigma_update

	# -------------------- apply noise -----------------------
	def wnoise(self):
		self.dist = Normal(loc=0,scale=self.__sigmas)
		noise = self.dist.rsample()
		return noise

	# -------------------- set -----------------------
	def set_sigma(self,newsigma):
		with torch.no_grad():
			self.__sigmas *= 0
			self.__sigmas += newsigma[:self.__sigmas.shape[0]]

	# -------------------- get -----------------------
	def get_sigma(self):
		with torch.no_grad():
			sigmas = self.__sigmas.detach()
		return sigmas


	def add_sigma(self,dsig,gain=1):
		with torch.no_grad():
			self.__sigmas += gain*dsig.detach()


	
		




