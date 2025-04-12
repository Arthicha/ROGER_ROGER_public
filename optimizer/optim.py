# ------------------- import modules ---------------------

# standard modules
import time, sys, os
import warnings
from copy import deepcopy
import configparser

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array

#plot
import matplotlib.pyplot as plt

# ------------------- configuration variables ---------------------
warnings.simplefilter("ignore")
# ------------------- class Optim ---------------------

class Optim:


	# -------------------- constructor -----------------------
	# (private)

	def __init__(self, weight, configfile):

		# device
		if torch.cuda.is_available():
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')

		# initialize network to be optimize
		self.W = weight
		config = configparser.ConfigParser()
		config.read(configfile)

		# reset everything before use
		self.setup(config)
		self.freeze = True

	
	def setup(self,config):
		pass


	# -------------------- initialize matrix -----------------------
	# (private)

	def zeros(self,column,row,grad=False):
		if grad:
			return torch.nn.Parameter(torch.zeros((column,row)).to(self.device),requires_grad=True)
		else:
			return torch.zeros((column,row)).to(self.device)

	def identity(self,size,grad=False):
		if grad:
			return torch.nn.Parameter(torch.eye(size).to(self.device),requires_grad=True)
		else:
			return torch.eye(size).to(self.device)
			
	
	# -------------------- conversion -----------------------
	# (private)

	def torch(self,x):
		return x if torch.is_tensor(x) else  torch.FloatTensor(x).to(self.device)

	def numpy(self,x):
		return x.detach().cpu().numpy() if torch.is_tensor(x) else x


	# -------------------- handle functions -----------------------
	# (public)

	def reset(self):
		pass

	



