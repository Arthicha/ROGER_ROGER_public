
# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array

# modular network
from modules.torchNet import torchNet

# ------------------- class MotorNetwork ---------------------

class MotorNetwork(torchNet):	

	# -------------------- constructor -----------------------
	# (private)

	def __init__(self,hyperparams,load=0,outputgain=None, activation='identity'):

		# initialize network hyperparameter
		super().__init__()

		# device
		if torch.cuda.is_available():
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')

		self.__n_state = hyperparams.n_state
		self.__n_out = hyperparams.n_out
		self.__activation = activation	

		# initialize connection weight
		self.W = self.zeros(self.__n_state,self.__n_out,grad=True)
		self.Wn = self.W + self.zeros(self.__n_state,self.__n_out)

		# normalize all joints -> output gain
		self.__output_gain = self.zeros(1,self.__n_out) + 1.0 if outputgain is None else self.torch(outputgain)
		self.__output_gain = torch.reshape(self.__output_gain,(1,self.__n_out))
		
		self.reset()

	# -------------------- set values -----------------------
	# (public)

	def apply_noise(self,noise):
		self.Wn = self.W + noise

	# -------------------- handle functions -----------------------
	# (public)

	def reset(self):
		# reset connection
		pass

	def forward(self,x):
		activity = (x@self.Wn)*self.__output_gain
		if self.__activation is 'tanh':
			outputs = torch.tanh(activity)
		else:
			outputs = activity

		return outputs




	
		




