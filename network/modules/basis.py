# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array
# modular network
from modules.torchNet import torchNet

# ------------------- class BasisNetwork ---------------------
EPS = 1e-6
GAMMA = 0.05

class BasisNetwork(torchNet):


	# -------------------- constructor -----------------------
	# (private)

	def __init__(self,hyperparams,nactiveneuron=4):

		super().__init__()

		# update hyperparameter
		self.__n_state = hyperparams.n_state

		# update weight-related hyperparameter
		self.__w_r1_r1 = 1.0-hyperparams.w_time
		self.__w_s1_r1 = hyperparams.w_time
		#print(hyperparams.w_time)

		# initialize neuron activity
		self.__state = self.zeros(1,self.__n_state)
		self.__basis = self.zeros(1,self.__n_state)

		# initialize connection weight
		self.__A = self.identity(self.__n_state) * self.__w_s1_r1
		self.__B = self.identity(self.__n_state) * self.__w_r1_r1

		self.__connection = hyperparams.connection

		# reset everything before use
		self.reset()

	# -------------------- get activity -----------------------
	# (public)

	# get state
	def get_outputs(self,torch=True): 
		return self.torch(self.__basis) if torch else self.numpy(self.__basis)

	# get state index
	def get_basis_index(self):
		return self.numpy(torch.argmax(self.__basis[0]).flatten())

	def get_connection(self):
		return self.__A.detach()

	# -------------------- update/handle functions -----------------------
	# (private)

	def __reset_connection(self):
		self.__A = self.identity(self.__n_state) * self.__w_s1_r1
		self.__B = self.identity(self.__n_state) * self.__w_r1_r1

		# normal connection
		for i_from in range(self.__n_state):
			i_tos = torch.where(self.__connection[i_from] > GAMMA)
			self.__A[i_tos,i_from] += -0.5*self.__w_s1_r1
			for i_to in i_tos:
				_, i_to2 = torch.where(self.__connection[i_to] > GAMMA)
				self.__A[i_to2,i_from] += -0.25*self.__w_s1_r1
				

	def reset(self):
		self.__basis *= 0.0
		self.__reset_connection()
		
	def forward(self,state):
		self.__state = state
		self.__basis = self.ReLU((self.__state@self.__A)   +   (self.__basis@self.__B))
		return self.__basis

	def ReLU(self,x):
		return torch.clamp(x,0,1)






