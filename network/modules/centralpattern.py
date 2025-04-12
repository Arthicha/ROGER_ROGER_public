
# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array

# modular network
from modules.torchNet import torchNet


# ------------------- configuration variables ---------------------
EPSILON = 1e-6
GAMMA = 0.05
TAU_CONNECTION = 0.1

# ------------------- class SequentialCentralPatternGenerator ---------------------

class SequentialCentralPatternGenerator(torchNet):

	# -------------------- constructor -----------------------
	# (private)

	def __init__(self,hyperparams):

		super().__init__()

		# update hyperparameter
		self.__n_state = hyperparams.n_state
		self.__n_in = hyperparams.n_in


		# update weight-related hyperparameter
		self.__w_s1_s1 = hyperparams.w_recurrent
		self.__w_s1_s2 = hyperparams.w_forward
		self.__w_s2_s1 = hyperparams.w_backward
		self.__bias = hyperparams.b_state

		# initialize neuron activity
		self.__state = self.zeros(1,self.__n_state) 
		self.__inputs = self.zeros(1,self.__n_state)
		self.__basis = self.zeros(1,self.__n_state)

		# initialize connection weight
		self.__A = self.identity(self.__n_state)   *  self.__w_s1_s1
		self.__B = self.zeros(self.__n_state,self.__n_state)

		# initialize state connection proability
		self.__connection = hyperparams.connection
		self.forcing = 0
		self.slowforcing = 0
		# reset everything before use
		self.reset()

	# -------------------- set activity -----------------------
	# (public)

	# -------------------- get activity -----------------------
	# (public)

	# get state
	def get_outputs(self,torch=True): 
		return self.torch(self.__state) if torch else self.numpy(self.__state)

	# get state index
	def get_state_index(self):
		return self.numpy(torch.argmax(self.__state[0]).flatten())


	def get_connection(self):
		return self.__B.detach()


	# -------------------- update/handle functions -----------------------
	# (private)

	def __reset_connection(self):


		for i_from in range(self.__n_state):
			
			is_to = torch.where(self.__connection[i_from] > GAMMA)
			
			for i_to in is_to:

				# state to state
				self.__A[i_from,i_to] = self.__w_s1_s2
				self.__A[i_to,i_from] = self.__w_s2_s1

				# basis to state
				self.__B[i_from,i_to] = self.__w_s1_s2


	# -------------------- update/handle functions -----------------------
	# (public)

	def reset(self):

		# reset state
		self.__state *= 0.0
		self.__state[0,0] = 1.0

		# reset connection
		self.__reset_connection()

	def forward(self,observation,basis,delta=0,scaling=1,jacz=None):
		self.__inputs = observation
		self.__basis = basis
		self.__state[0,0] = 1 if torch.all(self.__state < GAMMA) else self.__state[0,0]

		self.forcing = delta #self.__A@delta
		
		#if jacz is None: # approx
		#	self.forcing = self.forcing[:,1]
		#else:
		#	self.forcing = torch.sum(self.forcing*self.torch(jacz).unsqueeze(0),-1)

		#self.forcing = self.forcing.unsqueeze(0)*scaling
		#print(self.forcing)
		#self.slowforcing = torch.clamp(self.slowforcing*0.999 + self.forcing*0.001,-0.1,0.1)
		#self.forcing = torch.clamp(self.forcing,-0.1,0.1)
		#print(self.forcing)
		self.__state = torch.sigmoid(
			(self.__state@self.__A)+
			(self.__basis@self.__B)+
			self.__bias + self.forcing)

		return self.__state








