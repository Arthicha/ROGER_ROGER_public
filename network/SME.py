
# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy
import configparser

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array
import torch.nn as nn

# modular network
from modules.utils import HyperParams
from modules.torchNet import torchNet 
from modules.centralpattern import SequentialCentralPatternGenerator
from modules.basis import BasisNetwork 
from modules.motor import MotorNetwork 

#plot
import matplotlib.pyplot as plt

# ------------------- configuration variables ---------------------

EPSILON = 1e-6 # a very small value

# ------------------- class SDN ---------------------

class SequentialMotionExecutor(torchNet):
	'''
	Sequential Motion Executor : Actor Network
	Parameters:
		connection/transition matrix from 'connection' 
		hyperparameter from a .init file at 'configfile'
	'''

	# ---------------------- constructor ------------------------ 
	def __init__(self,configfile, connection):

		super().__init__()

		# initialize hyperparameter
		config = configparser.ConfigParser()
		config.read(configfile)

		self.__n_state = int(config['HYPERPARAM']['NSTATE'])
		self.__n_in = int(config['HYPERPARAM']['NIN'])
		self.__n_out = int(config['HYPERPARAM']['NOUT'])
		self.__t_init = int(config['HYPERPARAM']['TINIT'])

		self.__connection = connection

		self.__hyperparams = HyperParams(self.__n_state,self.__n_in,self.__n_out)
		self.__hyperparams.w_time = float(config['C']['W_TIME'])
		self.__hyperparams.connection = self.__connection

		# ---------------------- initialize modular neural network ------------------------ 
		# (update in this order)

		self.zpg = SequentialCentralPatternGenerator(self.__hyperparams)
		self.bfn = BasisNetwork(self.__hyperparams) 

		motor_params = deepcopy(self.__hyperparams)
		motor_params.n_state = self.__n_state
		self.mn = MotorNetwork(motor_params,
			outputgain=[float(gain) for gain in list((config['MN']['GAIN']).split(","))],
			activation='tanh')

		# ---------------------- initialize neuron activity ------------------------ 
		self.__state = self.zeros(1,self.__n_state)
		self.__basis = self.zeros(1,self.__n_state)
		self.__filtered_inputs = self.zeros(1,self.__n_in)
		self.outputs = self.zeros(1,self.__n_out)

		# ---------------------- reset modular neural network ------------------------ 
		self.reset()

	# ---------------------- debugging   ------------------------ 

	def get_state(self,torch=False):
		if torch:
			return self.__state
		else:
			return self.__state.detach().cpu().numpy()[0]

	def get_basis(self,torch=False):
		if torch:
			return self.__basis
		else:
			return self.__basis.detach().cpu().numpy()[0]

	def get_output(self,torch=False):
		if torch:
			return self.outputs
		else:
			return self.outputs.detach().cpu().numpy()[0,:3]

	def explore(self,wnoise):
		self.mn.Wn = self.mn.W + wnoise

	def set_weight(self,newweight):
		with torch.no_grad():
			self.mn.W *= 0
			self.mn.W += newweight[:self.mn.W.shape[0]]

	def zero_grad(self):
		with torch.no_grad():
			self.mn.W.grad = None
			self.mn.Wn.grad = None


	# ---------------------- update   ------------------------

	def reset(self):
		self.zpg.reset()
		self.bfn.reset()
		self.mn.reset()

		for i in range(self.__t_init):
			self.halfforward()

	def halfforward(self):
		self.__state = self.zpg(self.__filtered_inputs,self.__basis)
		self.__basis = self.bfn(self.__state)

	def forward(self):
		self.__state = self.zpg(self.__filtered_inputs,self.__basis)
		self.__basis = self.bfn(self.__state)
		self.outputs = self.mn(self.__basis)

		return self.outputs

	




