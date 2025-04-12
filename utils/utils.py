
# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy

# math-related modules
import numpy as np # cpu array
import torch # pytorch

# ------------------- configuration variables ---------------------


# ------------------- class TorchReplay ---------------------
class TorchReplay:

	# -------------------- class variable -----------------------
	__data = [] # data replay
	__nmax = 10 # maximum length
	__first = True 

	def __init__(self,nmax=10,shape=(1,1),fillfirst = True):
		if torch.cuda.is_available():
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')

		self.__nmax = nmax
		self.__fillfirst = fillfirst
		self.__data = torch.zeros((nmax,)+shape).to(self.device)
		self.__subdata = torch.zeros(shape).to(self.device)
		self.__sid = 0

	def add(self,value,convert=False):
		value_ = torch.FloatTensor(value).to(self.device) if convert else value
		
		self.__subdata[self.__sid] = value_
		self.__sid += 1 

		if self.__sid >= self.__data.shape[1]:
			self.__sid = 0
			if (self.__first) and (self.__fillfirst):
				self.__data = self.__data*0 + self.__subdata.clone()
				self.__first = False
			else:
				self.__data[:-1] = self.__data[1:].clone()
				self.__data[-1] = self.__subdata.clone()

	def data(self):
		return self.__data

	def get_min(self):
		return torch.min(self.__data,dim=0).values

	def get_max(self):
		return torch.max(self.__data,dim=0).values

	def get_range(self):
		return self.get_max()-self.get_min()

	def get_previous(self):
		return self.__data[-1]

	def mean(self,last,const=None):
		x = self.__data[-last:]
		return torch.mean(x,dim=0).unsqueeze(0)

	def std(self,last,const=None):
		x = self.__data[-last:]
		return torch.std(x,dim=0).unsqueeze(0)




