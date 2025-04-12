# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array

# ------------------- class HyperParams ---------------------
class HyperParams:


	def __init__(self,n_state,n_in,n_out,epsilon=0.01,one=0.95,gamma=0.5,infinity=8):
		
		# neuron/state number
		self.n_state = int(n_state)
		self.n_in = int(n_in)
		self.n_out = int(n_out)

		# calculation parameters
		self.epsilon = epsilon
		self.one = one
		self.gamma = gamma
		self.infinity = infinity

		self.connection = None

		# solve for w
		self.w_forward, self.w_recurrent, self.w_backward, _, self.b_state = self.solve(self.n_state-1,self.epsilon,self.one,self.gamma,self.infinity)

	def solve(self,n,e,i,g,inf):
		# solve for X in AX = B	

		# coefficient matrix
		A = np.array([[e, i, e, e, 1],
			[i, e, e, e, 1],
			[e, e, e, i, 1],
			[i, i, i, i, 1],
			[i, e, e, i, 1]])

		# answer matrix
		B = np.array([[inf],
			[-inf],
			[-inf],
			[-inf],
			[g]])

		# solution
		X = np.linalg.solve(A,B)

		return X.flatten()		






