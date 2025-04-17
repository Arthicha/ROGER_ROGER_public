
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


class ReplayBuffer():
	def __init__(self, action_prob_exist, max_size, state_dim, num_action,reward_dim=1):
		self.max_size = max_size
		self.data_idx = 0
		self.action_prob_exist = action_prob_exist
		self.data = {}
		
		self.data['state'] = np.zeros((self.max_size, state_dim))
		self.data['action'] = np.zeros((self.max_size, num_action))
		self.data['reward'] = np.zeros((self.max_size, reward_dim))
		self.data['next_state'] = np.zeros((self.max_size, state_dim))
		self.data['done'] = np.zeros((self.max_size, 1))
		if self.action_prob_exist :
			self.data['log_prob'] = np.zeros((self.max_size, 1))

	def put_data(self, transition):
		idx = self.data_idx % self.max_size
		self.data['state'][idx] = transition['state']
		self.data['action'][idx] = transition['action']
		self.data['reward'][idx] = transition['reward']
		self.data['next_state'][idx] = transition['next_state']
		self.data['done'][idx] = float(transition['done'])
		if self.action_prob_exist :
			self.data['log_prob'][idx] = transition['log_prob']
		
		self.data_idx += 1

	def sample(self, shuffle, batch_size = None):
		if shuffle :
			sample_num = min(self.max_size, self.data_idx)
			rand_idx = np.random.choice(sample_num, batch_size,replace=False)
			sampled_data = {}
			sampled_data['state'] = self.data['state'][rand_idx]
			sampled_data['action'] = self.data['action'][rand_idx]
			sampled_data['reward'] = self.data['reward'][rand_idx]
			sampled_data['next_state'] = self.data['next_state'][rand_idx]
			sampled_data['done'] = self.data['done'][rand_idx]
			if self.action_prob_exist :
				sampled_data['log_prob'] = self.data['log_prob'][rand_idx]
			return sampled_data
		else:
			return self.data

	def size(self):
		return min(self.max_size, self.data_idx)

class RunningMeanStd(object):
	def __init__(self, epsilon=1e-4, shape=()):
		self.mean = np.zeros(shape, 'float64')
		self.var = np.ones(shape, 'float64')
		self.count = epsilon

	def update(self, x):
		batch_mean = np.mean(x, axis=0)
		batch_var = np.var(x, axis=0)
		batch_count = x.shape[0]
		self.update_from_moments(batch_mean, batch_var, batch_count)
		

	def update_from_moments(self, batch_mean, batch_var, batch_count):
		self.mean, self.var, self.count = update_mean_var_count_from_moments(
			self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

	def save(self, path):
		state = {
			'mean': self.mean,
			'var': self.var,
			'count': self.count
		}
		torch.save(state, path)
		print(f"RunningMeanStd state saved to {path}")

	# Load method for RunningMeanStd
	def load(self, path):
		state = torch.load(path)
		self.mean = state['mean']
		self.var = state['var']
		self.count = state['count']
		print(f"RunningMeanStd state loaded from {path}")


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
	delta = batch_mean - mean
	tot_count = count + batch_count

	new_mean = mean + delta * batch_count / tot_count
	m_a = var * count
	m_b = batch_var * batch_count
	M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
	new_var = M2 / tot_count
	new_count = tot_count

	return new_mean, new_var, new_count


class Dict(dict):
	def __init__(self,config,section_name,location = False):
		super(Dict,self).__init__()
		self.initialize(config, section_name,location)
	def initialize(self, config, section_name,location):

		for key,value in config.items(section_name):
			if location :
				self[key] = value
			else:
				self[key] = eval(value)
	def __getattr__(self,val):
		return self[val]
	
def make_transition(state,action,reward,next_state,done,log_prob=None,enable=1,activity1=None,activity2=None):
	transition = {}
	transition['state'] = state
	transition['action'] = action
	transition['reward'] = reward
	transition['next_state'] = next_state
	transition['log_prob'] = log_prob
	transition['done'] = done
	transition['enable'] = enable
	#transition['activity1'] = activity1
	#transition['activity2'] = activity2
	return transition

def make_mini_batch(*value,shuffle=True):
	mini_batch_size = value[0]
	full_batch_size = len(value[1])
	full_indices = np.arange(full_batch_size)
	if shuffle:
		np.random.shuffle(full_indices)
	for i in range(full_batch_size // mini_batch_size):
		indices = full_indices[mini_batch_size*i : mini_batch_size*(i+1)]
		yield [x[indices] for x in value[1:]]
		
def convert_to_tensor(*value):
	device = value[0]
	return [torch.tensor(x).float().to(device) for x in value[1:]]





