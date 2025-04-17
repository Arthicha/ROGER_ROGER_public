# ROGER: Reward-Oriented Gains via Embodied Regulation
# Arthicha Srisuchinnawong, RSS 2025
# minimalistic example code: locomotion learning with MuJoCo Locomotion Tasks
# update 17 April 2025
# (PPO code modified from https://github.com/seolhokim/Mujoco-Pytorch)

# ------------------- import modules ---------------------

# standard modules
import os, sys
from copy import deepcopy

# math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# custom modules
from network.network import Actor, Critic
from optimizer.roger import ROGER # ROGER
from utils.utils import ReplayBuffer, make_mini_batch, convert_to_tensor



# ------------------- Proximal Policy Optimization --------------------- 
class PPO(nn.Module):
	def __init__(self, device, state_dim, action_dim, args,constrain_ths=[1.0,0.2]):
		super(PPO,self).__init__()
		self.args = args
		self.nreward = len(constrain_ths) + 1
		self.device = device
		
		# replay buffer
		self.data = ReplayBuffer(action_prob_exist = True, max_size = self.args.traj_length, state_dim = state_dim, num_action = action_dim, reward_dim=self.nreward)
		
		# actor - critic architecture
		self.actor = Actor(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim, \
						   self.args.activation_function,None,self.args.trainable_std)
		self.critic = Critic(self.args.layer_num, state_dim, self.nreward, \
							 self.args.hidden_dim, self.args.activation_function,None)
		
		# optimizer
		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.args.critic_lr)

		# ROGER
		self.roger = ROGER(threshold=constrain_ths, ksigma = 3)
	
	# get gaussian policy
	def get_action(self,x): 
		mu,sigma = self.actor(x)
		return mu,sigma
	
	# get predicted state value
	def v(self,x): 
		return self.critic(x)
	
	# add data to replay buffer
	def put_data(self,transition): 
		self.data.put_data(transition)
		
	# compute generalized advantage estimate
	def get_gae(self, states, rewards, next_states, dones): 
		values = self.v(states).detach()
		td_target = rewards + self.args.gamma * self.v(next_states) * (1 - dones)
		delta = (td_target - values).detach().cpu().numpy()

		advantage_lst = []
		advantage = 0.0
		for idx in reversed(range(len(delta))):
			if dones[idx] == 1:
				advantage = 0.0
			advantage = self.args.gamma * self.args.lambda_ * advantage + delta[idx]
			advantage_lst.append(advantage)

		advantage_lst.reverse()
		advantages = torch.tensor(np.array(advantage_lst), dtype=torch.float).to(self.device)

		return values, advantages
	
	# compute "average" discounted return
	def get_avgdiscountedreturn(self,rewards,dones):

		returns_lst = []
		returns = 0.0
		gamma_sum = 0.0
		stacked_gamma = 1.0

		for idx in reversed(range(len(rewards))):
			if dones[idx] == 1:
				returns = 0.0
				gamma_sum = 0.0
				stacked_gamma = 1.0

			returns = self.args.gamma *returns + rewards[idx]
			gamma_sum += stacked_gamma
			stacked_gamma *= self.args.gamma
			returns_lst.append(returns/(1e-6+gamma_sum)) #divided by gamma_sum to average them all
		
		returns_lst.reverse()
		discountedreturns = torch.stack(returns_lst)#.to(self.device)
		return discountedreturns.unsqueeze(0).unsqueeze(2)


	def train_net(self,n_epi):
		# get data from replay buffer
		data = self.data.sample(shuffle = False) 
		states, actions, rewards, next_states, dones, old_log_probs = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'], data['log_prob'])
		
		# call roger % get reward weighting gains
		old_values, advantages = self.get_gae(states, rewards, next_states, dones)
		avgreturns = self.get_avgdiscountedreturn(rewards,dones)
		channelgains = self.roger.compute_gains(rewards=avgreturns,predicted_rewards=avgreturns,reducedims=[0,1,2])[0,:,0]
		
		# weighted sum in advantage level
		advantages = (advantages * channelgains).sum(-1,keepdim=True)
		sumold_values = (old_values * channelgains).sum(-1,keepdim=True)
		returns = advantages + sumold_values

		# normalize advantage
		advantages = (advantages - advantages.mean())/(advantages.std()+1e-6)
		
		for i in range(self.args.train_epoch):

			# loop through minibatch
			for state,action,old_log_prob,advantage,return_,old_value \
			in make_mini_batch(self.args.batch_size, states, actions, \
										   old_log_probs,advantages,returns,old_values): 

				
				# compute policy loss + entropy loss
				curr_mu,curr_sigma = self.get_action(state)
				curr_dist = torch.distributions.Normal(curr_mu,torch.clamp(curr_sigma,1e-4,None))
				entropy = curr_dist.entropy() * self.args.entropy_coef
				curr_log_prob = curr_dist.log_prob(action).sum(1,keepdim = True)

				# compute PPO surrogate loss
				ratio = torch.exp(curr_log_prob)/(1e-6+torch.exp(old_log_prob.detach()))
				surr1 = ratio * advantage
				surr2 = torch.clamp(ratio, 1-self.args.max_clip, 1+self.args.max_clip) * advantage
				actor_loss = (-torch.min(surr1, surr2) - entropy).mean() 
				
				# compute value loss using value clipping (PPO2 technique)
				value = self.v(state).float()
				value_loss = (value - return_.detach().float()).pow(2)
				old_value_clipped = old_value + (value - old_value).clamp(-self.args.max_clip,self.args.max_clip)
				value_loss_clipped = (old_value_clipped - return_.detach().float()).pow(2)
				critic_loss = 0.5 * self.args.critic_coef * torch.max(value_loss,value_loss_clipped).mean()
				
				# update actor
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
				self.actor_optimizer.step()
				
				# update critic
				self.critic_optimizer.zero_grad()
				critic_loss.backward()
				nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
				self.critic_optimizer.step()
				