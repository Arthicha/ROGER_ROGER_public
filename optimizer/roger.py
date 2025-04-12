# ------------------- import modules ---------------------

# standard modules
from copy import deepcopy

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array

# audio module
import simpleaudio as sa

# ------------------- configuration variables ---------------------
EPS = 1e-6

# ------------------- class Reward-0riented Gains via Embodied Regulation  ---------------------
class ROGER():

	# -------------------- constructor -----------------------
	def __init__(self,threshold, ksigma = 3, scream=True):
		
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.constraint_threshold = torch.FloatTensor(np.array(threshold)).to(self.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # 2,2
		self.ksigma = ksigma

		# roger roger
		self.screamflag = scream
		self.wave_obj = sa.WaveObject.from_wave_file("optimizer/roger.wav")


	
	def compute_gains(self, rewards, predicted_rewards):
		# rewards & predicted rewards -> (#batch, #timestep, 1, #channel)
		
		with torch.no_grad():
			# compute constraint estimate
			mu = predicted_rewards[:,:,:,1:]
			sd = torch.mean((rewards[:,:,:,1:]-mu).pow(2),dim=(0),keepdim=True).sqrt()
			const_estimate = torch.clamp(mu - self.ksigma*sd,None,0.0)
			
			# compute proximities
			proximities = rewards.clone()*0
			proximities[:,:,:,1:] = torch.pow(const_estimate.unsqueeze(0)/self.constraint_threshold,2) # proximities

			# compute auxiliary variables
			sumproximities = proximities[:,:,:,1:].sum(-1,keepdim=True) 
			ratios = proximities[:,:,:,1:]/(EPS+sumproximities)
			deltas = torch.clamp(sumproximities,0,1)

			# compute lambda
			lambdas = rewards.clone()*0
			lambdas[:,:,:,1:] = ratios*deltas
			lambdas[:,:,:,[0]] = 1-deltas

			# roger roger
			if self.screamflag:
				self.scream()

			return lambdas


	def scream(self):
		play_obj = self.wave_obj.play()
		play_obj.wait_done()
		print("B1 Battle Droid: ROGER ROGER!")

	

