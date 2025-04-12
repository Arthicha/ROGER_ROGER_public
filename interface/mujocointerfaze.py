# interface/__init__.py

'''
Class: MujocoInterfaze
created by: arthicha srisuchinnawong
e-mail: zumoarthicha@gmail.com
date: 24 June 2024

This class provide easy direct interface between python3 and Mujoco simulation
aiming mainly for reinforcement learning of a quadrupped robot (Unitree B2, similar to B1) 

NOTE THAT: this class use numpy array
'''

# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy

# mujoco interface module 
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

# math-related modules
import numpy as np # cpu array

# visualization modules
import cv2 



# ------------------- configuration variables ---------------------
EPS = 1e-6

# ------------------- class MujocoInterface ---------------------

class MujocoInterfaze(MujocoEnv, utils.EzPickle):

	_home = np.array([[-0.2, 0.608813, -1.21763],[0.2, 0.608813, -1.21763],[-0.2, 0.608813, -1.21763],[0.2, 0.608813, -1.21763]])
	__joint_handle = np.zeros((4,3)).astype(int) # joint handle (leg l, joint j)
	__target_positions = _home # joint target position (leg l, joint j)
	__previous_positions = _home
	__fc_handle = np.zeros(4).astype(int) # joint handle (leg l, joint j)
	
	metadata = {"render_modes": ["human","rgb_array","depth_array",],"render_fps": 1000,}
	directions = np.array([[-1,1,1],[1,1,1],[-1,1,1],[1,1,1]])

	# kinematics parameter
	upperlength = 0.35
	lowerlength = 0.35
	hiplength = 0.11973
	bodywidth = 0.144
	bodylength = 0.657
	bodyweight = 660

	# phase estimation variables
	hiperror_th = 0.02

	# state
	previous_pos = np.zeros((6))
	previous_vel = np.zeros((6))

	current_pos = np.zeros((6))
	current_vel = np.zeros((6))
	current_acc = np.zeros((6))


	# ---------------------- constructor ------------------------ 
	def __init__(self, hz = 30, dt=0.001, print_step = None, **kwargs):
		utils.EzPickle.__init__(self, **kwargs)
		
		observation_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64) # change shape of observation to your observation space size
		
		MujocoEnv.__init__(self,os.path.abspath("./simulation/b2scene.xml"),0.5,observation_space=observation_space,**kwargs) # load your MJCF model with env and choose frames count between actions

		self.step_number = 0
		self.hz = hz
		self.model.opt.timestep = dt
		self.integrated_error = 0
		
		self.print_step = print_step
		self.reset()
		print("INFO: MujocoInterfaze is initialized successfully.")


	# ---------------------- actuation  ------------------------ 
	def set_robot_joint(self,target_pos):
		self.__previous_positions = self.__target_positions
		self.__target_positions = target_pos.reshape((4,3)) + self._home
	
	def joint_to_control(self,joint_pos):
		return np.expand_dims((self.directions).flatten(),0)*(joint_pos - np.expand_dims((self._home).flatten(),0))

	def set_zero(self):
		self.set_robot_joint(np.zeros((12)))

	# ---------------------- get simulation data  ------------------------
	def get_robot_pose(self):
		posarray = self.data.body("base_link").xpos
		quat = np.array(self.data.body("base_link").xquat)
		orienarray = self.quaternion_to_euler(quat[0],quat[1],quat[2],quat[3])
		pose = np.array([posarray[0],posarray[1],posarray[2],orienarray[0],orienarray[1],orienarray[2]])
		return pose

	def get_obs(self):
		quat = np.array(self.data.body("base_link").xquat)
		euler = self.quaternion_to_euler(quat[0],quat[1],quat[2],quat[3])
		obs = np.concatenate(((euler),
							  np.array(self.data.joint("FR_hip_joint").qpos),
							  np.array(self.data.joint("FR_hip_joint").qvel),
							  np.array(self.data.joint("FR_thigh_joint").qpos),
							  np.array(self.data.joint("FR_thigh_joint").qvel),
							  np.array(self.data.joint("FR_calf_joint").qpos),
							  np.array(self.data.joint("FR_calf_joint").qvel),
							  np.array(self.data.joint("FL_hip_joint").qpos),
							  np.array(self.data.joint("FL_hip_joint").qvel),
							  np.array(self.data.joint("FL_thigh_joint").qpos),
							  np.array(self.data.joint("FL_thigh_joint").qvel),
							  np.array(self.data.joint("FL_calf_joint").qpos),
							  np.array(self.data.joint("FL_calf_joint").qvel),
							  np.array(self.data.joint("RR_hip_joint").qpos),
							  np.array(self.data.joint("RR_hip_joint").qvel),
							  np.array(self.data.joint("RR_thigh_joint").qpos),
							  np.array(self.data.joint("RR_thigh_joint").qvel),
							  np.array(self.data.joint("RR_calf_joint").qpos),
							  np.array(self.data.joint("RR_calf_joint").qvel),
							  np.array(self.data.joint("RL_hip_joint").qpos),
							  np.array(self.data.joint("RL_hip_joint").qvel),
							  np.array(self.data.joint("RL_thigh_joint").qpos),
							  np.array(self.data.joint("RL_thigh_joint").qvel),
							  np.array(self.data.joint("RL_calf_joint").qpos),
							  np.array(self.data.joint("RL_calf_joint").qvel)), axis=0)
		return obs

	# ---------------------- simulation control  ------------------------

	def update(self,contact=np.array([1.0,1.0,1.0,1.0])):
		t_target = 1/self.hz
		nt = int(t_target/self.model.opt.timestep)
		
		target = np.zeros((nt,4,3))

		for l in range(4):
			for j in range(3):
				target[:,l,j] = np.interp(np.arange(0,1,1/nt),np.array([0,1]),[self.__previous_positions[l][j],self.__target_positions[l][j]])

		for i in range(nt):
			self.step(target[i],contact=contact)

		self.step_number += 1


		
	def step(self,action,contact=np.array([1.0,1.0,1.0,1.0])):
		# get observation
		obs = self.get_obs()
		roll_t = obs[0]
		pitch_t = obs[1]
		theta_t = np.reshape(obs[3::2],(4,3))
		dtheta_t = np.reshape(obs[4::2],(4,3))
		target_t = deepcopy(action)
		self.current_pos = self.get_robot_pose()
		self.current_vel = (self.current_pos - self.previous_pos)/self.model.opt.timestep
		self.current_acc = (self.current_vel - self.previous_vel)/self.model.opt.timestep
		self.previous_pos = self.get_robot_pose()
		self.previous_vel = deepcopy(self.current_vel)

		# ---------------  compute jacobian  ---------------
		J = self.jacobian(theta_t[:,0],theta_t[:,1], theta_t[:,2],roll_t, pitch_t)
		

		# ------------  leg phase estimation ----------------
		forcegain, contact = self.estimate_legphase(target_t,theta_t,roll_t,pitch_t,contact=contact)
		
		# ---------------  estimate static torque ---------------
		
		tau0_t = np.zeros((4,3))
		vx = self.batchmul(J,np.expand_dims(dtheta_t,1))[:,0,0]
		fx = 0.1*self.bodyweight*(np.clip(vx/1.0,-1,1))*contact*forcegain
		fy = np.zeros((4))
		fz = 1.0*self.bodyweight*np.ones((4))

		Jt = np.transpose(J,axes=(0, 2, 1))
		Fx = np.expand_dims(np.transpose(np.array([fx,fy,fz]),(1,0)),1)
		Fq = self.batchmul(Jt,Fx)[:,0,:]

		
		for j in range(3):
			#tau0_t[:,j] = -Fq[:,j] * np.clip(phase_t,0,1) * np.array(contact)
			tau0_t[:,j] = -Fq[:,j] * (contact * forcegain)


		# ---------------  update the simulation ---------------
		self.do_simulation(np.concatenate([target_t.flatten(),tau0_t.flatten()]),1)

		


	def reset(self):
		self.step_number = 0
		qpos = np.array([0,0,0.56,1,0,0,0,0.2, 0.608813, -1.21763,-0.2, 0.608813, -1.21763,0.2, 0.608813, -1.21763,-0.2, 0.608813, -1.21763])
		#qpos = np.array([0,0,0.56,1,0,0,0,0.2, 0.85, -1.56,-0.2, 0.85, -1.56,0.2, 0.85, -1.56,-0.2, 0.85, -1.56])
		
		self.set_state(qpos, self.init_qvel*0)

		euler = self.quaternion_to_euler(qpos[3],qpos[4],qpos[5],qpos[6])


		self.previous_pos = np.concatenate([qpos[:3],euler])
		self.previous_vel = np.zeros((6))

		for i in range(500):
			self.step(self._home,contact=np.array([1,1,1,1]))
			if 0:#i %10 == 0:
				cv2.imshow("image", self.render())
				cv2.waitKey(1)

		theta_t = np.reshape(self.get_obs()[3::2],(4,3))




	# ----------------------  auxillary function  ------------------------

	def jacobian(self, q1, q2, q3, roll, pitch):
		l1 = self.upperlength
		l2 = self.lowerlength
		a = self.hiplength*np.array([-1, 1, -1, 1])
		s = np.sin
		c = np.cos

		# Precompute common terms
		sigma1 = c(q1) * c(roll) - c(pitch) * s(q1) * s(roll)
		sigma2 = c(q1) * s(roll) + c(pitch) * c(roll) * s(q1)
		sigma6 = s(q2) * (s(q1) * s(roll) - c(pitch) * c(q1) * c(roll)) - c(q2) * c(roll) * s(pitch)
		sigma7 = s(q2) * (c(q1) * s(roll) + c(pitch) * c(roll) * s(q1)) + c(q2) * s(pitch) * s(roll)
		sigma8 = c(pitch) * c(q2) - c(q1) * s(pitch) * s(q2)
		sigma9 = s(q1) * s(roll) - c(pitch) * c(q1) * c(roll)
		sigma10 = c(roll) * s(q1) + c(pitch) * c(q1) * s(roll)

		sigma3 = l2 * (c(q3) * sigma6 + s(q3) * (c(q2) * sigma9 + c(roll) * s(pitch) * s(q2)))
		sigma4 = l2 * (c(q3) * sigma7 + s(q3) * (c(q2) * sigma10 - s(pitch) * s(q2) * s(roll)))
		sigma5 = l2 * (c(q3) * sigma8 - s(q3) * (c(pitch) * s(q2) + c(q1) * c(q2) * s(pitch)))

		# Compute the Jacobian tensor
		J11 = l2 * (c(q2) * c(q3) * s(pitch) * s(q1) - s(pitch) * s(q1) * s(q2) * s(q3)) + a * c(q1) * s(pitch) + l1 * c(q2) * s(pitch) * s(q1)
		J12 = -l1 * sigma8 - sigma5
		J13 = -sigma5

		J21 = l2 * (c(q2) * c(q3) * sigma1 - s(q2) * s(q3) * sigma1) - a * sigma10 + l1 * c(q2) * sigma1
		J22 = -sigma4 - l1 * sigma7
		J23 = -sigma4

		J31 = l2 * (c(q2) * c(q3) * sigma2 - s(q2) * s(q3) * sigma2) - a * sigma9 + l1 * c(q2) * sigma2
		J32 = -sigma3 - l1 * sigma6
		J33 = -sigma3

		# Assemble the Jacobian matrix
		jacobian = np.array([
			[J11, J12, J13],
			[J21, J22, J23],
			[J31, J32, J33]
		])

		J = np.transpose(jacobian,(2,0,1))
		return J

	def estimate_onlylegphase(self,target):
		contact = np.array([0,0,0,0])
		_, _, zs = self.calculate_xyz(target[:,0],target[:,1], target[:,2],0, 0)
		
		for i in range(0,2):
			if np.abs(zs[2*i] - zs[2*i+1]) > 0.002:
				if zs[2*i] < zs[2*i+1]:
					contact[2*i] = 1 
					contact[2*i+1] = 0
				else:
					contact[2*i+1] = 1 
					contact[2*i] = 0
			else:
				contact[2*i] = 1 
				contact[2*i+1] = 1

		return contact

	def estimate_legphase(self,target,theta,roll,pitch,contact=np.array([1,1,1,1])):



		# compute contect if not provided
		if contact is None:
			#error = target - theta
			#phase = np.clip((-self.directions[:,0]*(error[:,0])/self.hiperror_th),0,1)

			#contact = np.power(phase,1)

			contact = np.array([0,0,0,0])
			_, _, zs = self.calculate_xyz(target[:,0],target[:,1], target[:,2],roll, pitch)
			

			for i in range(0,2):
				if np.abs(zs[2*i] - zs[2*i+1]) > 0.002:
					if zs[2*i] < zs[2*i+1]:
						contact[2*i] = 1 
						contact[2*i+1] = 0
					else:
						contact[2*i+1] = 1 
						contact[2*i] = 0
				else:
					contact[2*i] = 1 
					contact[2*i+1] = 1



		xs, ys, _ = self.calculate_xyz(theta[:,0],theta[:,1], theta[:,2],roll, pitch)


		xsign = np.array([1,1,-1,-1])*contact
		ysign = np.array([-1,1,-1,1])*contact

		fx = np.sum(xsign*xs)/(EPS+xsign*xs)
		fy = np.sum(ysign*ys)/(EPS+ysign*ys)

		fx = np.clip(fx,0,None)
		fy = np.clip(fy,0,None)
		fc = (fx*fy*contact)
		fc /= np.sum(fc)+EPS

		return np.clip(fc,0,1), contact


	def quaternion_to_euler(self,w, x, y, z):
		# Roll (x-axis rotation)
		sinr_cosp = 2 * (w * x + y * z)
		cosr_cosp = 1 - 2 * (x * x + y * y)
		roll = np.arctan2(sinr_cosp, cosr_cosp)
		
		# Pitch (y-axis rotation)
		sinp = 2 * (w * y - z * x)
		pitch = np.arcsin(sinp) if np.abs(sinp) <= 1 else np.pi/2 * np.sign(sinp)
		
		# Yaw (z-axis rotation)
		siny_cosp = 2 * (w * z + x * y)
		cosy_cosp = 1 - 2 * (y * y + z * z)
		yaw = np.arctan2(siny_cosp, cosy_cosp)
		
		return roll, pitch, yaw

	def calculate_xyz(self, q1, q2, q3, roll, pitch):
		l1 = self.upperlength
		l2 = self.lowerlength
		a = self.hiplength*np.array([-1, 1, -1, 1])
		l = self.bodylength*np.array([1, 1, -1, -1])
		w = self.bodywidth*np.array([-1, 1, -1, 1])

		# Compute auxillary variables
		sigma_4 = np.cos(roll) * np.sin(q1) + np.cos(pitch) * np.cos(q1) * np.sin(roll)
		sigma_5 = np.sin(q1) * np.sin(roll) - np.cos(pitch) * np.cos(q1) * np.cos(roll)
		sigma_1 = np.cos(pitch) * np.sin(q2) + np.cos(q1) * np.cos(q2) * np.sin(pitch)
		sigma_2 = np.cos(q2) * sigma_4 - np.sin(pitch) * np.sin(q2) * np.sin(roll)
		sigma_3 = np.cos(q2) * sigma_5 + np.cos(roll) * np.sin(pitch) * np.sin(q2)
		
		# Compute x, y, and z
		x = (l * np.cos(pitch)) / 2 - l2 * (np.cos(q3) * sigma_1 + np.sin(q3) * (np.cos(pitch) * np.cos(q2) - np.cos(q1) * np.sin(pitch) * np.sin(q2))) - l1 * sigma_1 + a * np.sin(pitch) * np.sin(q1)
		y = a * (np.cos(q1) * np.cos(roll) - np.cos(pitch) * np.sin(q1) * np.sin(roll)) + (w * np.cos(roll)) / 2 + l2 * (np.cos(q3) * sigma_2 - np.sin(q3) * (np.sin(q2) * sigma_4 + np.cos(q2) * np.sin(pitch) * np.sin(roll))) + l1 * sigma_2 + (l * np.sin(pitch) * np.sin(roll)) / 2
		z = a * (np.cos(q1) * np.sin(roll) + np.cos(pitch) * np.cos(roll) * np.sin(q1)) + l2 * (np.cos(q3) * sigma_3 - np.sin(q3) * (np.sin(q2) * sigma_5 - np.cos(q2) * np.cos(roll) * np.sin(pitch))) + (w * np.sin(roll)) / 2 + l1 * sigma_3 - (l * np.cos(roll) * np.sin(pitch)) / 2
		
		return x, y, z
		

	def batchmul(self, a,b): # 3d tensor
		return np.transpose((a*b).sum(2,keepdims=True),axes=(0,2,1))





