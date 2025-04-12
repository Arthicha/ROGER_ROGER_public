# ROGER: Reward-Oriented Gains via Embodied Regulation

<p align="center">
    <img width="75%" src="/pictures/framework.jpg" />
</p>

Existing robot locomotion learning techniques rely heavily on the offline selection of proper reward weighting gains and cannot guarantee constraint satisfaction (i.e., constraint violation) during training. Thus, this work aims to address both issues by proposing Reward-Oriented Gains via Embodied Interaction (ROGER), adapting reward-weighting gains online based on penalties received throughout the embodied interaction process. The ratio between the positive reward (primary reward) and negative reward (penalty) gains is automatically reduced as the learning approaches the constraint thresholds to avoid violation. Conversely, the ratio is increased when the learning is in safe states to prioritize performance. With a 60-kg quadruped robot, ROGER achieved near-zero constraint violation throughout multiple learning trials. It also achieved up to 50% more primary reward than the equivalent state-of-the-art techniques. In MuJoCo continuous locomotion benchmarks, including a single-leg hopper, ROGER exhibited comparable or up to 100% higher performance and 60% less torque usage and orientation deviation, compared to those trained with the default reward function. Finally, real-world locomotion learning of a physical quadruped robot was achieved from scratch within one hour without any falling. Therefore, this work contributes to constraint-satisfying real-world continual robot locomotion learning and the simplification of reward weighting gain tuning, potentially facilitating the development of physical robots and those that learn in the real world. The video is available at [https://youtu.be/F1olq7W6J9g?si=0YCqNpM5_RHw9Hv_](https://youtu.be/F1olq7W6J9g?si=0YCqNpM5_RHw9Hv_).



# Contents
- [Requirements](#Requirements)
- [Running](#Running)

# Requirements

* simulation software
	- Mujoco [https://mujoco.org/](https://mujoco.org/)
	- OpenAI Gym [https://www.gymlibrary.dev/environments/mujoco/index.html](https://www.gymlibrary.dev/environments/mujoco/index.html)

* python 3.6.5
	- numpy 1.22.4
	- pytorch 2.0.1+cu118
	- mujoco_py 2.1.2.14
	- gymnasium 0.29.1
 
# Running

Use the following command to run example code (locomotion learning of a quadruped robot).
```
python main.py
```

If you want to try different hyperparameter values, you can modify them according to the table below.

| Location | Parameter | Meaning  |
| ------------- | ------------- | ------------- |
| network.ini | W_TIME | transition speed/walking freqeuncy | 
| optimizer.ini | MINGRAD | gradient clipping (prevent exploding gradient) | 
|  | LR | learning rate | 
|  | SIGMA | starting exploration standard deviation (between 0.001-0.05)|
| main.py | NREPLAY | number of episodes/roll-outs used |
|  | NTIMESTEP | number of timesteps per episode | 
|  | NEPISODE | number of episode used for learning | 

Enjoy! With a proper set of hyperparameters, the robot should start walking within the first 70 episodes.

# Note
Implementation of ROGER is available at optimizer/roger.py.

# Reference

Arthicha Srisuchinnawong and Poramate Manoonpong (2025), Gain Tuning Is Not What You Need: Reward Gain Adaptation for Constrained Locomotion Learning, Robotics: Science and Systems.