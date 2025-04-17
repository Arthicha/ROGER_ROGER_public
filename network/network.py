import torch
import torch.nn as nn
import torch.nn.functional as F

from network.base import Network

class Xtor(Network):

    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function = torch.tanh,last_activation = None, trainable_std = False):
        super(Xtor, self).__init__(layer_num, input_dim, output_dim, hidden_dim, lambda x: x ,lambda x: x)
        
    def forward(self, x):
        mu = self._forward(x)

        return mu, None


class Actor(Network):

    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function = torch.tanh,last_activation = None, trainable_std = False):
        super(Actor, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation)
        self.trainable_std = trainable_std
        if self.trainable_std == True:
            self.logstd = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, x):
        mu = self._forward(x)
        
        if self.trainable_std == True:
            std = torch.exp(self.logstd)
        else:
            logstd = torch.zeros_like(mu)
            std = torch.exp(logstd)
        return mu,std

class Actor2(Network):

    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function = torch.tanh,last_activation = None, trainable_std = False):
        super(Actor2, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation)
        self.trainable_std = trainable_std
        if self.trainable_std == True:
            self.logstd = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, x):
        mu, mu2, mu3 = self._fforward(x)
        return mu, mu2, mu3



class Critic(Network):
    
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function, last_activation = None):
        super(Critic, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation)
        
    def forward(self, *x):
        x = torch.cat(x,-1)
        return self._forward(x)
    
    