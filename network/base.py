from abc import *

import torch
import torch.nn as nn
class NetworkBase(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        super(NetworkBase, self).__init__()
    @abstractmethod
    def forward(self, x):
        return x


    
class Network(NetworkBase):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function = torch.nn.functional.tanh,last_activation = None):
        super(Network, self).__init__()
        self.activation = activation_function
        self.last_activation = last_activation
        layers_unit = [input_dim]+ [hidden_dim]*(layer_num-1) 
        layers = ([nn.Linear(layers_unit[idx],layers_unit[idx+1]) for idx in range(len(layers_unit)-1)])
        self.layers = nn.ModuleList(layers)
        self.last_layer = nn.Linear(layers_unit[-1],output_dim)
        self.network_init()
        self.xs = []
        self.computexs = False
        self.gains = torch.ones((1,hidden_dim)).to('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        return self._forward(x)

    def _forward(self, x):
        if self.computexs:
            self.xs = []
        for i in range(len(self.layers)):
            if i == len(self.layers)-1:
                x = self.layers[i](x)
                if (len(x.shape) == 1):
                    gains = self.gains[0]
                else:
                    gains = self.gains
                
                x = self.activation(gains*x)
            else:
                x = self.activation(self.layers[i](x))
            if self.computexs:
                self.xs.append(x)

        x = self.last_layer(x)
        if self.last_activation != None:
            x = self.last_activation(x)
        return x

    def divgains(self,gains):
        with torch.no_grad():
            self.gains *= gains.detach()


    def _fforward(self, x):
        xs = []
        for i in range(len(self.layers)):
            x = self.activation(self.layers[i](x))
            xs.append(x)

        x = self.last_layer(x)
        if self.last_activation != None:
            x = self.last_activation(x)
        return x, xs[0], xs[1]

    def getactivity(self):
        return self.xs

    def network_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight,gain=1.41)
                layer.bias.data.zero_() 

