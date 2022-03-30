import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math

class Actor(nn.Module):
    def __init__(self, state_space=None, action_space=None, num_hidden_layer=2, hidden_dim=None):
        super(Actor, self).__init__()

        # state_space, action_space check
        assert state_space is not None, "None state_space input: state_space should be assigned."
        assert action_space is not None, "None action_space input: action_space should be assigned."

        if hidden_dim is None:
            hidden_dim = state_space * 2

        self.layers = nn. ModuleList()
        self.layers.append(nn.Linear(state_space, hidden_dim)) # input layer
        for i in range(num_hidden_layer): # hidden layer
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.LSTMCell(hidden_dim,hidden_dim))
        
        self.layers_act=nn.ModuleList()
        dim_num=hidden_dim
        for i in range(3): # hidden layer
            self.layers_act.append(nn.Linear(dim_num, int(dim_num/2)))
            dim_num=int(dim_num/2)
        self.layers_act.append(nn.Linear(dim_num,action_space))

        self.layers_partial=nn.ModuleList()
        dim_num=hidden_dim+1
        for i in range(3): # hidden layer
            self.layers_partial.append(nn.Linear(dim_num, int(math.floor(dim_num/2))))
            dim_num=int(math.floor(dim_num/2))
        self.layers_partial.append(nn.Linear(dim_num,1))

    def forward(self, x):
        x,(hx,cx)=x
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        hx,cx=self.layers[-1](x,(hx,cx))
        x=hx

        act_out=x
        for layer in self.layers_act[:-1]:
            act_out = F.relu(layer(act_out))
        out = F.softmax(self.layers_act[-1](act_out),dim=1)
        action_dist = Categorical(out)
        action = action_dist.sample()
        x=torch.cat([x,action.unsqueeze(1)],dim=1)
        

        for layer in self.layers_partial[:-1]:
            x = F.relu(layer(x))
        partial=torch.sigmoid(self.layers_partial[-1](x))

        return out,partial

class Critic(nn.Module):
    def __init__(self, state_space=None, num_hidden_layer=2, hidden_dim=None):
        super(Critic, self).__init__()

        # state_space check
        assert state_space is not None, "None state_space input: state_space should be assigned."

        if hidden_dim is None:
            hidden_dim = state_space * 2

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_space, hidden_dim)) # input layer
        for i in range(num_hidden_layer): # hidden layer
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim,1 )) # output layer

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        out = self.layers[-1](x)
        return out
