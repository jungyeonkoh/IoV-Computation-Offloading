import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.actor_softmax = nn.Linear(hidden_dim, action_space) #which server to offload
        self.actor_sigmoid = nn.Linear(hidden_dim, 1) # partial offloading
        #self.layers.append(nn.Linear(hidden_dim, action_space)) # which server to offload

    def forward(self, x):
        for layer in self.layers[:]:
            x = F.relu(layer(x))
        softmax = F.softmax(self.actor_softmax(x))
        sigmoid = F.sigmoid(self.actor_sigmoid(x))
        return softmax, sigmoid

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
