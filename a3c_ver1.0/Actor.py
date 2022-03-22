import torch
import torch.nn as nn
import torch.nn.functional as F

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_space=None,
                       action_space=None,
                       num_hidden_layer=2,
                       hidden_dim=None):

        super(Actor, self).__init__()

        # space size check
        assert state_space is not None, "None state_space input: state_space should be assigned."
        assert action_space is not None, "None action_space input: action_space should be assigned"
        

        if hidden_dim is None:
            hidden_dim = state_space * 2

        self.layers = nn.ModuleList()

        # Add input layer
        self.layers.append(nn.Linear(state_space, hidden_dim))

        # Add hidden layers
        for i in range(num_hidden_layer):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, action_space))

    def forward(self, x):

        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        out = F.softmax(self.layers[-1](x), dim=1)

        return out
