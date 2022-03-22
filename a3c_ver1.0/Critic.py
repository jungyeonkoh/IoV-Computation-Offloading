import torch
import torch.nn as nn
import torch.nn.functional as F

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_space=None,
                       num_hidden_layer=2,
                       hidden_dim=None):

        super(Critic, self).__init__()

        # space size check
        assert state_space is not None, "None state_space input: state_space should be assigned."
        
        if hidden_dim is None:
            hidden_dim = state_space * 2

        self.layers = nn.ModuleList()

        # Add input layer
        self.layers.append(nn.Linear(state_space, hidden_dim))

        # Add hidden layers
        for i in range(num_hidden_layer):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, 1))

    def forward(self, x):

        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        out = self.layers[-1](x)

        return out