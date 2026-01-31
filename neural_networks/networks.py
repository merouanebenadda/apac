import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
c_speed = 5.0  # Kinetic coefficient of hamiltonian (= max speed)
gamma_obstacle = 5.0
weight_goal = 1.0
weight_congestion = 1.0

# Cost functions

def f_obstacle(x):
    """
    x: (Batch, 2) tensor
    """
    matrix = torch.tensor([[5, 0],
                       [0, -1]], device=device)
    term = torch.relu(-(5.0 * x[:, 0]**2 - 1.0 * x[:, 1]**2) - 0.1 , 0)

    return gamma_obstacle*term

def F_congestion(x, y):
    """
    x: (Batch, 2) tensor
    y: (Batch, 2) tensor
    """
    squared_dist = torch.sum((x - y)**2, dim=1, keepdim=True)

    interaction = 1.0/(squared_dist + 1.0)

    return weight_congestion * torch.mean(interaction)

def distance_to_goal(x, goal):
    """
    x: (Batch, 2) tensor
    goal: (2,) tensor
    """
    return weight_goal * torch.norm(x - goal, p=2, dim=1, keepdim=True)


def Hamiltonian(x, p):
    """
    x: (Batch, 2) tensor

    H(x, p) = c * ||p||_2 + f_obstacle(x)
    """

    return c_speed * torch.norm(p, p=2, dim=1, keepdim=True) + f_obstacle(x)

# Neural Networks

class ResBlock(nn.Module):
    def __init__(self, dim, activation):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.activation = activation
        self.skip_weight = 0.5

    