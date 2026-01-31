import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Target position and initial distribution
initial_pos_mean = torch.tensor([-2.0, 0.0], device=device)
target = torch.tensor([2.0, 2.0], device=device)


# Hyperparameters
c_speed = 5.0  # Kinetic coefficient of hamiltonian (= max speed)
gamma_obstacle = 5.0
weight_goal = 3.0
weight_congestion = 1.0
nu = 0.1  # Diffusion coefficient
lam = 1.0  # Weight for HJB residual in phi loss

# Cost functions

def f_obstacle(x):
    """
    x: (Batch, 2) tensor
    """
    matrix = torch.tensor([[5, 0],
                       [0, -1]], device=device)
    term = torch.relu(-(5.0 * x[:, 0]**2 - 1.0 * x[:, 1]**2) - 0.1)

    return gamma_obstacle*term

def F_congestion(x, y):
    """
    x: (Batch, 2) tensor
    y: (Batch, 2) tensor
    """
    squared_dist = torch.sum((x - y)**2, dim=1, keepdim=True)

    interaction = 1.0/(squared_dist + 1.0)

    return weight_congestion * torch.mean(interaction)

def g(x, goal=target):
    """
    x: (Batch, 2) tensor
    goal: (2,) tensor
    """
    return weight_goal * torch.norm(x - goal, p=2, dim=1, keepdim=True)


def Hamiltonian(p):
    """
    x: (Batch, 2) tensor

    H(x, p) = c * ||p||_2 + f_obstacle(x)
    """

    return c_speed * torch.norm(p, p=2, dim=1, keepdim=True)


# Neural Networks

class ResBlock(nn.Module):
    def __init__(self, dim, activation):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.activation = activation
        self.skip_weight = 0.5

    def forward(self, x):
        residual = x
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        return self.activation(out * self.skip_weight + residual) 
    
class ResNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=100, activation=nn.Tanh):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)

        self.block1 = ResBlock(hidden_dim, activation())
        self.block2 = ResBlock(hidden_dim, activation())
        self.block3 = ResBlock(hidden_dim, activation())
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        out = self.fc_in(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.fc_out(out)
        return out
    
class PhiNet(nn.Module):
    def __init__(self, d, hidden_dim=100):
        super().__init__()
        self.N = ResNet(in_dim=d + 1, out_dim=1, hidden_dim=100, activation=nn.Tanh)

    def forward(self, x, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        
        return (1-t)*self.N(torch.cat([x, t], dim=1)) + t*g(x)
    
class GenNet(nn.Module):
    def __init__(self, d, hidden_dim=100):
        super().__init__()
        self.N = ResNet(in_dim=d + 1, out_dim=d, hidden_dim=100, activation=nn.ReLU)

    def forward(self, z, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)

        return (1-t)*z + t*self.N(torch.cat([z, t], dim=1))
    
# Loss Functions

def sample_batch(batch_size, d, device):
    z = torch.randn(batch_size, d, device=device)

    # Shift and scale to match initial distribution rho_0
    std_dev = 1.0 / np.sqrt(10.0)
    z = z * std_dev
    z[:, 0] = z[:, 0] - 2.0

    t = torch.rand(batch_size, 1, device=device)
    return z, t

def phi_loss(phi_net, gen_net, batch_size, device, d):
    z, t = sample_batch(batch_size, d, device)

    z2, _ = sample_batch(batch_size, d, device) # Independent samples for congestion term

    x = gen_net(z, t).detach().requires_grad_(True) # Generated positions at time t

    t = t.clone().detach().requires_grad_(True)

    # Compute gradients and Laplacian of phi
    phi = phi_net(x, t)
    grad_phi = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
    grad_t = torch.autograd.grad(phi, t, grad_outputs=torch.ones_like(phi), create_graph=True)[0]

    lap_phi = 0
    for i in range(d):
        grad2 = torch.autograd.grad(grad_phi[:, i], x, grad_outputs=torch.ones_like(grad_phi[:, i]), create_graph=True)[0][:, i]
        lap_phi += grad2
    lap_phi = lap_phi.unsqueeze(1)


    with torch.no_grad():
        x2 = gen_net(z2, t) # Independent samples for congestion term
    f_cong = F_congestion(x, x2)

    kinetic = Hamiltonian(grad_phi)

    lt_residual = grad_t + nu*lap_phi - kinetic
    l_HJB_residual = lt_residual + f_obstacle(x) + f_cong

    t0 = torch.zeros_like(t)
    l0 = phi_net(z, t0).mean()

    lt = lt_residual.mean()
    l_HJB = lam * (l_HJB_residual.norm(dim=1)).mean()

    return l_HJB - (l0 + lt)


def gen_loss(phi_net, gen_net, batch_size, device, d):
    z, t = sample_batch(batch_size, d, device)
    z2, _ = sample_batch(batch_size, d, device) # Independent samples for congestion term

    x = gen_net(z, t) # Generated positions at time t

    t_ref = t.clone().detach().requires_grad_(True)

    phi = phi_net(x, t_ref)
    grad_phi = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
    grad_t = torch.autograd.grad(phi, t_ref, grad_outputs=torch.ones_like(phi), create_graph=True)[0]

    lap_phi = 0
    for i in range(d):
        grad2 = torch.autograd.grad(grad_phi[:, i], x, grad_outputs=torch.ones_like(grad_phi[:, i]), create_graph=True)[0][:, i]
        lap_phi += grad2
    lap_phi = lap_phi.unsqueeze(1)

    kinetic = Hamiltonian(grad_phi)

    f_obst = f_obstacle(x)

    with torch.no_grad():
        x2 = gen_net(z2, t) # Independent samples for congestion term

    f_cong = F_congestion(x, x2)

    lt_residual = grad_t + nu*lap_phi - kinetic + f_obst + f_cong

    return lt_residual.mean()