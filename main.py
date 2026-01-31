import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from neural_networks.networks import PhiNet, GenNet, phi_loss, gen_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Problem Parameters
d = 2  # Dimension of the problem

# Training Parameters
num_epochs = 50000
batch_size = 512
print_interval = 500


# Initialize Networks

phi_net = PhiNet(d).to(device)
gen_net = GenNet(d).to(device)

# Optimizers

opt_phi = optim.Adam(
    phi_net.parameters(), 
    lr=4e-4,
    betas = (0.5, 0.9),
    weight_decay=1e-4
)
opt_G = optim.Adam(
    gen_net.parameters(), 
    lr=1e-4,
    betas = (0.5, 0.9),
    weight_decay=1e-4
)

# Schedulers

scheduler_phi = torch.optim.lr_scheduler.StepLR(
    opt_phi,
    step_size=15000,
    gamma=0.5
)

scheduler_G = torch.optim.lr_scheduler.StepLR(
    opt_G,
    step_size=15000,
    gamma=0.5
)

# Training Networks

print("Training started...")

for epoch in range(num_epochs+1):
    # Train Phi Network
    opt_phi.zero_grad()
    loss_phi_val = phi_loss(phi_net, gen_net, batch_size, device)
    loss_phi_val.backward()
    opt_phi.step()
    scheduler_phi.step()

    # Train Generator Network
    opt_G.zero_grad()
    loss_G_val = gen_loss(phi_net, gen_net, batch_size, device)
    loss_G_val.backward()
    opt_G.step()
    scheduler_G.step()

    if epoch % print_interval == 0:
        print(f"Epoch [{epoch}/{num_epochs}] | Phi Loss: {loss_phi_val.item():.4f} | Gen Loss: {loss_G_val.item():.4f}")

print("Training completed.")