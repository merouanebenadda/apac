import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


from neural_networks.networks import PhiNet, GenNet, phi_loss, gen_loss, initial_pos_mean


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Problem Parameters
d = 2  # Dimension of the problem

# Training Parameters
num_epochs = 200000
batch_size = 128
print_interval = 2000


# Visualization Parameters
num_frames = 64
num_agents = 50

# Initialize Networks

phi_net = PhiNet(d).to(device)
gen_net = GenNet(d).to(device)

# Loss histories
history_phi = []
history_G = []

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
    gamma=0.9
)

scheduler_G = torch.optim.lr_scheduler.StepLR(
    opt_G,
    step_size=15000,
    gamma=0.9
)

# Training Networks

print("Training started...")

for epoch in range(num_epochs+1):
    # Train Phi Network
    opt_phi.zero_grad()
    loss_phi_val = phi_loss(phi_net, gen_net, batch_size, device, d)
    loss_phi_val.backward()
    opt_phi.step()
    scheduler_phi.step()

    # Train Generator Network
    opt_G.zero_grad()
    loss_G_val = gen_loss(phi_net, gen_net, batch_size, device, d)
    loss_G_val.backward()
    opt_G.step()
    scheduler_G.step()

    # Record losses
    history_phi.append(loss_phi_val.item())
    history_G.append(loss_G_val.item())

    if epoch % print_interval == 0:
        print(f"Epoch [{epoch}/{num_epochs}] | Phi Loss: {loss_phi_val.item():.4f} | Gen Loss: {loss_G_val.item():.4f}")

        # Visualization

        with torch.no_grad():
            time_steps = torch.linspace(0, 1, num_frames, device=device)
            
            z = torch.randn(num_agents, d, device=device)
            std_dev = 1.0 / np.sqrt(10.0)
            z = z * std_dev + initial_pos_mean

            fig, ax = plt.subplots(1, 2, figsize=(16, 6))

            # Plot trajectories

            # Obstacle
            grid_x = np.linspace(-3, 3, 200)
            grid_y = np.linspace(-3, 3, 200)
            GX, GY = np.meshgrid(grid_x, grid_y)
            Val = GY**2 - 5*GX**2 - 0.1
            ax[0].contourf(GX, GY, Val, levels=[0, 100], colors=['red'], alpha=0.15)
            ax[0].contour(GX, GY, Val, levels=[0], colors='darkred', linewidths=2)

            cmap = plt.get_cmap('coolwarm') # Colormap for time steps

            for i, t_val in enumerate(time_steps):
                t_batch = torch.ones(num_agents, 1, device=device) * t_val
 
                x_pred = gen_net(z, t_batch).cpu().numpy() # .cpu() so Matplotlib can handle it

                color = cmap(i / num_frames)

                lbl = None
                if i == 0: lbl = 'Start (t=0)'
                elif i == num_frames - 1: lbl = 'End (t=1)'

                ax[0].scatter(x_pred[:, 0], x_pred[:, 1], color=color, s=10, label=lbl)

            # Target position
            ax[0].scatter([2], [2], c='green', marker='x', s=150, linewidth=3, label='Target')
            
            ax[0].set_xlim(-3, 3); ax[0].set_ylim(-3, 3)
            ax[0].set_title(f"Trajectories (Gradient t=0 to t=1) - Step {epoch}")
            ax[0].legend(loc='upper right')
            ax[0].grid(alpha=0.3)

            # Plot loss history
            ax[1].plot(history_phi, label='Loss Phi (Max)', alpha=0.7, linewidth=1)
            ax[1].plot(history_G, label='Loss G (Min)', alpha=0.7, linewidth=1)
            ax[1].set_yscale('symlog')
            ax[1].set_title("Loss history")
            ax[1].legend(); ax[1].grid(alpha=0.3)


            plt.tight_layout()
            plt.savefig(f"outputs/training_step_{epoch:05d}.png", dpi=150, bbox_inches='tight')
                



print("Training completed.")
