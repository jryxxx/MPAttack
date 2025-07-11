import torch
import matplotlib.pyplot as plt

# Load the loss data
data1 = torch.load("results_sp/loss/dino/dino_single_losses.pt")
data2 = torch.load("results_sp/loss/glip/glip_single_losses.pt")

# Set the target length
target_length = 50000

# Pad data1
if len(data1) < target_length:
    padding1 = torch.zeros(target_length - len(data1))
    data1 = torch.cat([data1, padding1])

# Pad data2
if len(data2) < target_length:
    padding2 = torch.zeros(target_length - len(data2))
    data2 = torch.cat([data2, padding2])

ymin = min(data1.min().item(), data2.min().item())
ymax = max(data1.max().item(), data2.max().item())

# Create subplots: 2 rows, 1 column
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# First subplot
axes[0].plot(data1, color="blue")
axes[0].set_title("Single Prompt Loss")
axes[0].set_ylabel("Loss")
axes[0].set_ylim(ymin, ymax)
axes[0].grid(True)

# Second subplot
axes[1].plot(data2, color="green")
axes[1].set_title("Multi Prompt Loss")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Loss")
axes[1].set_ylim(ymin, ymax)
axes[1].grid(True)

# Adjust layout
plt.tight_layout()
plt.savefig('loss.png')
