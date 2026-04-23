import json
import numpy as np
import matplotlib.pyplot as plt

with open("results/training_log.json") as f:
    log = json.load(f)

# Build per-epoch data with cumulative epoch count as x-axis
epochs = []
total_losses = []
policy_losses = []
value_losses = []
iter_starts = []  # epoch index where each iteration begins
cumulative_games_per_epoch = []

epoch_count = 0
cum_games = 0

for entry in log:
    iter_starts.append(epoch_count)
    cum_games += entry["self_play"]["num_games"]
    for ep in entry["training"]:
        epochs.append(epoch_count)
        total_losses.append(ep["total"])
        policy_losses.append(ep["policy"])
        value_losses.append(ep["value"])
        cumulative_games_per_epoch.append(cum_games)
        epoch_count += 1

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(epochs, total_losses, "-", label="Total", linewidth=1)
ax1.plot(epochs, policy_losses, "-", label="Policy", linewidth=1)
ax1.plot(epochs, value_losses, "-", label="Value", linewidth=1)

ax1.set_xlabel("Training Epoch")
ax1.set_ylabel("Loss")

ax2 = ax1.twinx()
ax2.plot(epochs, cumulative_games_per_epoch, "d-", color="gray", alpha=0.5, label="Games", markersize=2)
ax2.set_ylabel("Cumulative Self-Play Games")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", bbox_to_anchor=(0.08, 1.0))

plt.title("Training Loss")
ax1.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("results/loss_plot.png", dpi=150)
plt.show()
print("Saved to results/loss_plot.png")