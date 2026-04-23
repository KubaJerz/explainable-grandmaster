import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class SelfPlayDataset(Dataset):
    """Dataset wrapping self-play training samples."""

    def __init__(self, samples, decisive_weight=1.0):
        """
        Args:
            samples: list of (state_tensor, policy_target, value_target[, policy_weight])
            decisive_weight: sampling weight for decisive (non-draw) samples
        """
        self.states = torch.stack([s[0] for s in samples])
        self.policies = torch.stack([s[1] for s in samples])
        self.values = torch.stack([s[2] for s in samples])
        self.policy_weights = torch.stack([
            s[3] if len(s) > 3 else torch.tensor(1.0, dtype=torch.float32)
            for s in samples
        ])
        # Decisive samples (value != 0) get higher weight
        self.weights = torch.where(
            self.values.abs() > 0,
            torch.tensor(decisive_weight),
            torch.tensor(1.0),
        )

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx], self.policy_weights[idx]


"""
We build form: (Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm)silver2017masteringchessshogiselfplay

as they do we do: "The neural network parameters are updated so as to minimise the error between the predicted outcome vt and the game outcome z, and to maximise the similarity of the policy vector pt to the search probabilities t. Specifically, the parameters are adjusted by gradient descent on a loss function l that sums over mean-squared error and cross-entropy losses respectively"

they also not have L2 regularizationin the paper we do viz weight decay in the optimizer
"""


def train(model, samples, epochs=5, batch_size=64, lr=1e-3, weight_decay=1e-4,
          device="cpu", decisive_weight=1.0):
    """Train the model on self-play data using AlphaZero loss.

    Loss = MSE(value) + CE(policy) + L2 regularization (we do via weight decay in loss func)

    Args:
        model: BaseModel instance
        samples: list of (state_tensor, policy_target, value_target[, policy_weight])
        epochs: number of training epochs
        batch_size: mini-batch size
        lr: learning rate
        weight_decay: L2 regularization strength
        decisive_weight: sampling weight for decisive (non-draw) samples

    Returns:
        list of per-epoch average losses
    """
    dataset = SelfPlayDataset(samples, decisive_weight=decisive_weight)
    sampler = WeightedRandomSampler(dataset.weights, num_samples=len(dataset), replacement=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()

    epoch_losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for states, policy_targets, value_targets, policy_weights in dataloader:
            states = states.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            policy_weights = policy_weights.to(device)
            policy_logits, value_preds = model(states)
            value_preds = value_preds.squeeze(-1)

            value_loss = F.mse_loss(value_preds, value_targets)

            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_losses = -torch.sum(policy_targets * log_probs, dim=1)
            policy_weight_sum = policy_weights.sum()
            if policy_weight_sum.item() > 0:
                policy_loss = torch.sum(policy_losses * policy_weights) / policy_weight_sum
            else:
                policy_loss = torch.zeros((), device=device)

            loss = value_loss + policy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_policy = total_policy_loss / num_batches
        avg_value = total_value_loss / num_batches
        epoch_losses.append({
            "total": avg_loss,
            "policy": avg_policy,
            "value": avg_value,
        })
        print(f"    Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} (policy: {avg_policy:.4f}, value: {avg_value:.4f})")

    return epoch_losses
