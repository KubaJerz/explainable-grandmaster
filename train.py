import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class SelfPlayDataset(Dataset):
    """Dataset wrapping self-play training samples."""

    def __init__(self, samples):
        """
        Args:
            samples: list of (state_tensor, policy_target, value_target)
        """
        self.states = torch.stack([s[0] for s in samples])
        self.policies = torch.stack([s[1] for s in samples])
        self.values = torch.stack([s[2] for s in samples])

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]


""" 
We build form: (Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm)silver2017masteringchessshogiselfplay 

as they do we do: "The neural network parameters are updated so as to minimise the error between the predicted outcome vt and the game outcome z, and to maximise the similarity of the policy vector pt to the search probabilities t. Specifically, the parameters are adjusted by gradient descent on a loss function l that sums over mean-squared error and cross-entropy losses respectively"

they also not have L2 regularizationin the paper we do viz weight decay in the optimizer
"""


def train(model, samples, epochs=5, batch_size=64, lr=1e-3, weight_decay=1e-4):
    """Train the model on self-play data using AlphaZero loss.

    Loss = MSE(value) + CE(policy) + L2 regularization (we do via weight decay in loss func)



    Args:
        model: BaseModel instance
        samples: list of (state_tensor, policy_target, value_target)
        epochs: number of training epochs
        batch_size: mini-batch size
        lr: learning rate
        weight_decay: L2 regularization strength

    Returns:
        list of per-epoch average losses
    """
    dataset = SelfPlayDataset(samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()

    epoch_losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for states, policy_targets, value_targets in dataloader:
            policy_logits, value_preds = model(states)
            value_preds = value_preds.squeeze(-1)

            # MSE
            value_loss = F.mse_loss(value_preds, value_targets)

            # Policy loss: cross-entropy with soft targets
            # CE = -sum(target * log_softmax(logits))
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.mean(torch.sum(policy_targets * log_probs, dim=1))

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
        print(f"    Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f} (policy: {avg_policy:.4f}, value: {avg_value:.4f})")

    return epoch_losses
