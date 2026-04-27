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


def train(model, samples, steps_per_iter=150, batch_size=64, lr=1e-2, momentum=0.9,
          weight_decay=1e-4, value_weight=0.25, device="cpu", decisive_weight=1.0,
          sample_with_replacement=False, log_chunks=3):
    """Train the model on self-play data using AlphaZero loss.

    Samples a fixed number of mini-batches (weighted toward decisive samples)
    from the buffer. Matches AlphaGo Zero's "N steps per iteration" rather than
    full epochs over the buffer. With sample_with_replacement=False (default),
    the draw is clamped to the buffer size so each sample is seen at most once;
    with replacement, samples can repeat across the steps_per_iter draws.

    Loss = CE(policy) + value_weight * MSE(value) + L2 (via weight_decay).
    Leela Chess Zero down-weights value (0.25) so the policy head dominates and
    the value head doesn't blow up early training.

    Args:
        model: BaseModel instance
        samples: list of (state_tensor, policy_target, value_target[, policy_weight])
        steps_per_iter: total mini-batches to train on this iteration
        batch_size: mini-batch size
        lr: learning rate
        momentum: SGD momentum (paper uses 0.9)
        weight_decay: L2 regularization strength
        value_weight: scalar weight on the MSE value loss (Leela uses 0.25)
        decisive_weight: sampling weight for decisive (non-draw) samples
        log_chunks: split steps into this many chunks for progress reporting

    Returns:
        list of per-chunk average losses (length log_chunks; preserves
        downstream consumers that read losses[-1])
    """
    dataset = SelfPlayDataset(samples, decisive_weight=decisive_weight)
    num_samples = steps_per_iter * batch_size
    if not sample_with_replacement:
        num_samples = min(num_samples, len(dataset))
    sampler = WeightedRandomSampler(dataset.weights, num_samples=num_samples,
                                    replacement=sample_with_replacement)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    model.train()

    chunk_size = max(1, steps_per_iter // log_chunks)
    chunk_losses = []
    chunk_total = 0.0
    chunk_policy = 0.0
    chunk_value = 0.0
    chunk_count = 0
    chunk_idx = 0

    for step, (states, policy_targets, value_targets, policy_weights) in enumerate(dataloader):
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

        loss = policy_loss + value_weight * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        chunk_total += loss.item()
        chunk_policy += policy_loss.item()
        chunk_value += value_loss.item()
        chunk_count += 1

        # Flush a chunk when it's full or we're at the last step.
        is_last = (step + 1) == steps_per_iter
        if chunk_count >= chunk_size or is_last:
            avg_loss = chunk_total / chunk_count
            avg_policy = chunk_policy / chunk_count
            avg_value = chunk_value / chunk_count
            chunk_losses.append({
                "total": avg_loss,
                "policy": avg_policy,
                "value": avg_value,
            })
            chunk_idx += 1
            print(f"    Chunk {chunk_idx}/{log_chunks} ({chunk_count} steps) "
                  f"- loss: {avg_loss:.4f} (policy: {avg_policy:.4f}, value: {avg_value:.4f})")
            chunk_total = chunk_policy = chunk_value = 0.0
            chunk_count = 0

    return chunk_losses
