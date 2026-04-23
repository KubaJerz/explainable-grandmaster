import math

import torch

from utils.game_utils import ACTION_SIZE, GameState, index_to_move, is_terminal, make_output_valid, terminal_state_evaluation

# Dominik Klein Neural Networks for Chess
# Mastering the game of Go without human knowledge (Silver)


class MCTSNode:
    def __init__(self, game_state, policy_len=ACTION_SIZE):
        self.game_state = game_state  # GameState (board + history)
        self.prior_probs = torch.zeros(policy_len)
        self.W = torch.zeros(policy_len)
        self.visit_counts = torch.zeros(policy_len)
        self.children = {}
        self.value_sum = 0.0

    @property
    def Q(self):
        return self.W / torch.clamp(self.visit_counts, min=1.0)

    def set_prior_probs(self, priors):
        priors = make_output_valid(priors, self.game_state.board)
        self.prior_probs = priors


class MCTS:
    """Monte Carlo Tree Search implementation for chess.

    Initialize at the node from which to run the search, then call mcts_search
    to perform the search and get the best action.
    """
    def __init__(self, evaluate_fn, c_puct=1.0, tau=1.0, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        self.evaluate_fn = evaluate_fn
        self.c_puct = c_puct
        self.tau = tau
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.root = None

    def mcts_search(self, game_state, num_simulations):
        # Encode state and get initial policy/value from the network
        tensor = game_state.encode()
        nn_priors, value = self.evaluate_fn(tensor)

        self.root = MCTSNode(game_state)
        self.root.set_prior_probs(nn_priors)
        self.root.value_sum = value

        # Add Dirichlet noise to root priors for exploration (Silver et al.)
        # P(s,a) = (1 - ε) * p_a + ε * η_a, where η ~ Dir(α)
        self._root_noise = torch.zeros_like(self.root.prior_probs)
        if self.dirichlet_epsilon > 0:
            legal_mask = self.root.prior_probs > 0
            num_legal = legal_mask.sum().item()
            if num_legal > 0:
                noise = torch.zeros_like(self.root.prior_probs)
                dist = torch.distributions.Dirichlet(torch.full((int(num_legal),), self.dirichlet_alpha))
                noise[legal_mask] = dist.sample()
                self._root_noise = noise
                self.root.prior_probs = (1 - self.dirichlet_epsilon) * self.root.prior_probs + self.dirichlet_epsilon * noise

        for _ in range(num_simulations):
            self.simulate(self.root)

        return self.select_best_action()

    def select_action_UCT(self, node):
        # UCT Formula (Silver page 355)
        parent_visits = torch.sum(node.visit_counts)
        if parent_visits.item() == 0:
            return torch.multinomial(node.prior_probs, 1).item()
        U = self.c_puct * node.prior_probs * (math.sqrt(parent_visits) / (1.0 + node.visit_counts))

        action_scores = node.Q + U
        # Mask illegal moves (prior == 0) to -inf so they're never selected
        illegal_mask = node.prior_probs == 0
        action_scores[illegal_mask] = float('-inf')
        return torch.argmax(action_scores).item()

    def expand_node(self, parent_node, action):
        move = index_to_move(action, parent_node.game_state.board)
        new_state = parent_node.game_state.apply_move(move)
        leaf_node = MCTSNode(new_state, len(parent_node.prior_probs))

        if is_terminal(new_state.board):
            value = terminal_state_evaluation(new_state.board)
        else:
            tensor = new_state.encode()
            nn_priors, value = self.evaluate_fn(tensor)
            leaf_node.set_prior_probs(nn_priors)
        leaf_node.value_sum = value

        return leaf_node, leaf_node.value_sum

    def simulate(self, node):
        """
        Recursively traverses the tree until an unexpanded node or terminal state is reached.
        Returns the evaluation value, which cascades back up the recursion stack.
        """
        if is_terminal(node.game_state.board):
            return terminal_state_evaluation(node.game_state.board)

        action = self.select_action_UCT(node)

        if action in node.children:
            child_node = node.children[action]
            sim_value = self.simulate(child_node)
        else:
            child_node, sim_value = self.expand_node(node, action)
            node.children[action] = child_node

        # --- BACKPROPAGATION ---
        # Negate: child's value is from the child's perspective (opponent),
        # so flip sign to get value from this node's (current player's) perspective.
        negated_value = -sim_value
        node.W[action] += negated_value
        node.visit_counts[action] += 1

        return negated_value

    def get_policy(self, prune_noise_visits=False):
        """Extract the MCTS policy from root visit counts, respecting temperature.

        If prune_noise_visits=True, subtracts the expected visit contribution from
        Dirichlet noise before computing the policy (KataGo policy target pruning).
        This makes training targets reflect the network's own search, not forced exploration.
        """
        visit_counts = self.root.visit_counts.clone()

        if prune_noise_visits and self.dirichlet_epsilon > 0:
            # Estimate visits attributable to noise: ε * noise_prior * total_visits
            # Subtract these so the policy target reflects the network's search, not noise
            total_visits = visit_counts.sum()
            noise_visits = self.dirichlet_epsilon * self._root_noise * total_visits
            visit_counts = torch.clamp(visit_counts - noise_visits, min=0)

        if self.tau <= 0.01:
            # Greedy: one-hot on most-visited
            policy = torch.zeros_like(visit_counts)
            policy[torch.argmax(visit_counts)] = 1.0
        else:
            counts = visit_counts ** (1.0 / self.tau)
            total = counts.sum()
            policy = counts / total if total > 0 else counts
        return policy

    def select_best_action(self):
        policy = self.get_policy()
        if self.tau > 0.01:
            return torch.multinomial(policy, 1).item()
        return torch.argmax(policy).item()
