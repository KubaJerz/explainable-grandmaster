import math

import torch

from utils.game_utils import make_output_valid, is_terminal, terminal_state_evaluation, index_to_move, GameState

# Dominik Klein Neural Networks for Chess
# Mastering the game of Go without human knowledge (Silver)


class MCTSNode:
    def __init__(self, game_state, policy_len=4672):
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
    def __init__(self, evaluate_fn, c_puct=1.0, tau=1.0):
        self.evaluate_fn = evaluate_fn
        self.c_puct = c_puct
        self.tau = tau
        self.root = None

    def mcts_search(self, game_state, num_simulations):
        # Encode state and get initial policy/value from the network
        tensor = game_state.encode()
        nn_priors, value = self.evaluate_fn(tensor)

        self.root = MCTSNode(game_state)
        self.root.set_prior_probs(nn_priors)
        self.root.value_sum = value

        for _ in range(num_simulations):
            self.simulate(self.root)

        return self.select_best_action()

    def select_action_UCT(self, node):
        # UCT Formula (Silver page 355)
        parent_visits = torch.sum(node.visit_counts)
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

    def get_policy(self):
        """Extract the MCTS policy from root visit counts, respecting temperature."""
        visit_counts = self.root.visit_counts
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
