import torch
import math

#TODO make imports
from .utils import make_output_valid, is_terminal, terminal_sate_evaluation

class MCTSNode:
    def __init__(self, state, policy_len=4672):
        self.state = state
        self.prior_probs = torch.zeros(policy_len)
        self.W = torch.zeros(policy_len)
        self.visit_counts = torch.ones(policy_len)
        self.children = {} 
        self.value_sum = 0.0 
        
    @property
    def Q(self):
        # Calculate Q dynamically
        return self.W / self.visit_counts

class MCTS:
    def __init__(self, model, c_puct=1.0, tau=1.0):
        self.model = model
        self.c_puct = c_puct
        self.tau = tau
        self.root = None

    def mcts_search(self, state, nn_priors, num_simulations):
        self.root = MCTSNode(state, len(nn_priors))
        self.root.prior_probs = torch.tensor(nn_priors, dtype=torch.float32)
        
        for _ in range(num_simulations):
            self.simulate(self.root)

        return self.select_best_action()

    def select_action_UCT(self, node):
        # Removes illegal moves
        prior_probs = make_output_valid(node.prior_probs) 
        
        #  UCT Formula
        parent_visits = torch.sum(node.visit_counts)
        U = self.c_puct * prior_probs * (math.sqrt(parent_visits) / (1.0 + node.visit_counts))
        
        # Maximize Q + U
        action_scores = node.Q + U
        
        return torch.argmax(action_scores).item()

    def execute_action(self, state, action):
        # Execute the action on the state and return the *new* state
        pass

    def expand_node(self, parent_node, action):
        new_state = self.execute_action(parent_node.state, action)
        leaf_node = MCTSNode(new_state, len(parent_node.prior_probs))
        
        self.model.eval()
        with torch.no_grad():
            # Get priors and value from the neural network for the new state
            nn_priors, value = self.model(new_state)
            leaf_node.prior_probs = nn_priors.squeeze() 
            leaf_node.value_sum = value.item()
            
        return leaf_node, leaf_node.value_sum

    def simulate(self, node):
        """
        Recursively traverses the tree until an unexpanded node or terminal state is reached.
        Returns the evaluation value, which cascades back up the recursion stack.
        """
        if is_terminal(node.state):
            # Evaluate terminal states using game rules rather than the neural network
            return terminal_sate_evaluation(node.state)
            
        action = self.select_action_UCT(node)
        
        if action in node.children:
            # Node is already expanded: recurse deeper into the tree
            child_node = node.children[action]
            sim_value = self.simulate(child_node)
        else:
            # Node is unexplored: expand it, grab the network evaluation, and stop searching
            child_node, sim_value = self.expand_node(node, action)
            node.children[action] = child_node
            
        # --- BACKPROPAGATION ---
        # As the recursion unwinds, update W and visit counts for every node traversed
        node.W[action] += sim_value
        node.visit_counts[action] += 1
        
        return sim_value
        
    def select_best_action(self):
        # Select action based on visit counts and temperature scaling (tau)
        if self.tau == 0:
            best_action = torch.argmax(self.root.visit_counts).item()
        else:
            counts = self.root.visit_counts ** (1.0 / self.tau)
            probs = counts / torch.sum(counts)
            best_action = torch.argmax(probs).item()
            
        return best_action