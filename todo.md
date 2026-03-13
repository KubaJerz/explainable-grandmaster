
# TODO 
- [] make move_encoding.py
- [] finish MCTS
- [] Self-play loop (data gen loop)
    use NN to gen tuples of (state, pi vector after MCTS, G return for the episode)
- def loss function 
    loss = mse( state_value_pred, G return) + cross_entropy(model_pred_vec, pi vector after MCTS)
- [] Train Loop 
- [] Full pipline
- [] Run Training
