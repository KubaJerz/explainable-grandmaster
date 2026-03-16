
# TODO 
- [x] make move_encoding.py
- [x] finish MCTS
- [x] Check MCTS
- [x] Self-play loop (data gen loop)
    use NN to gen tuples of (state, pi vector after MCTS, G return for the episode)
- [x] def loss function 
    loss = mse( state_value_pred, G return) + cross_entropy(model_pred_vec, pi vector after MCTS)
- [x] Train Loop 
- [x] Full pipline
- [] Run Training

## deviations from Paper
- [x] MCTS backprop doesn't negate value across alternating players when doing backprop
- [] Add Dirichlet noise at root of mcts as in tha paper
- [x] Replay buffer across iterations (paper uses last "some amount of" games, not just current iter)
- [x] Add BN+ReLU to stem (base.py) — paper specifies conv→BN→ReLU
- [] should we switch to SGD w/ momentum 0.9 instead of Adam (per Silver paper)
- [] Resignation mechanism
