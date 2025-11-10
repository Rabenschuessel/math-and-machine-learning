# Filestructure

- chess_ml
    - data: 
    - env: 
        - Environment.py: env handling board and rewards
        - Rewards.py: reward functions
    - model: 
        - ChessNN.py: base class with wrappers for RL and legal move masking
        - FeedForward.py: feed forward implementation of base class
    - train: 
        - Immitation.py: training routine immitation learning 
        - Reinforcement.py: training routine reinforcement learning
- data: 
    - transform_data.py: transforms kaggle dataset to labeled dataset

# TODO

- [ ] RL loop 
- [ ] Immitation training
