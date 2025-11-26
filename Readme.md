# Get Started

1. download [dataset](https://www.kaggle.com/datasets/tianmin/lichess-chess-puzzle-dataset) to ./data/ 
2. unzip dataset to filename `./data/lichess-chess-puzzle-transformed.csv`
3. transform dataset with `python data/transform_data.py` (takes about ~20min)
4. train/test/save a model with `python -m chess_ml.train.Immitation` (takes more than ~20min)


# Filestructure

```
.
├── chess_ml
│   ├── data
│   │   └── Puzzles.py
│   ├── env
│   │   ├── Environment.py: env handling board and rewards
│   │   └── Rewards.py: reward functions
│   ├── model
│   │   ├── ChessNN.py: base class with wrappers for RL and legal move masking
│   │   └── FeedForward.py: feed forward implementation of base class
│   ├── test.py
│   └── train
│       ├── Immitation.py: training routine immitation learning 
│       └── Reinforcement.py: training routine reinforcement learning
└── data
    ├── lichess_puzzle_labeled.csv
    ├── lichess_puzzle_transformed.csv
    ├── lichess_transformed.csv
    └── transform_data.py: transforms kaggle dataset to labeled dataset
```


# Study

Immitation, Immitation + RL, RL 
RL: 3-4 reward function sets? 

Training: 

- Immitation
- Immitation reward1
- Immitation reward2
- Immitation reward3
- Immitation reward4
- Reward1
- Reward2
- Reward3
- Reward4


Testing: 

1000 games against models and stockfish




# TODO



1.  
    - [ ] Model Architecture 
        - [ ] Network Structure (CNN,...)
        - [ ] Hyperparameter Tuning (hidden layer # and size)
2. 
    - [ ] Overlay
    - [ ] Controller for parameters
    - [ ] Arena against stockfish


Patric: 

- [ ] Slides 
- [ ] Saving statistics in training and playing
    - [ ] Training 
    - [ ] Playing 
    - [ ] Environment
- [ ] 2 paper RL und hyperparam

Jannis: 

- [ ] Related Works (existing architectures,...) -> in directory on git
    - [ ] Hyperparameter tuning (general/chess specific)
        - [ ] architecture (convolutional?)
        - [ ] learning rate
        - [ ] number / size of layers 
    - [ ] Reinformcenet learning reward modelling 
        - [ ] weighting of different reward functions 
        - [ ] normalization of rewards,...
        - [ ] discounted rewards gamma
    - [ ] Existing models 

Jakob: 

- [ ] auf cluster laufen lassen
    - [ ] immitation learning 
    - [ ] reinforcement learning 
    - [ ] arena results
- [x] RL 
    - [x] Selfplay script
    - [x] Reward Functions
- [ ] improvements
    - [ ] cnn 
    - [ ] masking illegal moves in immitation learning
    - [ ] bigger batch size immitation learning 512-4096
