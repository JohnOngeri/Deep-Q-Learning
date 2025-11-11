# Deep Q-Network (DQN) for Atari Games

This project implements a Deep Q-Network agent using Stable Baselines3 and Gymnasium to play Atari games.

## 📋 Project Structure

\`\`\`
.
├── train.py              # Training script for DQN agent
├── play.py              # Evaluation script to watch trained agent
├── models/              # Saved models directory
│   ├── dqn_model.zip
│   └── best_dqn_model.zip
└── logs/                # Training logs and checkpoints
\`\`\`

## 🚀 Installation

Install the required dependencies:

\`\`\`bash
pip install gymnasium[atari]
pip install stable-baselines3[extra]
pip install ale-py
pip install tensorboard
pip install matplotlib
\`\`\`

For ROMs, you may need to install:
\`\`\`bash
pip install "gymnasium[accept-rom-license]"
\`\`\`

## 🎮 Usage

### Training

Train a DQN agent on an Atari game:

\`\`\`bash
python train.py
\`\`\`

**Configuration**: Edit the `CONFIG` dictionary in `train.py` to:
- Change the Atari environment (`env_name`)
- Adjust hyperparameters (learning rate, gamma, batch size, etc.)
- Modify training duration (`total_timesteps`)

**Popular Atari Games**:
- `'ALE/Breakout-v5'` - Break bricks with a paddle
- `'ALE/Pong-v5'` - Classic Pong game
- `'ALE/SpaceInvaders-v5'` - Shoot aliens
- `'ALE/MsPacman-v5'` - Navigate mazes

### Evaluation

Watch your trained agent play:

\`\`\`bash
python play.py
\`\`\`

The agent will use a **GreedyQPolicy** (no exploration, only exploitation) to maximize performance.

## 🔬 Understanding DQN Components

### 1. Q-Learning Basics

**Q-Value**: Expected cumulative reward for taking action `a` in state `s`
- Formula: `Q(s,a) = r + γ * max(Q(s',a'))`
- `r`: immediate reward
- `γ` (gamma): discount factor (how much we value future rewards)
- `s'`: next state

### 2. Deep Q-Network

Instead of storing Q-values in a table, DQN uses a neural network to approximate Q-values:
- **Input**: Game frames (stacked for temporal information)
- **Output**: Q-values for each possible action
- **Training**: Minimize difference between predicted and target Q-values

### 3. Key DQN Features

**Experience Replay**:
- Stores experiences `(s, a, r, s')` in a replay buffer
- Samples random mini-batches for training
- Breaks correlation between consecutive experiences

**Target Network**:
- Separate network for computing target Q-values
- Updated periodically to stabilize training
- Prevents moving target problem

**Epsilon-Greedy Exploration**:
- With probability `ε`: take random action (explore)
- With probability `1-ε`: take best action (exploit)
- `ε` decays from `epsilon_start` to `epsilon_end` over `exploration_fraction` of training

## 📊 Hyperparameter Tuning Guide

### Learning Rate (`learning_rate`)
- **Lower (1e-5)**: Slower but more stable learning
- **Higher (1e-3)**: Faster but potentially unstable
- **Recommended**: Start with 1e-4

### Gamma (γ) - Discount Factor (`gamma`)
- **Lower (0.9)**: Values immediate rewards more
- **Higher (0.99)**: Values future rewards more
- **Effect**: Higher gamma needed for games with delayed rewards

### Batch Size (`batch_size`)
- **Smaller (16-32)**: Faster updates, more noise
- **Larger (64-128)**: Slower updates, more stable gradients
- **Trade-off**: Memory vs. stability

### Exploration Parameters
- `exploration_initial_eps`: Start with high exploration (1.0)
- `exploration_final_eps`: Minimum exploration (0.01-0.05)
- `exploration_fraction`: How much of training to decay epsilon (0.1-0.2)

### Buffer Size (`buffer_size`)
- **Smaller**: Less diverse experiences, faster replay
- **Larger**: More diverse experiences, better generalization
- **Recommended**: 100,000 - 1,000,000

## 📈 Monitoring Training

### TensorBoard

View training progress in real-time:

\`\`\`bash
tensorboard --logdir=./logs/
\`\`\`

Open browser at `http://localhost:6006` to see:
- Reward curves over time
- Episode lengths
- Loss values
- Exploration rate (epsilon)

### Key Metrics

- **Episode Reward**: Total reward per episode (should increase)
- **Episode Length**: Steps per episode
- **Training Loss**: Should decrease over time
- **Epsilon**: Should decay from 1.0 to final value

## 🎯 Expected Performance

Training times and performance vary by game:

| Game | Easy to Learn | Typical Training Time |
|------|---------------|----------------------|
| Pong | ✓ | 1-2 hours |
| Breakout | ✓ | 2-4 hours |
| Space Invaders | Medium | 4-6 hours |
| Ms. Pacman | Hard | 8+ hours |

**Note**: Times are approximate on GPU. CPU training takes 3-5x longer.

## 🔍 Troubleshooting

### Agent not learning
- Increase `total_timesteps`
- Decrease `learning_rate`
- Increase `buffer_size`
- Check that `learning_starts` is appropriate

### Training unstable
- Decrease `learning_rate`
- Increase `batch_size`
- Increase `target_update_interval`

### Agent too random
- Check that epsilon is decaying properly
- Ensure using trained model (not during training)
- Verify `deterministic=True` in evaluation

## 🏆 Policy Comparison: MLP vs CNN

### CNN Policy (Recommended for Atari)
- **Architecture**: Convolutional layers + fully connected
- **Input**: Raw pixels (84x84x4 stacked frames)
- **Advantages**: Learns spatial features, better for visual games
- **Performance**: Excellent on Atari

### MLP Policy
- **Architecture**: Fully connected layers only
- **Input**: Flattened pixel values
- **Disadvantages**: Can't learn spatial relationships well
- **Performance**: Poor on Atari (use CNN instead)

## 📝 Hyperparameter Experiment Table

| Hyperparameter Set | Noted Behavior |
|-------------------|----------------|
| lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | Baseline configuration - stable learning |
| lr=5e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | Faster initial learning, some instability |
| lr=1e-4, gamma=0.95, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | More reactive to immediate rewards |
| lr=1e-4, gamma=0.99, batch=64, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | More stable gradients, slower convergence |
| lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.2 | Better exploration, useful for complex games |

## 👥 Group Collaboration

**Team Members**: John Ouma, Jeremiah Agbaje, Tanguy Kwizera

### Individual Contributions
- Document who worked on which parts
- Track code contributions
- Note any specific hyperparameter experiments each person ran

## 📚 Additional Resources

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Atari Environments](https://gymnasium.farama.org/environments/atari/)
- [DQN Paper](https://www.nature.com/articles/nature14236) - Original DeepMind paper
- [TensorBoard Tutorial](https://www.tensorflow.org/tensorboard)

## 🎓 Learning Objectives

After completing this assignment, you should understand:
1. How Q-learning works and why we use neural networks
2. The exploration-exploitation tradeoff
3. Why experience replay and target networks are important
4. How different hyperparameters affect learning
5. The difference between training (exploration) and evaluation (exploitation)

## ⚡ Quick Start

1. **Install dependencies**
2. **Run training**: `python train.py` (go get coffee ☕)
3. **Watch agent play**: `python play.py`
4. **View training graphs**: `tensorboard --logdir=./logs/`
5. **Experiment with hyperparameters** in the CONFIG dictionary
6. **Document your observations** in the hyperparameter table

Good luck with your DQN training! 🚀🎮
