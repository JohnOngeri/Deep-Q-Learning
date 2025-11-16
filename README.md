# Deep Q-Network (DQN) for Atari Games

This project implements a Deep Q-Network agent using Stable Baselines3 and Gymnasium to play Atari games.


<img width="1359" height="723" alt="image" src="https://github.com/user-attachments/assets/7b15004e-a07f-4664-879d-9f2d3ba642f7" />


```bash
Deep-Q-Learning/
│
├── .gitignore // Ignore cache, model, and temp files
├── README.md // Project documentation and results
├── package.json // (Optional) For npm-related configs if used
├── play.py // Script to load and evaluate trained agent
├── requirements.txt // Dependencies for Python environment
├── train.py // Script to train the DQN agent
│
└── dqn_model.zip // (Will appear after training - saved model)
```


## 🚀 Installation

Install the required dependencies:

```bash
pip install gymnasium[atari]
pip install stable-baselines3[extra]
pip install ale-py
pip install tensorboard
pip install matplotlib

```

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

## 📝 Hyperparameter Experiment Documentation

**Instructions**: Each group member must conduct 10 experiments with different hyperparameter combinations. Document your observations below.
## 🎯 Experiment Design Summary

Each group member conducted **10 independent Deep Q-Network (DQN) experiments** using the Atari `ALE/Breakout-v5` environment (with optional variations such as Pong and SpaceInvaders).  
The objective was to systematically study how different hyperparameter combinations affect the performance, stability, and learning efficiency of the agent.

Each experiment varied one or more of the following key hyperparameters:
- **Learning Rate (lr):** Controls the step size for gradient updates.  
- **Discount Factor (γ / gamma):** Determines how much future rewards influence current decisions.  
- **Batch Size:** Affects training stability and computational efficiency.  
- **Exploration Parameters (ε_start, ε_end, ε_decay):** Define the ε-greedy exploration schedule — how long the agent explores before exploiting learned behavior.

Each team member explored a distinct range of configurations:
- **John Ouma:** Focused on balancing learning rate and exploration strategies.
- **Jeremiah Agbaje:** Investigated effects of varying gamma (reward discounting) and learning rates.
- **Tanguy Kwizera:** Examined batch size effects and interaction between exploration and learning rate.

All experiments automatically logged training results, episode rewards, and configurations into  
📁 `experiments/experiment_results.csv` and corresponding `.json` files for analysis.

---


---

### 🧑‍💻 John Ouma - Experiments

| Experiment # | Hyperparameter Set | Description |
|--------------|-------------------|--------------|
| 1 | lr=1e-4, γ=0.99, batch=32, ε_start=1.0, ε_end=0.01, decay=0.1 | Baseline configuration for DQN. |
| 2 | lr=5e-4, γ=0.99, batch=32, ε_start=1.0, ε_end=0.01, decay=0.1 | Faster learning rate to test aggressive updates. |
| 3 | lr=5e-5, γ=0.99, batch=32, ε_start=1.0, ε_end=0.01, decay=0.1 | Smaller learning rate for more stable learning. |
| 4 | lr=1e-4, γ=0.995, batch=32, ε_start=1.0, ε_end=0.01, decay=0.1 | Higher discount factor for long-term focus. |
| 5 | lr=1e-4, γ=0.95, batch=32, ε_start=1.0, ε_end=0.01, decay=0.1 | Lower discount factor to emphasize immediate rewards. |
| 6 | lr=1e-4, γ=0.99, batch=64, ε_start=1.0, ε_end=0.01, decay=0.1 | Larger batch size for smoother gradient updates. |
| 7 | lr=1e-4, γ=0.99, batch=16, ε_start=1.0, ε_end=0.01, decay=0.1 | Smaller batch size for faster updates but higher variance. |
| 8 | lr=1e-4, γ=0.99, batch=32, ε_start=1.0, ε_end=0.05, decay=0.2 | Longer exploration period before exploiting. |
| 9 | lr=1e-4, γ=0.99, batch=32, ε_start=1.0, ε_end=0.01, decay=0.05 | Faster shift from exploration to exploitation. |
| 10 | lr=1e-3, γ=0.95, batch=64, ε_start=1.0, ε_end=0.01, decay=0.05 | High LR, low gamma, large batch, and rapid exploitation. |

---

### 🧑‍💻 Jeremiah Agbaje - Experiments

| Experiment # | Hyperparameter Set | Description |
|--------------|-------------------|--------------|
| 1 | lr=1e-4, γ=0.90, batch=32, ε_start=1.0, ε_end=0.01, decay=0.1 | Focus on short-term rewards (low discount). |
| 2 | lr=1e-4, γ=0.999, batch=32, ε_start=1.0, ε_end=0.01, decay=0.1 | Focus on very long-term rewards (almost no discount). |
| 3 | lr=1e-4, γ=0.97, batch=32, ε_start=1.0, ε_end=0.01, decay=0.1 | Balanced gamma between short and long term. |
| 4 | lr=1e-5, γ=0.97, batch=32, ε_start=1.0, ε_end=0.01, decay=0.1 | Slower learning for more stability. |
| 5 | lr=7e-4, γ=0.97, batch=32, ε_start=1.0, ε_end=0.01, decay=0.1 | Aggressive updates for faster convergence. |
| 6 | lr=1e-4, γ=0.97, batch=32, ε_start=1.0, ε_end=0.1, decay=0.3 | Sustain exploration longer during training. |
| 7 | lr=1e-4, γ=0.97, batch=32, ε_start=1.0, ε_end=0.01, decay=0.05 | Shift quickly from exploration to exploitation. |
| 8 | lr=1e-4, γ=0.995, batch=16, ε_start=1.0, ε_end=0.01, decay=0.1 | Small batch with strong long-term focus. |
| 9 | lr=1e-4, γ=0.92, batch=64, ε_start=1.0, ε_end=0.01, decay=0.1 | Large batch, focuses on shorter horizons. |
| 10 | lr=3e-4, γ=0.98, batch=32, ε_start=1.0, ε_end=0.02, decay=0.15 | Balanced mix of all parameters for control comparison. |

---

### 🧑‍💻 Tanguy Kwizera - Experiments

| Experiment # | Hyperparameter Set | Description |
|--------------|-------------------|--------------|
| 1 | lr=1e-4, γ=0.99, batch=128, ε_start=1.0, ε_end=0.01, decay=0.1 | Test stability with very large batch size. |
| 2 | lr=1e-4, γ=0.99, batch=8, ε_start=1.0, ε_end=0.01, decay=0.1 | Very small batch size for fast updates. |
| 3 | lr=5e-4, γ=0.99, batch=32, ε_start=1.0, ε_end=0.01, decay=0.1 | Balance between learning rate and batch size. |
| 4 | lr=1e-4, γ=0.99, batch=32, ε_start=1.0, ε_end=0.1, decay=0.25 | Explore for much longer before exploiting. |
| 5 | lr=1e-4, γ=0.99, batch=32, ε_start=1.0, ε_end=0.01, decay=0.05 | Transition to exploitation early in training. |
| 6 | lr=1e-3, γ=0.93, batch=64, ε_start=1.0, ε_end=0.01, decay=0.1 | Faster updates but shorter-term learning focus. |
| 7 | lr=5e-5, γ=0.995, batch=32, ε_start=1.0, ε_end=0.01, decay=0.1 | Slow learning, emphasizes long-term planning. |
| 8 | lr=1e-4, γ=0.99, batch=32, ε_start=1.0, ε_end=0.2, decay=0.3 | Keeps exploration high for much longer period. |
| 9 | lr=7e-4, γ=0.97, batch=16, ε_start=1.0, ε_end=0.01, decay=0.1 | Fast updates with high variance gradients. |
| 10 | lr=3e-4, γ=0.98, batch=32, ε_start=1.0, ε_end=0.02, decay=0.15 | Balanced parameters for comparison baseline. |


## 📊 Experiment Analysis Guidelines

After completing your experiments, analyze:

1. **Learning Rate Impact**: How did different learning rates affect convergence speed and stability?
2. **Gamma Effects**: How did the discount factor influence the agent's strategy (short-term vs long-term)?
3. **Batch Size Trade-offs**: What was the relationship between batch size, stability, and training speed?
4. **Exploration Strategy**: How did epsilon decay parameters affect the agent's ability to discover optimal policies?
5. **Best Configuration**: Which hyperparameter combination worked best for your chosen Atari game and why?

## 👥 Group Collaboration

**Team Members**: John Ouma, Jeremiah Agbaje, Tanguy Kwizera

### Individual Contribution Template

**John Ouma**:
- Responsibilities: [e.g., Initial setup, training script development, experiments 1-10]
- Key Contributions: [Specific code sections, documentation, testing]
- Experiments Conducted: 10 hyperparameter configurations documented above

**Jeremiah Agbaje**:
- Responsibilities: [e.g., Evaluation script, visualization, experiments 1-10]
- Key Contributions: [Specific code sections, documentation, testing]
- Experiments Conducted: 10 hyperparameter configurations documented above

**Tanguy Kwizera**:
- Responsibilities: [e.g., Documentation, analysis, experiments 1-10]
- Key Contributions: [Specific code sections, documentation, testing]
- Experiments Conducted: 10 hyperparameter configurations documented above

### Collaboration Notes
- **Meetings**: [Document meeting dates and discussions]
- **Division of Work**: [How tasks were divided]
- **Challenges**: [Any difficulties encountered and how they were resolved]
- **Group Insights**: [Collective learnings from the project]

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

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run training**: `python train.py` (go get coffee ☕)
3. **Watch agent play**: `python play.py`
4. **View training graphs**: `tensorboard --logdir=./logs/`
5. **Experiment with hyperparameters** in the CONFIG dictionary
6. **Document your observations** in the hyperparameter table above
7. **Complete all 10 experiments per team member** (30 total)

Good luck with your DQN training! 🚀🎮
