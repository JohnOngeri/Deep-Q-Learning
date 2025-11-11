# Hyperparameter Experiment Template

Use this template to document each of your 10 experiments systematically.

## Experiment Template

### Experiment #X - [Descriptive Name]

**Date**: [YYYY-MM-DD]  
**Team Member**: [Your Name]  
**Atari Game**: [e.g., ALE/Breakout-v5]

#### Hyperparameters

\`\`\`python
learning_rate = 
gamma = 
batch_size = 
buffer_size = 
learning_starts = 
target_update_interval = 
exploration_fraction = 
exploration_initial_eps = 
exploration_final_eps = 
total_timesteps = 
\`\`\`

#### Hypothesis
*What do you expect to happen with these parameters and why?*

[Your hypothesis here]

#### Observed Behavior

**Training Metrics**:
- Final Average Reward: 
- Peak Reward Achieved: 
- Training Time: 
- Convergence Speed (episodes to stabilize): 

**Qualitative Observations**:
[Describe how the agent behaved during training and evaluation]

#### Analysis

**What Worked Well**:
- [Point 1]
- [Point 2]

**Issues Encountered**:
- [Point 1]
- [Point 2]

**Comparison to Baseline**:
[How did this configuration compare to the baseline experiment?]

#### Conclusion
[Overall assessment of this configuration]

---

## Tips for Systematic Experimentation

### Vary One Parameter at a Time (Initially)
1. **Experiment 1**: Baseline (all default values)
2. **Experiments 2-4**: Vary only learning rate
3. **Experiments 5-7**: Vary only gamma
4. **Experiments 8-10**: Combine best parameters or test batch size/exploration

### Important Notes to Track

**Learning Rate Experiments**:
- Too high → Unstable, divergent learning
- Too low → Very slow convergence
- Optimal range typically: 1e-5 to 5e-4 for Atari

**Gamma Experiments**:
- Low (0.90-0.95) → Short-sighted behavior
- High (0.99+) → Long-term planning, slower initial progress
- Game dependent: Simple games can use lower gamma

**Batch Size Experiments**:
- Small (16-32) → Noisy but fast
- Large (64-128) → Stable but slow
- Consider GPU memory constraints

**Exploration Experiments**:
- Fast decay → Quick exploitation, may miss optimal policy
- Slow decay → Better exploration, longer training time
- Complex games need longer exploration

### Recording Best Practices

1. **Save models** for each experiment: `models/exp_{number}_model.zip`
2. **Screenshot TensorBoard** graphs for each experiment
3. **Record videos** of agent playing after training
4. **Note computational resources**: GPU/CPU, training duration
5. **Compare side-by-side**: Keep a spreadsheet of all metrics

### Questions to Answer in Your Analysis

1. How did the agent's behavior evolve during training?
2. Did the agent learn the game mechanics? Which ones?
3. What was the exploration-exploitation pattern?
4. Were there any sudden jumps or drops in performance?
5. How consistent was performance across evaluation episodes?
6. What would you change for better performance?

---

## Example Experiment Documentation

### Experiment #1 - Baseline Configuration

**Date**: 2025-01-15  
**Team Member**: John Ouma  
**Atari Game**: ALE/Breakout-v5

#### Hyperparameters

\`\`\`python
learning_rate = 1e-4
gamma = 0.99
batch_size = 32
buffer_size = 100000
learning_starts = 50000
target_update_interval = 10000
exploration_fraction = 0.1
exploration_initial_eps = 1.0
exploration_final_eps = 0.01
total_timesteps = 1000000
\`\`\`

#### Hypothesis
This baseline configuration should provide stable learning with moderate convergence speed. The standard parameters are well-tested for Atari games.

#### Observed Behavior

**Training Metrics**:
- Final Average Reward: 45.3
- Peak Reward Achieved: 78.0
- Training Time: 3.5 hours
- Convergence Speed: ~400k timesteps

**Qualitative Observations**:
The agent successfully learned to move the paddle and hit the ball consistently. Initial exploration phase (first 100k steps) showed random behavior. After 400k steps, agent developed a strategy of positioning under the ball. Occasional failures when ball speed increased.

#### Analysis

**What Worked Well**:
- Stable training throughout, no divergence
- Agent learned basic game mechanics
- Smooth reward curve with steady improvement

**Issues Encountered**:
- Plateau around 500k steps, limited further improvement
- Agent sometimes fails to track fast-moving balls
- Conservative strategy, doesn't aim for specific bricks

**Comparison to Baseline**:
This IS the baseline - all other experiments compare to these results.

#### Conclusion
Solid baseline performance demonstrating stable DQN learning. Agent achieves competent but not expert level play. Good foundation for exploring hyperparameter variations to potentially improve performance.
