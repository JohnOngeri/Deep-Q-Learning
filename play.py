"""
DQN Agent Evaluation Script
This script loads a trained DQN model and plays the Atari game with visualization.
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import time
import json
import os


class GreedyQPolicy:
    """
    Greedy Q-Policy: Always selects the action with the highest Q-value.
    This is used for evaluation to maximize performance (no exploration).
    """
    def __init__(self, model):
        self.model = model
    
    def predict(self, observation, deterministic=True):
        """
        Select action with highest Q-value.
        
        Args:
            observation: Current state
            deterministic: Always True for greedy policy
        
        Returns:
            action: Best action according to Q-values
        """
        return self.model.predict(observation, deterministic=True)


def load_model_and_config(model_path):
    """
    Load the trained DQN model and its configuration.
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        model: Loaded DQN model
        config: Configuration dictionary
    """
    print("="*60)
    print("LOADING TRAINED MODEL")
    print("="*60)
    
    # Load the model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    model = DQN.load(model_path)
    print(f"✓ Model loaded from: {model_path}")
    
    # Try to load configuration
    config_path = model_path.replace('.zip', '_config.json')
    config = {}
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Configuration loaded from: {config_path}")
        print(f"\nModel Details:")
        print(f"  Environment: {config.get('env_name', 'Unknown')}")
        print(f"  Policy: {config.get('policy_type', 'Unknown')}")
        print(f"  Training Steps: {config.get('total_timesteps', 'Unknown'):,}")
    else:
        print("⚠ Configuration file not found. Using default environment.")
    
    print("="*60)
    
    return model, config


def create_evaluation_environment(env_name, render_mode='human'):
    """
    Create an environment for evaluation with rendering enabled.
    
    Args:
        env_name: Name of the Atari environment
        render_mode: Rendering mode ('human' for GUI display)
    
    Returns:
        Wrapped environment
    """
    # Create single environment with rendering
    def make_env():
        env = gym.make(env_name, render_mode=render_mode)
        return env
    
    # Wrap in DummyVecEnv for compatibility
    env = DummyVecEnv([make_env])
    
    # Stack frames to match training
    env = VecFrameStack(env, n_stack=4)
    
    return env


def play_episodes(model, env, n_episodes=5, render_delay=0.01):
    """
    Play multiple episodes with the trained agent and display performance.
    
    Args:
        model: Trained DQN model
        env: Evaluation environment
        n_episodes: Number of episodes to play
        render_delay: Delay between frames (seconds) for better visualization
    
    Returns:
        episode_rewards: List of total rewards for each episode
        episode_lengths: List of episode lengths
    """
    print("\n" + "="*60)
    print("PLAYING WITH TRAINED AGENT (GREEDY POLICY)")
    print("="*60)
    print(f"Episodes to play: {n_episodes}")
    print("Policy: Greedy (always selects best action)")
    print("="*60)
    
    # Use GreedyQPolicy for evaluation
    greedy_policy = GreedyQPolicy(model)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        print(f"\n--- Episode {episode + 1}/{n_episodes} ---")
        
        while not done:
            # Get action from greedy policy (deterministic, no exploration)
            action, _ = greedy_policy.predict(obs, deterministic=True)
            
            # Take action in environment
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            episode_length += 1
            
            # Render the game
            env.render()
            
            # Small delay for better visualization
            time.sleep(render_delay)
            
            # Check if episode ended
            if done[0]:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1} finished:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Episode Length: {episode_length} steps")
    
    return episode_rewards, episode_lengths


def display_statistics(episode_rewards, episode_lengths):
    """
    Display performance statistics across all episodes.
    
    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
    """
    print("\n" + "="*60)
    print("EVALUATION STATISTICS")
    print("="*60)
    
    print(f"\nNumber of Episodes: {len(episode_rewards)}")
    print(f"\nReward Statistics:")
    print(f"  Mean Reward: {np.mean(episode_rewards):.2f}")
    print(f"  Std Reward: {np.std(episode_rewards):.2f}")
    print(f"  Min Reward: {np.min(episode_rewards):.2f}")
    print(f"  Max Reward: {np.max(episode_rewards):.2f}")
    
    print(f"\nEpisode Length Statistics:")
    print(f"  Mean Length: {np.mean(episode_lengths):.2f} steps")
    print(f"  Std Length: {np.std(episode_lengths):.2f} steps")
    print(f"  Min Length: {np.min(episode_lengths)} steps")
    print(f"  Max Length: {np.max(episode_lengths)} steps")
    
    print("\n" + "="*60)


def evaluate_model(model_path, n_episodes=5, render_delay=0.01):
    """
    Complete evaluation pipeline: load model, play episodes, show statistics.
    
    Args:
        model_path: Path to the trained model
        n_episodes: Number of episodes to evaluate
        render_delay: Delay between frames for visualization
    """
    # Load model and configuration
    model, config = load_model_and_config(model_path)
    
    # Get environment name
    env_name = config.get('env_name', 'ALE/Breakout-v5')
    
    # Create evaluation environment with rendering
    env = create_evaluation_environment(env_name, render_mode='human')
    
    try:
        # Play episodes
        episode_rewards, episode_lengths = play_episodes(
            model, env, n_episodes, render_delay
        )
        
        # Display statistics
        display_statistics(episode_rewards, episode_lengths)
        
    finally:
        # Clean up
        env.close()
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = './models/dqn_model.zip'  # Path to your trained model
    N_EPISODES = 5  # Number of episodes to play
    RENDER_DELAY = 0.01  # Delay between frames (adjust for speed)
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     DQN AGENT EVALUATION - ATARI GAME PLAYER            ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("\nPlease train a model first by running train.py")
        
        # Try to find alternative models
        models_dir = './models/'
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
            if model_files:
                print(f"\nFound these models in {models_dir}:")
                for i, model_file in enumerate(model_files, 1):
                    print(f"  {i}. {model_file}")
                print("\nUpdate MODEL_PATH in play.py to use one of these models.")
    else:
        # Run evaluation
        evaluate_model(MODEL_PATH, n_episodes=N_EPISODES, render_delay=RENDER_DELAY)
        
        print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                    TIPS FOR EVALUATION                   ║
    ╠══════════════════════════════════════════════════════════╣
    ║ • Greedy Policy: Agent always picks best action         ║
    ║ • No exploration during evaluation                       ║
    ║ • Adjust RENDER_DELAY to control playback speed         ║
    ║ • Try evaluating the best_model.zip for best results    ║
    ╚══════════════════════════════════════════════════════════╝
        """)
