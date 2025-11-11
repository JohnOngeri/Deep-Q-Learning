"""
Deep Q-Network (DQN) Training Script for Atari Games
This script trains a DQN agent using Stable Baselines3 on a selected Atari environment.
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# Configuration
CONFIG = {
    # Environment settings
    'env_name': 'ALE/Breakout-v5',  # Change to any Atari environment you prefer
    # Other popular options: 'ALE/Pong-v5', 'ALE/SpaceInvaders-v5', 'ALE/MsPacman-v5'
    
    # Training settings
    'total_timesteps': 1_000_000,  # Total training steps
    'n_envs': 4,  # Number of parallel environments
    'frame_stack': 4,  # Number of frames to stack
    
    # Model settings
    'policy_type': 'CnnPolicy',  # 'CnnPolicy' or 'MlpPolicy'
    
    # DQN Hyperparameters - Experiment with these!
    'learning_rate': 1e-4,
    'gamma': 0.99,  # Discount factor
    'batch_size': 32,
    'buffer_size': 100_000,
    'learning_starts': 50_000,
    'target_update_interval': 1000,
    'train_freq': 4,
    'gradient_steps': 1,
    'exploration_fraction': 0.1,  # Fraction of training for epsilon decay
    'exploration_initial_eps': 1.0,  # epsilon_start
    'exploration_final_eps': 0.01,  # epsilon_end
    
    # Logging
    'log_dir': './logs/',
    'model_save_path': './models/dqn_model.zip',
    'best_model_path': './models/best_dqn_model.zip',
}


def create_training_environment(config):
    """
    Create and wrap the Atari environment with appropriate wrappers.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Wrapped vectorized environment
    """
    # Create vectorized Atari environment with standard wrappers
    env = make_atari_env(
        config['env_name'],
        n_envs=config['n_envs'],
        seed=0
    )
    
    # Stack frames for temporal information
    env = VecFrameStack(env, n_stack=config['frame_stack'])
    
    return env


def setup_callbacks(config):
    """
    Setup training callbacks for logging and model checkpointing.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        CallbackList with evaluation and checkpoint callbacks
    """
    # Create log directory
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
    
    # Create evaluation environment
    eval_env = make_atari_env(config['env_name'], n_envs=1, seed=42)
    eval_env = VecFrameStack(eval_env, n_stack=config['frame_stack'])
    
    # Evaluation callback - saves best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(config['best_model_path']),
        log_path=config['log_dir'],
        eval_freq=10_000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Checkpoint callback - saves model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join(config['log_dir'], 'checkpoints/'),
        name_prefix='dqn_checkpoint'
    )
    
    return CallbackList([eval_callback, checkpoint_callback])


def train_dqn_agent(config):
    """
    Train a DQN agent with the specified configuration.
    
    Args:
        config: Configuration dictionary containing hyperparameters
    
    Returns:
        Trained DQN model
    """
    print("="*60)
    print("TRAINING DQN AGENT")
    print("="*60)
    print(f"Environment: {config['env_name']}")
    print(f"Policy: {config['policy_type']}")
    print(f"Total Timesteps: {config['total_timesteps']:,}")
    print("\nHyperparameters:")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Gamma (γ): {config['gamma']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Exploration: {config['exploration_initial_eps']} → {config['exploration_final_eps']}")
    print(f"  Exploration Fraction: {config['exploration_fraction']}")
    print("="*60)
    
    # Create environment
    env = create_training_environment(config)
    
    # Setup callbacks
    callbacks = setup_callbacks(config)
    
    # Create DQN model
    model = DQN(
        policy=config['policy_type'],
        env=env,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        batch_size=config['batch_size'],
        buffer_size=config['buffer_size'],
        learning_starts=config['learning_starts'],
        target_update_interval=config['target_update_interval'],
        train_freq=config['train_freq'],
        gradient_steps=config['gradient_steps'],
        exploration_fraction=config['exploration_fraction'],
        exploration_initial_eps=config['exploration_initial_eps'],
        exploration_final_eps=config['exploration_final_eps'],
        verbose=1,
        tensorboard_log=config['log_dir']
    )
    
    # Train the model
    print("\nStarting training...")
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    model.save(config['model_save_path'])
    print(f"\nModel saved to: {config['model_save_path']}")
    
    # Save configuration
    config_save_path = config['model_save_path'].replace('.zip', '_config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to: {config_save_path}")
    
    return model


def compare_policies(env_name, total_timesteps=500_000):
    """
    Compare MLP and CNN policies for the same environment.
    
    Args:
        env_name: Name of the Atari environment
        total_timesteps: Number of training steps
    """
    print("\n" + "="*60)
    print("COMPARING POLICIES: MLP vs CNN")
    print("="*60)
    
    policies = ['MlpPolicy', 'CnnPolicy']
    results = {}
    
    for policy in policies:
        print(f"\n{'='*60}")
        print(f"Training with {policy}")
        print(f"{'='*60}")
        
        config_copy = CONFIG.copy()
        config_copy['policy_type'] = policy
        config_copy['total_timesteps'] = total_timesteps
        config_copy['model_save_path'] = f'./models/dqn_{policy.lower()}.zip'
        
        try:
            model = train_dqn_agent(config_copy)
            results[policy] = "Training completed successfully"
        except Exception as e:
            results[policy] = f"Error: {str(e)}"
    
    print("\n" + "="*60)
    print("POLICY COMPARISON RESULTS")
    print("="*60)
    for policy, result in results.items():
        print(f"{policy}: {result}")
    print("\nNote: For Atari games, CNNPolicy typically performs much better")
    print("as it can process visual information from game frames.")


def hyperparameter_experiments():
    """
    Run multiple training sessions with different hyperparameter configurations.
    This helps understand how each parameter affects performance.
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING EXPERIMENTS")
    print("="*60)
    
    # Define different hyperparameter sets to test
    experiments = [
        {
            'name': 'Baseline',
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'batch_size': 32,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.01,
            'exploration_fraction': 0.1,
            'description': 'Standard DQN hyperparameters'
        },
        {
            'name': 'High Learning Rate',
            'learning_rate': 5e-4,
            'gamma': 0.99,
            'batch_size': 32,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.01,
            'exploration_fraction': 0.1,
            'description': 'Faster learning but potentially less stable'
        },
        {
            'name': 'Low Discount Factor',
            'learning_rate': 1e-4,
            'gamma': 0.95,
            'batch_size': 32,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.01,
            'exploration_fraction': 0.1,
            'description': 'More focus on immediate rewards'
        },
        {
            'name': 'Large Batch Size',
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'batch_size': 64,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.01,
            'exploration_fraction': 0.1,
            'description': 'More stable gradients but slower updates'
        },
        {
            'name': 'Extended Exploration',
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'batch_size': 32,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.05,
            'exploration_fraction': 0.2,
            'description': 'More exploration throughout training'
        },
    ]
    
    # Create results table
    print("\nHYPERPARAMETER EXPERIMENT CONFIGURATIONS:")
    print("-" * 100)
    print(f"{'Set':<20} {'LR':<10} {'Gamma':<8} {'Batch':<8} {'Eps_Start':<12} {'Eps_End':<10} {'Eps_Frac':<10}")
    print("-" * 100)
    
    for exp in experiments:
        print(f"{exp['name']:<20} {exp['learning_rate']:<10.6f} {exp['gamma']:<8.2f} "
              f"{exp['batch_size']:<8} {exp['exploration_initial_eps']:<12.2f} "
              f"{exp['exploration_final_eps']:<10.2f} {exp['exploration_fraction']:<10.2f}")
        print(f"  → {exp['description']}")
    
    print("\n" + "="*60)
    print("To run these experiments, uncomment the training loop below")
    print("="*60)
    
    # Uncomment to actually run the experiments (Warning: Takes a long time!)
    """
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Running Experiment: {exp['name']}")
        print(f"{'='*60}")
        
        config_copy = CONFIG.copy()
        config_copy.update(exp)
        config_copy['model_save_path'] = f"./models/dqn_{exp['name'].lower().replace(' ', '_')}.zip"
        config_copy['total_timesteps'] = 500_000  # Shorter training for experiments
        
        train_dqn_agent(config_copy)
    """


if __name__ == "__main__":
    # Main training with current configuration
    model = train_dqn_agent(CONFIG)
    
    # Display hyperparameter documentation table
    print("\n" + "="*80)
    print("HYPERPARAMETER DOCUMENTATION TABLE")
    print("="*80)
    print("\nRecord your observations for each hyperparameter configuration:\n")
    print("-" * 120)
    print(f"{'Hyperparameter Set':<100} {'Noted Behavior':<20}")
    print("-" * 120)
    
    hyperparameter_sets = [
        "lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1",
        "lr=5e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1",
        "lr=1e-4, gamma=0.95, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1",
        "lr=1e-4, gamma=0.99, batch=64, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1",
        "lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.2",
    ]
    
    for hp_set in hyperparameter_sets:
        print(f"{hp_set:<100} {'[Record observations]':<20}")
    
    print("-" * 120)
    print("\nTo view training progress, run: tensorboard --logdir=./logs/")
    
    # Optional: Run hyperparameter experiments
    # Uncomment the line below to see all experiment configurations
    # hyperparameter_experiments()
    
    # Optional: Compare policies
    # Uncomment the line below to compare MLP vs CNN policies
    # compare_policies(CONFIG['env_name'], total_timesteps=500_000)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Final model saved at: {CONFIG['model_save_path']}")
    print(f"Best model saved at: {CONFIG['best_model_path']}")
    print("\nNext steps:")
    print("1. Run play.py to evaluate the trained agent")
    print("2. View training logs with TensorBoard")
    print("3. Experiment with different hyperparameters")
    print("="*80)
