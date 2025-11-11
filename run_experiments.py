"""
Automated Experiment Runner

This script helps you run multiple experiments with different hyperparameter configurations.
Use this to systematically test all 10 required experiments.
"""

import subprocess
import time
from pathlib import Path

# Define 10 different hyperparameter configurations for each team member
EXPERIMENT_CONFIGS = [
    {
        'num': 1,
        'name': 'Baseline',
        'params': {
            'lr': 1e-4,
            'gamma': 0.99,
            'batch_size': 32,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.1
        },
        'description': 'Standard DQN hyperparameters as baseline'
    },
    {
        'num': 2,
        'name': 'High Learning Rate',
        'params': {
            'lr': 5e-4,
            'gamma': 0.99,
            'batch_size': 32,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.1
        },
        'description': 'Increased learning rate for faster learning'
    },
    {
        'num': 3,
        'name': 'Low Learning Rate',
        'params': {
            'lr': 5e-5,
            'gamma': 0.99,
            'batch_size': 32,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.1
        },
        'description': 'Decreased learning rate for more stable learning'
    },
    {
        'num': 4,
        'name': 'Low Discount Factor',
        'params': {
            'lr': 1e-4,
            'gamma': 0.95,
            'batch_size': 32,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.1
        },
        'description': 'Lower gamma - focus on immediate rewards'
    },
    {
        'num': 5,
        'name': 'High Discount Factor',
        'params': {
            'lr': 1e-4,
            'gamma': 0.995,
            'batch_size': 32,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.1
        },
        'description': 'Higher gamma - focus on long-term rewards'
    },
    {
        'num': 6,
        'name': 'Large Batch Size',
        'params': {
            'lr': 1e-4,
            'gamma': 0.99,
            'batch_size': 64,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.1
        },
        'description': 'Larger batch for more stable gradients'
    },
    {
        'num': 7,
        'name': 'Small Batch Size',
        'params': {
            'lr': 1e-4,
            'gamma': 0.99,
            'batch_size': 16,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.1
        },
        'description': 'Smaller batch for faster updates'
    },
    {
        'num': 8,
        'name': 'Extended Exploration',
        'params': {
            'lr': 1e-4,
            'gamma': 0.99,
            'batch_size': 32,
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.2
        },
        'description': 'More exploration throughout training'
    },
    {
        'num': 9,
        'name': 'Quick Exploitation',
        'params': {
            'lr': 1e-4,
            'gamma': 0.99,
            'batch_size': 32,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.05
        },
        'description': 'Rapid transition to exploitation'
    },
    {
        'num': 10,
        'name': 'Aggressive Learning',
        'params': {
            'lr': 1e-3,
            'gamma': 0.95,
            'batch_size': 64,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.05
        },
        'description': 'High LR, low gamma, large batch, fast exploitation'
    }
]


def run_experiment(member_name: str, config: dict, timesteps: int = 500_000, env: str = 'ALE/Breakout-v5'):
    """
    Run a single experiment with the given configuration.
    
    Args:
        member_name: Name of the team member
        config: Experiment configuration dictionary
        timesteps: Number of training timesteps
        env: Atari environment name
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT {config['num']}: {config['name']}")
    print(f"Team Member: {member_name}")
    print(f"{'='*80}")
    print(f"Description: {config['description']}")
    print(f"\nHyperparameters:")
    for key, value in config['params'].items():
        print(f"  {key}: {value}")
    print(f"{'='*80}\n")
    
    # Build command
    cmd = [
        'python', 'train.py',
        '--member-name', member_name,
        '--experiment-num', str(config['num']),
        '--env', env,
        '--timesteps', str(timesteps),
        '--lr', str(config['params']['lr']),
        '--gamma', str(config['params']['gamma']),
        '--batch-size', str(config['params']['batch_size']),
        '--epsilon-start', str(config['params']['epsilon_start']),
        '--epsilon-end', str(config['params']['epsilon_end']),
        '--epsilon-decay', str(config['params']['epsilon_decay'])
    ]
    
    # Run training
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n✓ Experiment {config['num']} completed in {elapsed/3600:.2f} hours")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment {config['num']} failed: {e}")
    except KeyboardInterrupt:
        print(f"\n⚠ Experiment {config['num']} interrupted by user")
        raise


def run_all_experiments(member_name: str, start_from: int = 1, env: str = 'ALE/Breakout-v5'):
    """
    Run all 10 experiments for a team member.
    
    Args:
        member_name: Name of the team member
        start_from: Which experiment number to start from (useful if resuming)
        env: Atari environment name
    """
    print(f"\n{'='*80}")
    print(f"RUNNING ALL EXPERIMENTS FOR: {member_name}")
    print(f"Environment: {env}")
    print(f"Starting from experiment #{start_from}")
    print(f"{'='*80}\n")
    
    configs_to_run = [c for c in EXPERIMENT_CONFIGS if c['num'] >= start_from]
    
    print(f"Will run {len(configs_to_run)} experiments:")
    for config in configs_to_run:
        print(f"  {config['num']}. {config['name']}")
    
    input("\nPress Enter to start, or Ctrl+C to cancel...")
    
    total_start = time.time()
    
    for config in configs_to_run:
        try:
            run_experiment(member_name, config, timesteps=500_000, env=env)
        except KeyboardInterrupt:
            print("\n⚠ Experiment run interrupted by user")
            break
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETE!")
    print(f"Total time: {total_elapsed/3600:.2f} hours")
    print(f"{'='*80}")
    print(f"\nResults saved to: experiments/experiment_results.csv")
    print(f"Don't forget to add your observations for each experiment!")


def print_experiment_summary():
    """Print a summary of all experiment configurations"""
    print(f"\n{'='*80}")
    print("EXPERIMENT CONFIGURATIONS SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'#':<4} {'Name':<25} {'LR':<10} {'Gamma':<8} {'Batch':<8} {'Eps_End':<10} {'Eps_Decay':<10}")
    print(f"{'-'*80}")
    
    for config in EXPERIMENT_CONFIGS:
        p = config['params']
        print(f"{config['num']:<4} {config['name']:<25} {p['lr']:<10.0e} {p['gamma']:<8.3f} "
              f"{p['batch_size']:<8} {p['epsilon_end']:<10.2f} {p['epsilon_decay']:<10.2f}")
        print(f"     → {config['description']}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import sys
    
    # Print summary
    print_experiment_summary()
    
    # Get team member name
    if len(sys.argv) > 1:
        member_name = sys.argv[1]
    else:
        print("Enter your name:")
        print("1. John Ouma")
        print("2. Jeremiah Agbaje")
        print("3. Tanguy Kwizera")
        print("Or type your custom name:")
        
        choice = input("\nYour choice: ").strip()
        
        if choice == '1':
            member_name = 'John Ouma'
        elif choice == '2':
            member_name = 'Jeremiah Agbaje'
        elif choice == '3':
            member_name = 'Tanguy Kwizera'
        else:
            member_name = choice
    
    print(f"\nSelected team member: {member_name}")
    
    # Ask for environment
    print("\nSelect Atari environment:")
    print("1. ALE/Breakout-v5 (recommended)")
    print("2. ALE/Pong-v5")
    print("3. ALE/SpaceInvaders-v5")
    print("4. Custom environment")
    
    env_choice = input("\nYour choice (default: 1): ").strip() or '1'
    
    env_map = {
        '1': 'ALE/Breakout-v5',
        '2': 'ALE/Pong-v5',
        '3': 'ALE/SpaceInvaders-v5'
    }
    
    if env_choice in env_map:
        env = env_map[env_choice]
    else:
        env = env_choice
    
    print(f"Selected environment: {env}")
    
    # Ask which experiments to run
    print("\nOptions:")
    print("1. Run all 10 experiments")
    print("2. Run a single experiment")
    print("3. Resume from a specific experiment")
    
    run_choice = input("\nYour choice: ").strip()
    
    if run_choice == '1':
        run_all_experiments(member_name, env=env)
    elif run_choice == '2':
        exp_num = int(input("Enter experiment number (1-10): "))
        config = next(c for c in EXPERIMENT_CONFIGS if c['num'] == exp_num)
        run_experiment(member_name, config, env=env)
    elif run_choice == '3':
        start_from = int(input("Resume from experiment number: "))
        run_all_experiments(member_name, start_from=start_from, env=env)
    else:
        print("Invalid choice")
