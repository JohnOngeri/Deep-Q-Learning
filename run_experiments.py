"""
Automated Experiment Runner

This script helps you run multiple experiments with different hyperparameter configurations.
Use this to systematically test all 10 required experiments.
"""

import subprocess
import time
from pathlib import Path

# ======================================================================
# ALL TEAM MEMBER EXPERIMENT CONFIGURATIONS
# ======================================================================

EXPERIMENT_CONFIGS = {
    "John Ouma": [
        {'num': 1, 'name': 'Baseline', 'params': {'lr': 1e-4, 'gamma': 0.99, 'batch_size': 32,
                                                  'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},
         'description': 'Baseline configuration for DQN.'},
        {'num': 2, 'name': 'Higher Learning Rate', 'params': {'lr': 5e-4, 'gamma': 0.99, 'batch_size': 32,
                                                              'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},
         'description': 'Faster learning rate to test aggressive updates.'},
        {'num': 3, 'name': 'Lower Learning Rate', 'params': {'lr': 5e-5, 'gamma': 0.99, 'batch_size': 32,
                                                             'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},
         'description': 'Smaller learning rate for more stable learning.'},
        {'num': 4, 'name': 'High Gamma', 'params': {'lr': 1e-4, 'gamma': 0.995, 'batch_size': 32,
                                                    'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},
         'description': 'Higher discount factor for long-term focus.'},
        {'num': 5, 'name': 'Low Gamma', 'params': {'lr': 1e-4, 'gamma': 0.95, 'batch_size': 32,
                                                   'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},
         'description': 'Lower discount factor to emphasize immediate rewards.'},
        {'num': 6, 'name': 'Larger Batch', 'params': {'lr': 1e-4, 'gamma': 0.99, 'batch_size': 64,
                                                      'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},
         'description': 'Larger batch size for smoother gradient updates.'},
        {'num': 7, 'name': 'Smaller Batch', 'params': {'lr': 1e-4, 'gamma': 0.99, 'batch_size': 16,
                                                       'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},
         'description': 'Smaller batch size for faster updates but higher variance.'},
        {'num': 8, 'name': 'Extended Exploration', 'params': {'lr': 1e-4, 'gamma': 0.99, 'batch_size': 32,
                                                              'epsilon_start': 1.0, 'epsilon_end': 0.05, 'epsilon_decay': 0.2},
         'description': 'Longer exploration period before exploiting.'},
        {'num': 9, 'name': 'Quick Exploitation', 'params': {'lr': 1e-4, 'gamma': 0.99, 'batch_size': 32,
                                                            'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.05},
         'description': 'Faster shift from exploration to exploitation.'},
        {'num': 10, 'name': 'Aggressive Learning', 'params': {'lr': 1e-3, 'gamma': 0.95, 'batch_size': 64,
                                                              'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.05},
         'description': 'High LR, low gamma, large batch, and rapid exploitation.'}
    ],

    "Jeremiah Agbaje": [
        {'num': 1, 'name': 'Low Gamma', 'params': {'lr': 1e-4, 'gamma': 0.90, 'batch_size': 32,
                                                   'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},
         'description': 'Focus on short-term rewards (low discount).'},
        {'num': 2, 'name': 'Very High Gamma', 'params': {'lr': 1e-4, 'gamma': 0.999, 'batch_size': 32,
                                                         'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},
         'description': 'Focus on very long-term rewards (almost no discount).'},
        {'num': 3, 'name': 'Moderate Gamma', 'params': {'lr': 1e-4, 'gamma': 0.97, 'batch_size': 32,
                                                        'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},
         'description': 'Balanced gamma between short and long term.'},
        {'num': 4, 'name': 'Lower Learning Rate', 'params': {'lr': 1e-5, 'gamma': 0.97, 'batch_size': 32,
                                                             'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},
         'description': 'Slower learning for more stability.'},
        {'num': 5, 'name': 'Higher Learning Rate', 'params': {'lr': 7e-4, 'gamma': 0.97, 'batch_size': 32,
                                                              'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},
         'description': 'Aggressive updates for faster convergence.'},
        {'num': 6, 'name': 'Wide Exploration', 'params': {'lr': 1e-4, 'gamma': 0.97, 'batch_size': 32,
                                                          'epsilon_start': 1.0, 'epsilon_end': 0.1, 'epsilon_decay': 0.3},
         'description': 'Sustain exploration longer during training.'},
        {'num': 7, 'name': 'Early Exploitation', 'params': {'lr': 1e-4, 'gamma': 0.97, 'batch_size': 32,
                                                            'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.05},
         'description': 'Shift quickly from exploration to exploitation.'},
        {'num': 8, 'name': 'Small Batch High Gamma', 'params': {'lr': 1e-4, 'gamma': 0.995, 'batch_size': 16,
                                                                'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},
         'description': 'Small batch with strong long-term focus.'},
        {'num': 9, 'name': 'Large Batch Low Gamma', 'params': {'lr': 1e-4, 'gamma': 0.92, 'batch_size': 64,
                                                               'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},
         'description': 'Large batch, focuses on shorter horizons.'},
        {'num': 10, 'name': 'Balanced Setup', 'params': {'lr': 3e-4, 'gamma': 0.98, 'batch_size': 32,
                                                         'epsilon_start': 1.0, 'epsilon_end': 0.02, 'epsilon_decay': 0.15},
         'description': 'Balanced mix of all parameters for control comparison.'}
    ],

    "Tanguy Kwizera": [
        {'num': 1, 'name': 'Large Batch', 'params': {'lr': 1e-4, 'gamma': 0.99, 'batch_size': 128,
                                                     'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},
         'description': 'Test stability with very large batch size.'},
        {'num': 2, 'name': 'Small Batch', 'params': {'lr': 1e-4, 'gamma': 0.99, 'batch_size': 8,
                                                     'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},
         'description': 'Very small batch size for fast updates.'},
        {'num': 3, 'name': 'Moderate Batch High LR', 'params': {'lr': 5e-4, 'gamma': 0.99, 'batch_size': 32,
                                                                'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},
         'description': 'Balance between learning rate and batch size.'},
        {'num': 4, 'name': 'Extended Exploration', 'params': {'lr': 1e-4, 'gamma': 0.99, 'batch_size': 32,
                                                              'epsilon_start': 1.0, 'epsilon_end': 0.1, 'epsilon_decay': 0.25},
         'description': 'Explore for much longer before exploiting.'},
        {'num': 5, 'name': 'Quick Exploitation', 'params': {'lr': 1e-4, 'gamma': 0.99, 'batch_size': 32,
                                                            'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.05},
         'description': 'Transition to exploitation early in training.'},
        {'num': 6, 'name': 'High LR Low Gamma', 'params': {'lr': 1e-3, 'gamma': 0.93, 'batch_size': 64,
                                                           'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},
         'description': 'Faster updates but shorter-term learning focus.'},
        {'num': 7, 'name': 'Low LR High Gamma', 'params': {'lr': 5e-5, 'gamma': 0.995, 'batch_size': 32,
                                                           'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},
         'description': 'Slow learning, emphasizes long-term planning.'},
        {'num': 8, 'name': 'Aggressive Exploration', 'params': {'lr': 1e-4, 'gamma': 0.99, 'batch_size': 32,
                                                                'epsilon_start': 1.0, 'epsilon_end': 0.2, 'epsilon_decay': 0.3},
         'description': 'Keeps exploration high for much longer period.'},
        {'num': 9, 'name': 'Small Batch High LR', 'params': {'lr': 7e-4, 'gamma': 0.97, 'batch_size': 16,
                                                             'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},
         'description': 'Fast updates with high variance gradients.'},
        {'num': 10, 'name': 'Balanced Learning', 'params': {'lr': 3e-4, 'gamma': 0.98, 'batch_size': 32,
                                                            'epsilon_start': 1.0, 'epsilon_end': 0.02, 'epsilon_decay': 0.15},
         'description': 'Balanced parameters for comparison baseline.'}
    ]
}




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
    
    # Build command for train.py
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
    
    # Run the experiment
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
    Run all experiments for a specific team member.
    
    Args:
        member_name: Name of the team member
        start_from: Experiment number to resume from
        env: Atari environment name
    """
    if member_name not in EXPERIMENT_CONFIGS:
        print(f"\n⚠ No experiment set found for {member_name}. Please check the name.")
        return

    configs = EXPERIMENT_CONFIGS[member_name]
    configs_to_run = [c for c in configs if c['num'] >= start_from]

    print(f"\n{'='*80}")
    print(f"RUNNING ALL EXPERIMENTS FOR: {member_name}")
    print(f"Environment: {env}")
    print(f"Starting from experiment #{start_from}")
    print(f"{'='*80}\n")

    print(f"Will run {len(configs_to_run)} experiments:")
    for config in configs_to_run:
        print(f"  {config['num']}. {config['name']}")

    input("\nPress Enter to start, or Ctrl+C to cancel...")

    total_start = time.time()

    for config in configs_to_run:
        try:
            run_experiment(member_name, config, timesteps=500_000, env=env)
        except KeyboardInterrupt:
            print("\n⚠ Experiment run interrupted by user.")
            break

    total_elapsed = time.time() - total_start
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Total time: {total_elapsed/3600:.2f} hours")
    print(f"{'='*80}")
    print("\nResults saved to: experiments/experiment_results.csv")
    print("Don't forget to add your observations for each experiment!")


def print_experiment_summary(member_name: str):
    """
    Print a summary of all experiment configurations for the selected team member.
    """
    if member_name not in EXPERIMENT_CONFIGS:
        print(f"\n⚠ No experiment configuration found for {member_name}.")
        return

    configs = EXPERIMENT_CONFIGS[member_name]

    print(f"\n{'='*80}")
    print(f"EXPERIMENT CONFIGURATIONS SUMMARY FOR {member_name.upper()}")
    print(f"{'='*80}\n")
    
    print(f"{'#':<4} {'Name':<25} {'LR':<10} {'Gamma':<8} {'Batch':<8} {'Eps_End':<10} {'Eps_Decay':<10}")
    print(f"{'-'*80}")

    for config in configs:
        p = config['params']
        print(f"{config['num']:<4} {config['name']:<25} {p['lr']:<10.0e} {p['gamma']:<8.3f} "
              f"{p['batch_size']:<8} {p['epsilon_end']:<10.2f} {p['epsilon_decay']:<10.2f}")
        print(f"     → {config['description']}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import sys

    # Select member name
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

    # Show the experiment summary for that member
    print_experiment_summary(member_name)

    # Select Atari environment
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
    env = env_map.get(env_choice, env_choice)
    print(f"Selected environment: {env}")

    # Choose run mode
    print("\nOptions:")
    print("1. Run all experiments")
    print("2. Run a single experiment")
    print("3. Resume from a specific experiment")

    run_choice = input("\nYour choice: ").strip()

    if run_choice == '1':
        run_all_experiments(member_name, env=env)
    elif run_choice == '2':
        exp_num = int(input("Enter experiment number (1-10): "))
        configs = EXPERIMENT_CONFIGS.get(member_name, [])
        config = next((c for c in configs if c['num'] == exp_num), None)
        if config:
            run_experiment(member_name, config, env=env)
        else:
            print(f"No experiment #{exp_num} found for {member_name}.")
    elif run_choice == '3':
        start_from = int(input("Resume from experiment number: "))
        run_all_experiments(member_name, start_from=start_from, env=env)
    else:
        print("Invalid choice.")
