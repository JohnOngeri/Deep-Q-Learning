"""
View and Analyze Experiment Results

This script loads experiment results from CSV and generates comparison reports.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def load_results(csv_file='experiments/experiment_results.csv'):
    """Load experiment results from CSV"""
    if not Path(csv_file).exists():
        print(f"No results file found at {csv_file}")
        print("Run some experiments first!")
        return None
    
    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df)} experiments")
    return df


def print_summary_table(df, member_name=None):
    """Print a formatted summary table"""
    if member_name:
        df = df[df['member_name'] == member_name]
        print(f"\n{'='*100}")
        print(f"RESULTS FOR: {member_name}")
        print(f"{'='*100}\n")
    else:
        print(f"\n{'='*100}")
        print("ALL EXPERIMENT RESULTS")
        print(f"{'='*100}\n")
    
    # Sort by average reward
    df_sorted = df.sort_values('avg_reward', ascending=False)
    
    # Print table
    print(f"{'Exp#':<6} {'Member':<20} {'Avg Reward':<12} {'Peak Reward':<12} {'LR':<10} {'Gamma':<8} {'Batch':<8}")
    print(f"{'-'*100}")
    
    for _, row in df_sorted.iterrows():
        print(f"{row['experiment_num']:<6} {row['member_name']:<20} {row['avg_reward']:<12.2f} "
              f"{row['peak_reward']:<12.2f} {row['learning_rate']:<10.2e} "
              f"{row['gamma']:<8.3f} {row['batch_size']:<8}")
    
    print(f"{'-'*100}\n")
    
    # Best experiment
    best = df_sorted.iloc[0]
    print(f"🏆 BEST PERFORMING EXPERIMENT:")
    print(f"   Member: {best['member_name']}")
    print(f"   Experiment #{best['experiment_num']}")
    print(f"   Average Reward: {best['avg_reward']:.2f}")
    print(f"   Hyperparameters: lr={best['learning_rate']:.2e}, gamma={best['gamma']:.3f}, "
          f"batch={best['batch_size']}, eps_end={best['epsilon_end']:.2f}")


def plot_hyperparameter_impacts(df):
    """Create plots showing impact of each hyperparameter"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Hyperparameter Impact on Average Reward', fontsize=16)
    
    params = ['learning_rate', 'gamma', 'batch_size', 'epsilon_end', 'epsilon_decay']
    
    for idx, param in enumerate(params):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Scatter plot
        ax.scatter(df[param], df['avg_reward'], alpha=0.6, s=100)
        ax.set_xlabel(param)
        ax.set_ylabel('Average Reward')
        ax.set_title(f'Impact of {param}')
        ax.grid(True, alpha=0.3)
        
        # Add trend line if we have enough points
        if len(df) > 3:
            z = np.polyfit(df[param], df['avg_reward'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df[param].min(), df[param].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8)
    
    # Remove empty subplot
    fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('experiments/hyperparameter_impacts.png', dpi=300)
    print("\n✓ Hyperparameter impact plots saved to: experiments/hyperparameter_impacts.png")
    plt.close()


def plot_member_comparison(df):
    """Create bar chart comparing team members' best results"""
    member_best = df.groupby('member_name')['avg_reward'].max().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    member_best.plot(kind='bar', color='steelblue')
    plt.title('Best Average Reward by Team Member', fontsize=14)
    plt.xlabel('Team Member')
    plt.ylabel('Best Average Reward')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiments/member_comparison.png', dpi=300)
    print("✓ Member comparison plot saved to: experiments/member_comparison.png")
    plt.close()


def generate_markdown_table(df, member_name):
    """Generate a markdown table for the member's experiments"""
    df_member = df[df['member_name'] == member_name].sort_values('experiment_num')
    
    print(f"\n{'='*100}")
    print(f"MARKDOWN TABLE FOR: {member_name}")
    print(f"{'='*100}\n")
    print("Copy this into your documentation:\n")
    
    print("| Exp# | Hyperparameter Set | Noted Behavior |")
    print("|------|-------------------|----------------|")
    
    for _, row in df_member.iterrows():
        hp_set = (f"lr={row['learning_rate']:.2e}, gamma={row['gamma']:.3f}, "
                 f"batch={row['batch_size']}, epsilon_start={row['epsilon_start']:.1f}, "
                 f"epsilon_end={row['epsilon_end']:.2f}, epsilon_decay={row['epsilon_decay']:.2f}")
        
        behavior = row['observations'] if pd.notna(row['observations']) and row['observations'] else \
                   f"Avg reward: {row['avg_reward']:.2f}, Peak: {row['peak_reward']:.2f}"
        
        print(f"| {row['experiment_num']} | {hp_set} | {behavior} |")


if __name__ == "__main__":
    # Load results
    df = load_results()
    
    if df is None:
        exit(1)
    
    # Print overall summary
    print_summary_table(df)
    
    # Print summary for each team member
    for member in df['member_name'].unique():
        print_summary_table(df, member)
        generate_markdown_table(df, member)
    
    # Generate plots
    if len(df) >= 3:
        plot_hyperparameter_impacts(df)
        plot_member_comparison(df)
    
    print(f"\n{'='*100}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*100}")
