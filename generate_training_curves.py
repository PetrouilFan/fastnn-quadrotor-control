#!/usr/bin/env python3
"""
Generate training curves from TensorBoard logs.
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import os

def load_tensorboard_data(log_dir, seeds=[0]):
    """Load training data from multiple seeds."""
    all_data = {
        'actor_loss': [],
        'critic_loss': [],
        'ent_coef': [],
        'rollouts': [],
    }

    for seed in seeds:
        seed_dir = f"{log_dir}/seed_{seed}/SAC_1"
        if not os.path.exists(seed_dir):
            print(f"Warning: {seed_dir} not found")
            continue

        events_file = os.listdir(seed_dir)[0]
        ea = event_accumulator.EventAccumulator(f"{seed_dir}/{events_file}")
        ea.Reload()

        seed_data = {}
        for tag in ['train/actor_loss', 'train/critic_loss', 'train/ent_coef']:
            events = ea.Scalars(tag)
            values = [e.value for e in events]
            steps = [e.step for e in events]
            seed_data[tag] = (steps, values)

        all_data['actor_loss'].append(seed_data.get('train/actor_loss', ([], [])))
        all_data['critic_loss'].append(seed_data.get('train/critic_loss', ([], [])))
        all_data['ent_coef'].append(seed_data.get('train/ent_coef', ([], [])))

    return all_data


def smooth(data, window=10):
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
    return smoothed


def plot_training_curves(data, save_path='training_curves.png'):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot actor loss
    ax = axes[0, 0]
    for seed_data in data['actor_loss']:
        steps, values = seed_data
        if len(values) > 0:
            smoothed = smooth(np.array(values), window=50)
            ax.plot(steps[-len(smoothed):], smoothed, alpha=0.7)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Actor Loss')
    ax.set_title('Actor Loss')
    ax.grid(True, alpha=0.3)

    # Plot critic loss
    ax = axes[0, 1]
    for seed_data in data['critic_loss']:
        steps, values = seed_data
        if len(values) > 0:
            smoothed = smooth(np.array(values), window=50)
            ax.plot(steps[-len(smoothed):], smoothed, alpha=0.7)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Critic Loss')
    ax.set_title('Critic Loss')
    ax.grid(True, alpha=0.3)

    # Plot entropy coefficient
    ax = axes[1, 0]
    for seed_data in data['ent_coef']:
        steps, values = seed_data
        if len(values) > 0:
            ax.plot(steps, values, alpha=0.7)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Entropy Coefficient')
    ax.set_title('Entropy Coefficient (α)')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Combined smoothed rewards
    ax = axes[1, 1]
    ax.set_xlabel('Training Steps (thousands)')
    ax.set_ylabel('Success Rate (simulated)')
    ax.set_title('Training Progress (100K step intervals)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to {save_path}")
    plt.close()


def main():
    print("Generating training curves...")

    data = load_tensorboard_data('tb_logs_sac/stage_3', seeds=[0, 1, 2])
    plot_training_curves(data, 'training_curves.png')

    print("Done!")


if __name__ == '__main__':
    main()