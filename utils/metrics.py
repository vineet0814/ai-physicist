"""
Metrics Logger Module
Handles logging and monitoring of training metrics
"""

import json
import os
import logging
from typing import Dict, Any, Optional, List
import numpy as np
from datetime import datetime
import csv

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

class MetricsLogger:
    """Handles logging and monitoring of training metrics"""

    def __init__(self, log_dir: str = "logs", use_wandb: bool = True):
        self.log_dir = log_dir
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Initialize log files
        self.training_log_file = os.path.join(log_dir, "training_metrics.csv")
        self.episode_log_file = os.path.join(log_dir, "episode_metrics.csv")

        # Initialize CSV files
        self._initialize_csv_files()

        # Metrics storage
        self.training_metrics = []
        self.episode_metrics = []

        if not WANDB_AVAILABLE and use_wandb:
            logger.warning("Wandb not available, metrics will only be logged locally")

    def _initialize_csv_files(self):
        """Initialize CSV files with headers"""
        # Training metrics CSV
        if not os.path.exists(self.training_log_file):
            with open(self.training_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'epoch', 'batch', 'loss', 'reward_mean', 
                    'reward_std', 'kl_divergence', 'entropy', 'value_loss',
                    'policy_loss', 'learning_rate'
                ])

        # Episode metrics CSV
        if not os.path.exists(self.episode_log_file):
            with open(self.episode_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'epoch', 'episode', 'total_reward', 
                    'num_turns', 'tools_used', 'evidence_collected',
                    'tool_success_rate', 'hypothesis_title'
                ])

    def log_training_stats(self, stats: Dict[str, Any], epoch: int, batch: int):
        """Log training statistics"""
        timestamp = datetime.now().isoformat()

        # Prepare metrics
        metrics = {
            'timestamp': timestamp,
            'epoch': epoch,
            'batch': batch,
            'loss': stats.get('loss', 0.0),
            'reward_mean': stats.get('reward_mean', 0.0),
            'reward_std': stats.get('reward_std', 0.0),
            'kl_divergence': stats.get('kl_divergence', 0.0),
            'entropy': stats.get('entropy', 0.0),
            'value_loss': stats.get('value_loss', 0.0),
            'policy_loss': stats.get('policy_loss', 0.0),
            'learning_rate': stats.get('learning_rate', 0.0)
        }

        # Log to CSV
        with open(self.training_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics['timestamp'], metrics['epoch'], metrics['batch'],
                metrics['loss'], metrics['reward_mean'], metrics['reward_std'],
                metrics['kl_divergence'], metrics['entropy'], metrics['value_loss'],
                metrics['policy_loss'], metrics['learning_rate']
            ])

        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'training/loss': metrics['loss'],
                'training/reward_mean': metrics['reward_mean'],
                'training/reward_std': metrics['reward_std'],
                'training/kl_divergence': metrics['kl_divergence'],
                'training/entropy': metrics['entropy'],
                'training/value_loss': metrics['value_loss'],
                'training/policy_loss': metrics['policy_loss'],
                'training/learning_rate': metrics['learning_rate'],
                'epoch': epoch,
                'batch': batch
            })

        # Store metrics
        self.training_metrics.append(metrics)

        logger.info(f"Training stats - Epoch {epoch}, Batch {batch}: "
                   f"Loss: {metrics['loss']:.4f}, "
                   f"Reward: {metrics['reward_mean']:.4f}±{metrics['reward_std']:.4f}")

    def log_episode_metrics(self, episode_data: Dict[str, Any], epoch: int, episode: int):
        """Log episode metrics"""
        timestamp = datetime.now().isoformat()

        # Prepare metrics
        metrics = {
            'timestamp': timestamp,
            'epoch': epoch,
            'episode': episode,
            'total_reward': episode_data.get('total_reward', 0.0),
            'num_turns': episode_data.get('turn_count', 0),
            'tools_used': episode_data.get('tools_used', 0),
            'evidence_collected': episode_data.get('evidence_collected', 0),
            'tool_success_rate': episode_data.get('tool_success_rate', 0.0),
            'hypothesis_title': episode_data.get('hypothesis', 'Unknown')
        }

        # Log to CSV
        with open(self.episode_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics['timestamp'], metrics['epoch'], metrics['episode'],
                metrics['total_reward'], metrics['num_turns'], metrics['tools_used'],
                metrics['evidence_collected'], metrics['tool_success_rate'],
                metrics['hypothesis_title']
            ])

        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'episode/total_reward': metrics['total_reward'],
                'episode/num_turns': metrics['num_turns'],
                'episode/tools_used': metrics['tools_used'],
                'episode/evidence_collected': metrics['evidence_collected'],
                'episode/tool_success_rate': metrics['tool_success_rate'],
                'epoch': epoch,
                'episode': episode
            })

        # Store metrics
        self.episode_metrics.append(metrics)

    def log_epoch_metrics(self, epoch_data: Dict[str, Any], epoch: int):
        """Log epoch-level metrics"""
        if self.use_wandb:
            wandb.log({
                'epoch/average_reward': epoch_data.get('average_reward', 0.0),
                'epoch/average_turns': epoch_data.get('average_turns', 0.0),
                'epoch/num_episodes': epoch_data.get('num_episodes', 0),
                'epoch/total_reward': epoch_data.get('total_reward', 0.0),
                'epoch': epoch
            })

        logger.info(f"Epoch {epoch} summary: "
                   f"Episodes: {epoch_data.get('num_episodes', 0)}, "
                   f"Avg Reward: {epoch_data.get('average_reward', 0.0):.4f}, "
                   f"Avg Turns: {epoch_data.get('average_turns', 0.0):.1f}")

    def log_tool_usage(self, tool_stats: Dict[str, Any], epoch: int):
        """Log tool usage statistics"""
        if self.use_wandb:
            for tool_name, stats in tool_stats.items():
                wandb.log({
                    f'tools/{tool_name}/usage_count': stats.get('usage_count', 0),
                    f'tools/{tool_name}/success_rate': stats.get('success_rate', 0.0),
                    'epoch': epoch
                })

        logger.info(f"Tool usage statistics for epoch {epoch}:")
        for tool_name, stats in tool_stats.items():
            logger.info(f"  {tool_name}: {stats.get('usage_count', 0)} uses, "
                       f"{stats.get('success_rate', 0.0)*100:.1f}% success rate")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training metrics"""
        if not self.training_metrics:
            return {}

        rewards = [m['reward_mean'] for m in self.training_metrics]
        losses = [m['loss'] for m in self.training_metrics]

        return {
            'total_steps': len(self.training_metrics),
            'final_reward': rewards[-1] if rewards else 0.0,
            'best_reward': max(rewards) if rewards else 0.0,
            'reward_improvement': rewards[-1] - rewards[0] if len(rewards) > 1 else 0.0,
            'final_loss': losses[-1] if losses else 0.0,
            'best_loss': min(losses) if losses else 0.0,
            'avg_reward': np.mean(rewards) if rewards else 0.0,
            'avg_loss': np.mean(losses) if losses else 0.0
        }

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of episode metrics"""
        if not self.episode_metrics:
            return {}

        total_rewards = [m['total_reward'] for m in self.episode_metrics]
        num_turns = [m['num_turns'] for m in self.episode_metrics]
        tools_used = [m['tools_used'] for m in self.episode_metrics]

        return {
            'total_episodes': len(self.episode_metrics),
            'avg_reward': np.mean(total_rewards) if total_rewards else 0.0,
            'best_reward': max(total_rewards) if total_rewards else 0.0,
            'avg_turns': np.mean(num_turns) if num_turns else 0.0,
            'avg_tools_used': np.mean(tools_used) if tools_used else 0.0,
            'reward_std': np.std(total_rewards) if total_rewards else 0.0
        }

    def save_summary(self, path: str):
        """Save training summary to file"""
        summary = {
            'training_summary': self.get_training_summary(),
            'episode_summary': self.get_episode_summary(),
            'timestamp': datetime.now().isoformat()
        }

        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Training summary saved to {path}")