"""
Metric-based checkpoint callback for Stage 11 training.

Saves checkpoints based on p95 CTE (primary), not SB3's default reward.
"""

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MetricCheckpointCallback(BaseCallback):
    """
    Callback that saves best model based on CTE metrics.
    
    Primary: p95 CTE (lower is better)
    Secondary: mean CTE (lower is better)
    """
    
    def __init__(
        self,
        eval_env_fn,
        save_freq: int = 100000,
        save_path: str = "checkpoints",
        n_eval_episodes: int = 20,
        name_prefix: str = "sac_pilot",
       verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.save_freq = save_freq
        self.save_path = save_path
        self.n_eval_episodes = n_eval_episodes
        self.name_prefix = name_prefix
        
        # Track best metrics
        self.best_p95 = float('inf')
        self.best_mean = float('inf')
        self.best_frac = 0.0
        self_calls = 0
        
    def _on_step(self) -> bool:
        # Evaluate periodically
        if self.n_calls % self.save_freq == 0:
            metrics = self._evaluate()
            p95 = metrics['p95_cte']
            mean = metrics['mean_cte']
            frac = metrics['frac_03']
            
            if self.verbose > 0:
                print(f"\n[MetricCB] Step {self.n_calls}:")
                print(f"  p95 CTE: {p95:.3f}m (best: {self.best_p95:.3f}m)")
                print(f"  mean CTE: {mean:.3f}m (best: {self.best_mean:.3f}m)")
                print(f"  frac < 0.3m: {frac*100:.1f}% (best: {self.best_frac*100:.1f}%)")
            
            # Save if p95 improved (primary metric)
            if p95 < self.best_p95:
                self.best_p95 = p95
                self.best_mean = mean
                self.best_frac = frac
                
                # Save checkpoint
                path = os.path.join(
                    self.save_path, 
                    f"{self.name_prefix}_p95_{p95:.3f}m_steps"
                )
                self.model.save(path)
                
                if self.verbose > 0:
                    print(f"  → Saved best model: p95={p95:.3f}m")
            
            # Also log current state
            if self.n_calls % (self.save_freq * 5) == 0:
                # Log to tensorboard if available
                try:
                    self.logger.record("eval/p95_cte", p95)
                    self.logger.record("eval/mean_cte", mean)
                    self.logger.record("eval/frac_03", frac)
                except:
                    pass
                    
        return True
    
    def _evaluate(self) -> dict:
        """Run evaluation and compute CTE metrics."""
        env = self.eval_env_fn()
        
        all_ctes = []
        ep_mean_ctes = []
        
        for ep in range(self.n_eval_episodes):
            obs, info = env.reset()
            ep_ctes = []
            
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                cte = info.get('cross_track_error', 1.0)
                ep_ctes.append(cte)
                all_ctes.append(cte)
                
                if terminated or truncated:
                    break
            
            ep_mean_ctes.append(np.mean(ep_ctes))
        
        env.close()
        
        all_ctes = np.array(all_ctes)
        
        return {
            'mean_cte': np.mean(all_ctes),
            'p95_cte': np.percentile(all_ctes, 95),
            'frac_03': np.mean(all_ctes < 0.3),
            'ep_mean_cte': np.mean(ep_mean_ctes),
        }


class TrackBucketEvalCallback(BaseCallback):
    """
    Evaluates performance broken down by track difficulty.
    
    Easy: line, oval
    Medium: circle
    Hard: figure8, spline
    """
    
    def __init__(
        self,
        eval_env_fn,
        eval_freq: int = 50000,
        n_eval_episodes: int = 30,
        verbose: int = 1,
    ):
        super().__init__()
        self.eval_env_fn = eval_env_fn
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.verbose = verbose
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            results = self._evaluate_by_track()
            
            if self.verbose > 0:
                print(f"\n[TrackEval] Step {self.n_calls}:")
                print(f"  Easy tracks: mean={results['easy']['mean']:.3f}m, p95={results['easy']['p95']:.3f}m")
                print(f"  Medium: mean={results['medium']['mean']:.3f}m, p95={results['medium']['p95']:.3f}m")
                print(f"  Hard: mean={results['hard']['mean']:.3f}m, p95={results['hard']['p95']:.3f}m")
        
        return True
    
    def _evaluate_by_track(self) -> dict:
        """Evaluate each track bucket separately."""
        from collections import defaultdict
        
        track_results = defaultdict(list)
        
        # Track type mapping
        track_bucket = {
            'line': 'easy',
            'oval': 'easy',
            'large_circle': 'medium',
            'small_circle': 'medium',
            'circle': 'medium',
            'figure8': 'hard',
            'spline': 'hard',
        }
        
        env = self.eval_env_fn()
        
        for ep in range(self.n_eval_episodes):
            obs, info = env.reset()
            track_type = env._trajectory_type
            bucket = track_bucket.get(track_type, 'hard')
            
            ep_ctes = []
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                cte = info.get('cross_track_error', 1.0)
                ep_ctes.append(cte)
                
                if terminated or truncated:
                    break
            
            track_results[bucket].extend(ep_ctes)
        
        env.close()
        
        # Compute metrics per bucket
        results = {}
        for bucket, ctes in track_results.items():
            ctes = np.array(ctes)
            results[bucket] = {
                'mean': np.mean(ctes),
                'p95': np.percentile(ctes, 95),
                'frac': np.mean(ctes < 0.3),
            }
        
        return results