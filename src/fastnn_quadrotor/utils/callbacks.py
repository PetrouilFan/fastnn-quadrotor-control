#!/usr/bin/env python3
"""
Curriculum Learning Callbacks for Phase 1 Training
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """
    Callback that manages curriculum learning stages.
    Advances through stages based on success/survival rate.
    """

    def __init__(
        self,
        env,
        stage_thresholds,
        eval_freq=5000,
        n_eval_episodes=30,
        log_path="logs",
    ):
        super().__init__()
        self.env = env
        self.stage_thresholds = stage_thresholds
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.current_stage = 1
        self.stage_episode_count = 0
        self.log_path = log_path
        self._last_step_count = 0
        # Bug #3 fix: track next eval step using total transitions (not n_calls)
        # n_calls counts vectorized steps, not total env transitions
        self._next_eval_step = eval_freq

    def _on_step(self) -> bool:
        # Count completed episodes from the vec env dones array
        dones = self.locals.get("dones", np.zeros(1))
        self.stage_episode_count += int(np.sum(dones))

        # Check if it's time for evaluation
        # Bug #3 fix: use model.num_timesteps (total transitions) instead of n_calls
        # n_calls counts vectorized steps; with 16 envs, 10000 n_calls = 160k transitions
        if self.model.num_timesteps >= self._next_eval_step:
            self._next_eval_step += self.eval_freq
            success_rate, survival_rate, mean_reward, max_tilt = (
                self._eval_current_stage()
            )

            print(
                f"[Stage {self.current_stage}] Step {self.model.num_timesteps}: "
                f"success={success_rate:.2f}, survival={survival_rate:.2f}, "
                f"reward={mean_reward:.1f}, max_tilt={max_tilt:.1f}°"
            )

            # Log to tensorboard
            self.logger.record("curriculum/stage", self.current_stage)
            self.logger.record("curriculum/success_rate", success_rate)
            self.logger.record("curriculum/survival_rate", survival_rate)
            self.logger.record("curriculum/mean_reward", mean_reward)
            self.logger.record("curriculum/max_tilt_deg", max_tilt)

            # Save checkpoint
            if survival_rate >= 0.90:
                self.model.save(
                    f"models/rma_phase1_best/stage{self.current_stage}_step{self.model.num_timesteps}"
                )
                print(f"  -> Saved checkpoint for stage {self.current_stage}")

            # Check for curriculum advancement - use SUCCESS (not survival) for advancement
            threshold = self.stage_thresholds.get(self.current_stage)
            if threshold is not None:
                min_eps, min_success = threshold
                print(
                    f"      [Episode check] {self.stage_episode_count}/{min_eps} eps, success={success_rate:.2f}, survival={survival_rate:.2f}"
                )
                if self.stage_episode_count >= min_eps and success_rate >= min_success:
                    self._advance_stage()

        return True

    def _advance_stage(self):
        """Advance to next curriculum stage."""
        if self.current_stage < 4:
            self.current_stage += 1
            self.stage_episode_count = 0  # Bug #7/8: reset per-stage count
            self._set_stage(self.current_stage)
            print(f"\n{'=' * 60}")
            print(f"  ADVANCING TO STAGE {self.current_stage}")
            print(f"{'=' * 60}\n")

    def _set_stage(self, stage):
        """Set curriculum stage on all vectorized envs."""
        self.env.env_method("set_curriculum_stage", stage)

    def _eval_current_stage(self):
        """Run deterministic evaluation using a single env (avoids vec-env issues)."""
        from fastnn_quadrotor.env_rma import RMAQuadrotorEnv

        # Spin up a fresh single env at the current stage
        single_env = RMAQuadrotorEnv(curriculum_stage=self.current_stage)

        successes = 0
        survivals = 0
        rewards = []
        max_tilts = []

        for ep in range(self.n_eval_episodes):
            obs, _ = single_env.reset()
            survived = True
            episode_reward = 0.0
            max_tilt = 0.0
            steps = 0

            while steps < 500:
                # Pass full 60-dim obs; DeployableExtractor slices to 52 internally
                action, _ = self.model.predict(obs[np.newaxis], deterministic=True)
                obs, reward, terminated, truncated, info = single_env.step(action[0])
                episode_reward += float(reward)
                steps += 1

                if "rpy" in info:
                    rpy = info["rpy"]
                    tilt = np.rad2deg(max(abs(rpy[0]), abs(rpy[1])))
                    max_tilt = max(max_tilt, tilt)

                if terminated:
                    survived = False
                    break
                if truncated:
                    survived = True
                    break

            if survived:
                survivals += 1
            if steps >= 500:
                successes += 1

            rewards.append(episode_reward)
            max_tilts.append(max_tilt)

        single_env.close()

        return (
            successes / self.n_eval_episodes,
            survivals / self.n_eval_episodes,
            np.mean(rewards),
            np.max(max_tilts) if max_tilts else 0.0,
        )


class SurvivalLoggingCallback(BaseCallback):
    """Callback to log survival metrics during training."""

    def __init__(self, eval_freq=5000, n_eval_episodes=10):
        super().__init__()
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if self.model.num_timesteps % self.eval_freq == 0:
            # Run quick evaluation
            eval_env = self.training_env

            episodes_reward = []
            episodes_survived = []

            for _ in range(self.n_eval_episodes):
                obs, _ = eval_env.reset()
                terminated, truncated = False, False
                episode_reward = 0

                while not (terminated or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    episode_reward += reward

                episodes_reward.append(episode_reward)
                episodes_survived.append(1 if episode_reward > 0 else 0)

            mean_reward = np.mean(episodes_reward)
            survival_rate = np.mean(episodes_survived)

            print(
                f"Step {self.model.num_timesteps}: mean_reward={mean_reward:.1f}, survival={survival_rate:.2f}"
            )

            self.logger.record("train/mean_reward", mean_reward)
            self.logger.record("train/survival_rate", survival_rate)

        return True