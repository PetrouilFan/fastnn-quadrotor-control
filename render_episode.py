#!/usr/bin/env python3
"""
Enhanced render/evaluation script for quadrotor NN policy.

Usage:
    python render_episode.py --model-path <path> --env-name <name> [options]
    
Options:
    --model-path PATH       Path to .zip checkpoint (required)
    --env-name NAME         Environment class name (required)
    --vecnorm-path PATH     Path to VecNormalize .pkl
    --encoder-path PATH     Path to encoder .pt (stage 15)
    --n-episodes N         Number of episodes (default: 3)
    --seed N               Random seed (default: 42)
    --no-render           Run headless (default: render with MuJoCo)
    --camera CAM           Camera mode: tracking (default), free, or camera name
    --speed FACTOR         Playback speed multiplier (default: 1.0)
    --perturbation NAME   Apply perturbation: nominal, delay_1, delay_2, lag_1.25, lag_1.5, noise_0.01, noise_0.03
    --benchmark           Run benchmark sweep with all perturbations
"""
import os
import sys
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Render/evaluate quadrotor NN policy")
    parser.add_argument("--model-path", type=str, required=True, help="Path to .zip checkpoint")
    parser.add_argument("--env-name", type=str, required=True, help="Environment class name")
    parser.add_argument("--vecnorm-path", type=str, default=None, help="Path to VecNormalize .pkl")
    parser.add_argument("--encoder-path", type=str, default=None, help="Path to encoder .pt")
    parser.add_argument("--n-episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-render", action="store_true", help="Run headless (no MuJoCo viewer)")
    parser.add_argument("--camera", type=str, default="tracking", 
                        help="Camera mode: tracking (default), free, or camera name")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--perturbation", type=str, default="nominal", help="Perturbation to apply")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark sweep")
    args = parser.parse_args()
    
    # Handle benchmark mode
    if args.benchmark:
        run_benchmark(args)
        return
    
    # Single episode run
    run_episodes(args)


def get_env_dt(env):
    """Get environment timestep."""
    # Try various ways to get dt
    if hasattr(env, 'dt'):
        return env.dt
    elif hasattr(env, 's10') and hasattr(env.s10, 'dt'):
        return env.s10.dt
    elif hasattr(env, 'base_env') and hasattr(env.base_env, 'dt'):
        return env.base_env.dt
    elif hasattr(env, 'env') and hasattr(env.env, 'dt'):
        return env.env.dt
    return 0.01  # default


def get_s10_env(env):
    """Get the underlying s10 environment for MuJoCo viewer."""
    # Direct s10
    if hasattr(env, 's10'):
        return env.s10
    # Wrapped environments
    elif hasattr(env, 'base_env') and hasattr(env.base_env, 's10'):
        return env.base_env.s10
    elif hasattr(env, 'env') and hasattr(env.env, 's10'):
        return env.env.s10
    elif hasattr(env, 'env') and hasattr(env.env, 'env') and hasattr(env.env.env, 's10'):
        return env.env.env.s10
    return None


def setup_viewer_camera(viewer, env, camera_mode):
    """Setup viewer camera based on camera mode."""
    if camera_mode == "tracking" or camera_mode == "free":
        # For tracking camera, we need to access the model camera
        # The passive viewer allows setting camera lookat
        pass  # Will handle in render loop
    else:
        # Try to set specific camera by name
        try:
            cam_id = viewer._camera_id
            if cam_id >= 0:
                pass  # Camera already set
        except:
            pass


def run_episodes(args):
    """Run episodes with the model."""
    import torch
    from stable_baselines3 import SAC
    
    print(f"Loading model from {args.model_path}...")
    model = SAC.load(args.model_path)
    
    # Create environment
    env = create_env(args)
    if env is None:
        print(f"Failed to create environment: {args.env_name}")
        return
    
    print(f"Running {args.n_episodes} episodes...")
    print(f"Perturbation: {args.perturbation}")
    print(f"Speed: {args.speed}x")
    print(f"Render: {not args.no_render}")
    if not args.no_render:
        print(f"Camera: {args.camera}")
    
    # Apply perturbation
    apply_perturbation(env, args.perturbation)
    
    # Setup MuJoCo viewer if not headless
    viewer_handle = None
    if not args.no_render:
        viewer_handle = setup_mujoco_viewer(env, args.camera)
        if viewer_handle is None:
            print("WARNING: Failed to launch MuJoCo viewer, running headless")
            args.no_render = True
    
    # Get environment timestep for playback speed control
    env_dt = get_env_dt(env)
    
    results = []
    for ep in range(args.n_episodes):
        np.random.seed(args.seed + ep)
        obs, _ = env.reset()
        done = False
        step = 0
        episode_reward = 0
        ctes = []
        actions = []
        
        print(f"\n=== Episode {ep + 1} ===")
        
        # Reset viewer camera for tracking mode on episode start
        if viewer_handle is not None and args.camera == "tracking":
            reset_tracking_camera(viewer_handle, env)
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step += 1
            
            # Sync viewer with physics state
            if viewer_handle is not None:
                sync_viewer(viewer_handle, env, args.camera)
            
            # Track metrics
            actions.append(np.linalg.norm(action))
            
            # Get CTE - properly handle nested environments
            try:
                s10_env = get_s10_env(env)
                if s10_env is not None:
                    # Check for reference position in various locations
                    ref_pos = None
                    if hasattr(s10_env, '_ref_pos'):
                        ref_pos = s10_env._ref_pos
                    elif hasattr(s10_env, 'ref_pos'):
                        ref_pos = s10_env.ref_pos
                    elif hasattr(s10_env, 'target_pos'):
                        ref_pos = s10_env.target_pos
                    
                    if ref_pos is not None:
                        qpos = s10_env.data.qpos[:3]
                        cte = np.linalg.norm(qpos - ref_pos)
                        ctes.append(cte)
            except Exception as e:
                pass  # CTE tracking optional
            
            # Control playback speed
            if not args.no_render and args.speed > 0:
                time.sleep(env_dt / args.speed)
            
            # Print progress every 20 steps
            if step % 20 == 0 or done:
                cte_str = f"{ctes[-1]:.3f}" if ctes else "N/A"
                print(f"  Step {step:3d} | R: {episode_reward:8.2f} | CTE: {cte_str}")
            
            if done:
                break
        
        # Episode summary
        mean_cte = np.mean(ctes) if ctes else 0
        mean_action = np.mean(actions)
        
        print(f"  Episode {ep + 1} complete:")
        print(f"    Steps: {step}")
        print(f"    Reward: {episode_reward:.2f}")
        print(f"    Mean CTE: {mean_cte:.3f}")
        print(f"    Mean Action: {mean_action:.3f}")
        
        results.append({
            'episode': ep + 1,
            'steps': step,
            'reward': episode_reward,
            'mean_cte': mean_cte,
            'mean_action': mean_action
        })
    
    # Summary
    print(f"\n=== Summary ===")
    rewards = [r['reward'] for r in results]
    ctes = [r['mean_cte'] for r in results]
    print(f"  Mean Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"  Mean CTE: {np.mean(ctes):.3f}")
    
    # Cleanup viewer
    if viewer_handle is not None:
        try:
            viewer_handle.close()
        except:
            pass
    
    env.close()


def setup_mujoco_viewer(env, camera_mode):
    """Setup MuJoCo viewer for 3D rendering."""
    try:
        from mujoco import viewer
        
        # Get the underlying s10 environment which has MuJoCo model and data
        s10_env = get_s10_env(env)
        if s10_env is None:
            print("WARNING: Could not find s10 environment for viewer")
            return None
        
        # Launch passive viewer
        print(f"Launching MuJoCo viewer (camera: {camera_mode})...")
        handle = viewer.launch_passive(s10_env.model, s10_env.data)
        
        # Setup camera if specified
        if camera_mode != "tracking":
            # Try to set camera by name
            try:
                # Get camera ID by name
                cam_id = s10_env.model.camera(camera_mode).id
                if cam_id >= 0:
                    handle.cam.fixedcamid = cam_id
            except:
                pass
        
        print("MuJoCo viewer launched successfully")
        return handle
        
    except Exception as e:
        print(f"WARNING: Could not launch MuJoCo viewer: {e}")
        return None


def reset_tracking_camera(viewer_handle, env):
    """Reset tracking camera to default position."""
    # Set camera to follow quadrotor with reasonable offset
    try:
        # Use free camera mode for tracking
        viewer_handle.cam.fixedcamid = -1
    except:
        pass


def sync_viewer(viewer_handle, env, camera_mode):
    """Sync viewer with current physics state."""
    if viewer_handle is None:
        return
    
    try:
        # Sync the viewer with current MuJoCo data
        viewer_handle.sync()
        
        # For tracking camera mode, update lookat to follow quadrotor
        if camera_mode == "tracking":
            s10_env = get_s10_env(env)
            if s10_env is not None:
                pos = s10_env.data.qpos[:3]
                # Update camera lookat to follow quadrotor
                # The viewer handles this automatically with passive mode
                # but we can set the target if needed
                pass  # Passive viewer automatically follows
                
    except Exception as e:
        pass  # Silently ignore sync errors


def run_benchmark(args):
    """Run benchmark sweep across perturbations."""
    perturbations = [
        "nominal", "delay_1", "delay_2", 
        "lag_1.25", "lag_1.5", 
        "noise_0.01", "noise_0.03"
    ]
    
    print("=" * 60)
    print("BENCHMARK SWEEP")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Episodes per perturbation: {args.n_episodes}")
    print("=" * 60)
    
    results = {}
    
    for pert in perturbations:
        print(f"\n--- {pert} ---")
        args.perturbation = pert
        
        try:
            run_episodes(args)
            results[pert] = "OK"
        except Exception as e:
            print(f"  FAILED: {e}")
            results[pert] = f"ERROR: {e}"
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    for pert, status in results.items():
        print(f"  {pert}: {status}")


def create_env(args):
    """Create environment based on name."""
    env_creators = {
        # Stage 10
        "QuadrotorEnvStage10": lambda: create_stage10(args),
        "QuadrotorEnvStage10Hierarchical": lambda: create_stage10(args),
        
        # Stage 11
        "QuadrotorEnvStage11": lambda: create_stage11(args),
        "QuadrotorEnvStage11Trajectory": lambda: create_stage11(args),
        "QuadrotorEnvStage11Primitive": lambda: create_stage11p(args),
        
        # Stage 12
        "QuadrotorEnvStage12": lambda: create_stage12(args),
        "QuadrotorEnvStage12Hist": lambda: create_stage12(args),
        "QuadrotorEnvStage12Combined": lambda: create_stage12comb(args),
        
        # Stage 13
        "QuadrotorEnvStage13Shielded": lambda: create_stage13s(args),
        "QuadrotorEnvStage13Bodyframe": lambda: create_stage13b(args),
        
        # Stage 14
        "QuadrotorEnvStage14GRUSAC": lambda: create_stage14(args),
        "QuadrotorEnvStage14Smoothed": lambda: create_stage14s(args),
        
        # Stage 15
        "QuadrotorEnvStage15Adaptive": lambda: create_stage15(args),
    }
    
    if args.env_name in env_creators:
        try:
            return env_creators[args.env_name]()
        except Exception as e:
            print(f"Error creating env {args.env_name}: {e}")
            # Try default stage 11
            return create_stage11(args)
    
    print(f"Unknown env: {args.env_name}, trying default stage 11")
    return create_stage11(args)


def create_stage10(args):
    """Create Stage 10 environment."""
    from quadrotor.env_stage10_hierarchical import QuadrotorEnvStage10Hierarchical
    return QuadrotorEnvStage10Hierarchical(
        max_episode_steps=5000,
        motor_lag_tau=0.08,
    )


def create_stage11(args):
    """Create Stage 11 Trajectory environment."""
    from quadrotor.env_stage11_trajectory import QuadrotorEnvStage11Trajectory
    
    # Auto-detect paths
    model_dir = os.path.dirname(args.model_path)
    
    # Try to find stage10 model and vecnorm
    s10_path = os.path.join(model_dir.replace("runs/", "runs/"), "stage10_rl_finetuned.zip")
    if not os.path.exists(s10_path):
        s10_path = "runs/stage10_rl_finetuned/stage10_rl_finetuned.zip"
    
    vecnorm_path = args.vecnorm_path
    if not vecnorm_path:
        base = s10_path.replace(".zip", "")
        for suffix in ["_vecnormalize.pkl", "_vec_normalize.pkl"]:
            if os.path.exists(base + suffix):
                vecnorm_path = base + suffix
                break
    
    try:
        return QuadrotorEnvStage11Trajectory(s10_path, vecnorm_path)
    except:
        return QuadrotorEnvStage11Trajectory(
            "runs/stage10_rl_finetuned/stage10_rl_finetuned.zip",
            "runs/stage10_rl_finetuned/stage10_rl_finetuned_vecnormalize.pkl"
        )


def create_stage11p(args):
    """Create Stage 11 Primitive environment."""
    from quadrotor.env_stage11_primitive_v3 import QuadrotorEnvStage11PrimitiveV3
    return QuadrotorEnvStage11PrimitiveV3()


def create_stage12(args):
    """Create Stage 12 Hist environment."""
    from quadrotor.env_stage11_trajectory import QuadrotorEnvStage11Trajectory
    from quadrotor.env_stage12_hist import QuadrotorEnvStage12Hist
    
    base = create_stage11(args)
    return QuadrotorEnvStage12Hist(base)


def create_stage12comb(args):
    """Create Stage 12 Combined environment."""
    from quadrotor.env_stage12_combined import QuadrotorEnvStage12Combined
    from quadrotor.env_stage11_trajectory import QuadrotorEnvStage11Trajectory
    
    base = QuadrotorEnvStage11Trajectory(
        "runs/stage10_rl_finetuned/stage10_rl_finetuned.zip",
        "runs/stage10_rl_finetuned/stage10_rl_finetuned_vecnormalize.pkl"
    )
    return QuadrotorEnvStage12Combined(base)


def create_stage13s(args):
    """Create Stage 13 Shielded environment."""
    from quadrotor.env_stage13_shielded import QuadrotorEnvStage13Shielded
    from quadrotor.env_stage11_trajectory import QuadrotorEnvStage11Trajectory
    
    base = QuadrotorEnvStage11Trajectory(
        "runs/stage10_rl_finetuned/stage10_rl_finetuned.zip",
        "runs/stage10_rl_finetuned/stage10_rl_finetuned_vecnormalize.pkl"
    )
    return QuadrotorEnvStage13Shielded(base)


def create_stage13b(args):
    """Create Stage 13 Bodyframe environment."""
    from quadrotor.env_stage13_bodyframe import QuadrotorEnvStage13Bodyframe
    from quadrotor.env_stage11_trajectory import QuadrotorEnvStage11Trajectory
    
    base = QuadrotorEnvStage11Trajectory(
        "runs/stage10_rl_finetuned/stage10_rl_finetuned.zip",
        "runs/stage10_rl_finetuned/stage10_rl_finetuned_vecnormalize.pkl"
    )
    return QuadrotorEnvStage13Bodyframe(base)


def create_stage14(args):
    """Create Stage 14 GRU-SAC environment."""
    from quadrotor.env_stage11_trajectory import QuadrotorEnvStage11Trajectory
    from quadrotor.env_stage12_hist import QuadrotorEnvStage12Hist
    
    base = create_stage11(args)
    return QuadrotorEnvStage12Hist(base)


def create_stage14s(args):
    """Create Stage 14 Smoothed environment."""
    from quadrotor.env_stage14_smoothed import QuadrotorEnvStage14Smoothed
    from quadrotor.env_stage11_trajectory import QuadrotorEnvStage11Trajectory
    
    base = QuadrotorEnvStage11Trajectory(
        "runs/stage10_rl_finetuned/stage10_rl_finetuned.zip",
        "runs/stage10_rl_finetuned/stage10_rl_finetuned_vecnormalize.pkl"
    )
    return QuadrotorEnvStage14Smoothed(base)


def create_stage15(args):
    """Create Stage 15 Adaptive environment."""
    from quadrotor.env_stage11_trajectory import QuadrotorEnvStage11Trajectory
    from quadrotor.env_stage12_hist import QuadrotorEnvStage12Hist
    from env_stage15_adaptive import QuadrotorEnvStage15Adaptive
    
    base = create_stage11(args)
    hist = QuadrotorEnvStage12Hist(base)
    
    encoder_path = args.encoder_path
    if not encoder_path:
        # Try to find encoder
        for fname in ["labeled_encoder.pt", "adaptation_encoder.pt"]:
            for root, dirs, files in os.walk("runs"):
                if fname in files:
                    encoder_path = os.path.join(root, fname)
                    break
    
    if not encoder_path:
        print("WARNING: No encoder path provided for Stage 15")
        return hist
    
    return QuadrotorEnvStage15Adaptive(hist, encoder_path)


def apply_perturbation(env, perturbation):
    """Apply perturbation to environment."""
    if perturbation == "nominal":
        return
    
    # Parse perturbation
    if perturbation.startswith("delay_"):
        delay = int(perturbation.split("_")[1])
        print(f"  Applying delay: {delay}")
        if hasattr(env, '_obs_latency'):
            env._obs_latency = delay
            env._obs_buffer = []
        elif hasattr(env, 'base_env') and hasattr(env.base_env, '_obs_latency'):
            env.base_env._obs_latency = delay
            env.base_env._obs_buffer = []
    
    elif perturbation.startswith("lag_"):
        lag = float(perturbation.split("_")[1])
        print(f"  Applying motor_lag: {lag}")
        try:
            if hasattr(env, 's10'):
                env.s10.drone_model.motor_time_constant = lag
            elif hasattr(env, 'base_env') and hasattr(env.base_env, 's10'):
                env.base_env.s10.drone_model.motor_time_constant = lag
        except:
            pass
    
    elif perturbation.startswith("noise_"):
        noise = float(perturbation.split("_")[1])
        print(f"  Applying sensor_noise: {noise}")
        try:
            if hasattr(env, 'dr_sensor_noise'):
                env.dr_sensor_noise = noise
            elif hasattr(env, 'base_env') and hasattr(env.base_env, 'dr_sensor_noise'):
                env.base_env.dr_sensor_noise = noise
        except:
            pass


if __name__ == "__main__":
    main()
