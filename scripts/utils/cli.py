#!/usr/bin/env python3
"""
Enhanced CLI for Quadrotor NN Control

Usage:
    python cli.py list                      # List all models and environments
    python cli.py menu                      # Interactive menu (default)
    python cli.py episode <n>              # Run episode on model n (default 0)
    python cli.py benchmark <n>           # Run benchmark on model n
    python cli.py render <n>              # Launch MuJoCo render on model n
    python cli.py results                  # View results history
    python cli.py help                    # Show help
"""
import os
import sys
import subprocess
import argparse
import json
import threading
import time
import shutil
from datetime import datetime
from typing import List, Tuple, Optional

# ANSI colors
GREEN = '\033[92m'
YELLOW = '\033[93m'  
RED = '\033[91m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
RESET = '\033[0m'
BOLD = '\033[1m'

def color(text: str, c: str) -> str:
    return f"{c}{text}{RESET}"

def header(text: str) -> str:
    return color(f"\n{'='*60}\n{text}\n{'='*60}", CYAN + BOLD)

def scan_models() -> List[Tuple[str, str, str]]:
    """Scan for model checkpoints. Returns [(stage, name, path), ...]"""
    models = []
    for root, dirs, files in os.walk("runs"):
        for f in files:
            if f.endswith(".zip"):
                path = os.path.join(root, f)
                stage = "unknown"
                for p in root.split(os.sep):
                    if "stage" in p.lower():
                        stage = p
                        break
                models.append((stage, f, path))
    # Sort by modification time (newest first)
    models.sort(key=lambda x: os.path.getmtime(x[2]), reverse=True)
    return models

def scan_envs() -> List[str]:
    """Scan for environment files. Returns [env_name, ...]"""
    envs = []
    # Scan root
    for f in os.listdir("."):
        if f.startswith("env_stage") and f.endswith(".py"):
            envs.append(f[:-3])
    # Scan quadrotor/
    if os.path.exists("quadrotor"):
        for f in os.listdir("quadrotor"):
            if f.startswith("env_stage") and f.endswith(".py"):
                envs.append(f"quadrotor.{f[:-3]}")
    return sorted(envs)

def get_env_for_stage(stage: str) -> str:
    """Auto-detect environment from stage string."""
    stage_lower = stage.lower()
    if "stage15" in stage_lower or "stage_15" in stage_lower or "adaptive" in stage_lower:
        return "QuadrotorEnvStage15Adaptive"
    elif "stage14" in stage_lower or "stage_14" in stage_lower or "gru" in stage_lower:
        return "QuadrotorEnvStage14GRUSAC"
    elif "stage12" in stage_lower or "stage_12" in stage_lower or "hist" in stage_lower:
        return "QuadrotorEnvStage12Hist"
    elif "stage11" in stage_lower or "stage_11" in stage_lower or "trajectory" in stage_lower:
        return "QuadrotorEnvStage11Trajectory"
    elif "stage13" in stage_lower or "stage_13" in stage_lower or "shielded" in stage_lower:
        return "QuadrotorEnvStage13Shielded"
    elif "stage10" in stage_lower or "stage_10" in stage_lower:
        return "QuadrotorEnvStage10Hierarchical"
    return "QuadrotorEnvStage11Trajectory"  # default

def find_encoder(model_path: str) -> Optional[str]:
    """Find matching encoder for model."""
    model_dir = os.path.dirname(model_path)
    # Check various encoder file names
    for fname in ["adaptation_encoder.pt", "labeled_encoder.pt"]:
        fpath = os.path.join(model_dir, fname)
        if os.path.exists(fpath):
            return fpath
    # Check parent dirs
    for parent in [model_dir, os.path.dirname(model_dir)]:
        for root, dirs, files in os.walk(parent):
            for f in files:
                if "encoder" in f and f.endswith(".pt"):
                    return os.path.join(root, f)
    return None

def find_vecnorm(model_path: str) -> Optional[str]:
    """Find matching VecNormalize for model."""
    base = model_path.replace(".zip", "")
    for suffix in ["_vecnormalize.pkl", "_vec_normalize.pkl"]:
        fpath = base + suffix
        if os.path.exists(fpath):
            return fpath
    return None

def get_python() -> str:
    """Get the correct Python interpreter."""
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
    if os.path.exists(venv_python):
        return venv_python
    return "python"

def run_command(cmd: List[str], live: bool = True):
    """Run command with optional live output."""
    print(f"  Running: {' '.join(cmd)}")
    if live:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(f"    {line.rstrip()}")
        proc.wait()
        return proc.returncode
    else:
        result = subprocess.run(cmd)
        return result.returncode

def show_menu(models: List[Tuple], envs: List[str], selected_model: int = 0):
    """Show interactive menu."""
    print(header("QUADROTOR NN CONTROL - Interactive Menu"))
    
    # Models section
    print(f"\n{color('Models:', CYAN + BOLD)} ({len(models)} found)")
    print(color("  #   Stage            Name                    Path", MAGENTA))
    for i, (stage, name, path) in enumerate(models[:20]):
        prefix = "→ " if i == selected_model else "  "
        name_short = name[:25] if len(name) <= 25 else name[:22] + "..."
        print(f"{prefix}{i:3}. {stage:15} {name_short:25} {path[:35]}")
    if len(models) > 20:
        print(f"     ... and {len(models)-20} more")
    
    # Envs section
    print(f"\n{color('Environments:', CYAN + BOLD)} ({len(envs)} found)")
    print(color("  #   Name", MAGENTA))
    for i, env in enumerate(envs[:10]):
        print(f"     {i:3}. {env}")
    if len(envs) > 10:
        print(f"     ... and {len(envs)-10} more")
    
    # Actions section
    print(f"\n{color('Actions:', CYAN + BOLD)}")
    print(f"  e [<n>]  - Run episode (default: {selected_model})")
    print(f"  b [<n>]  - Run benchmark (default: {selected_model})")
    print(f"  r [<n>]  - Launch MuJoCo render (default: {selected_model})")
    print(f"  l        - List all models")
    print(f"  res     - View results")
    print(f"  h       - Help")
    print(f"  q       - Quit")
    
    print(f"\n{color('Selection:', YELLOW)} Model [{selected_model}]: {models[selected_model][0]}: {models[selected_model][1]}")
    print(f"{color('Status:', YELLOW)} Ready")

def cmd_list(args):
    """List all models and environments."""
    models = scan_models()
    envs = scan_envs()
    
    print(header("QUADROTOR NN CONTROL - Model Browser"))
    print(f"\n{color('Models:', CYAN + BOLD)} ({len(models)} found)")
    print(color("  #   Stage            Name                    Path", MAGENTA))
    for i, (stage, name, path) in enumerate(models):
        print(f"  {i:3}. {stage:15} {name:30} {path[:45]}")
    
    print(f"\n{color('Environments:', CYAN + BOLD)} ({len(envs)} found)")
    print(color("  #   Name", MAGENTA))
    for i, env in enumerate(envs):
        print(f"  {i:3}. {env}")
    
    print(f"\n{color('Tip:', YELLOW)} Use: cli.py episode <n> | benchmark <n> | render <n>")

def cmd_menu(args):
    """Interactive menu mode."""
    models = scan_models()
    envs = scan_envs()
    selected = 0
    
    if not models:
        print(color("No models found!", RED))
        return
    
    while True:
        try:
            show_menu(models, envs, selected)
            choice = input(f"\n{color('>', CYAN)} ").strip().split()
            
            if not choice:
                continue
            
            cmd = choice[0].lower()
            
            if cmd == 'q':
                print(color("Goodbye!", GREEN))
                break
            
            elif cmd == 'h':
                print(header("HELP"))
                print("  e [<n>]  - Run episode on model n")
                print("  b [<n>]  - Run benchmark on model n")  
                print("  r [<n>]  - Launch MuJoCo render on model n")
                print("  l        - List all models")
                print("  res      - View results")
                print("  h        - Show this help")
                print("  q        - Quit")
                input("Press Enter to continue...")
            
            elif cmd == 'l':
                cmd_list(args)
                input("Press Enter to continue...")
            
            elif cmd == 'res':
                cmd_results(args)
                input("Press Enter to continue...")
            
            elif cmd in ['e', 'b', 'r']:
                # Get model index
                if len(choice) > 1:
                    try:
                        selected = int(choice[1])
                        selected = max(0, min(selected, len(models)-1))
                    except ValueError:
                        pass
                
                is_render = (cmd == 'r')
                
                # Run the episode/render
                stage, name, path = models[selected]
                env_name = get_env_for_stage(stage)
                encoder = find_encoder(path) if "stage15" in stage.lower() else None
                vecnorm = find_vecnorm(path)
                
                print(f"\n{color('Running:', YELLOW)} {cmd.upper()} on [{selected}] {stage}: {name}")
                print(f"  Env: {env_name}")
                print(f"  Encoder: {encoder or 'None'}")
                print(f"  VecNorm: {vecnorm or 'None'}")
                
                python = get_python()
                cmd = [python, "render_episode.py",
                       "--model-path", path,
                       "--env-name", env_name]
                
                if vecnorm:
                    cmd.extend(["--vecnorm-path", vecnorm])
                if encoder:
                    cmd.extend(["--encoder-path", encoder])
                
                if cmd == 'r':
                    # Render mode - don't use --no-render
                    cmd.extend(["--n-episodes", "3"])
                else:
                    cmd.extend(["--no-render", "--n-episodes", "1"])
                
                print(f"\n  {'='*40}")
                run_command(cmd)
                print(f"  {'='*40}")
                input("Press Enter to continue...")
            
            elif cmd.isdigit():
                selected = int(cmd)
                selected = max(0, min(selected, len(models)-1))
            
        except (EOFError, KeyboardInterrupt):
            print(color("\nGoodbye!", GREEN))
            break
        except Exception as e:
            print(color(f"Error: {e}", RED))
            input("Press Enter to continue...")

def cmd_results(args):
    """View results history."""
    print(header("RESULTS HISTORY"))
    
    results_dir = "runs"
    if not os.path.exists(results_dir):
        print(color("No results found!", YELLOW))
        return
    
    # Find result files
    result_files = []
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.endswith(".json") and "benchmark" in f.lower():
                result_files.append(os.path.join(root, f))
    
    if not result_files:
        print(color("No benchmark results found!", YELLOW))
        return
    
    # Sort by modification time
    result_files.sort(key=os.path.getmtime, reverse=True)
    
    print(color("Benchmark Results:", CYAN + BOLD))
    for i, path in enumerate(result_files[:10]):
        mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
        print(f"  {i}. {os.path.basename(os.path.dirname(path))}: {os.path.basename(path)}")
        print(f"      {mtime}")
    
    if len(result_files) > 10:
        print(f"  ... and {len(result_files)-10} more")

def cmd_episode(model_idx: int, args):
    """Run episode on model."""
    models = scan_models()
    if not models:
        print(color("No models found!", RED))
        return
    
    idx = max(0, min(model_idx, len(models)-1))
    stage, name, path = models[idx]
    env_name = get_env_for_stage(stage)
    encoder = find_encoder(path)
    vecnorm = find_vecnorm(path)
    
    print(f"\n{color('Running episode:', YELLOW)} [{idx}] {stage}: {name}")
    print(f"  Env: {env_name}")
    
    python = get_python()
    cmd = [python, "render_episode.py",
           "--model-path", path,
           "--env-name", env_name]
    
    if vecnorm:
        cmd.extend(["--vecnorm-path", vecnorm])
    if encoder:
        cmd.extend(["--encoder-path", encoder])
    
    cmd.extend(["--no-render", "--n-episodes", "1"])
    
    run_command(cmd)

def cmd_benchmark(model_idx: int, args):
    """Run benchmark on model."""
    models = scan_models()
    if not models:
        print(color("No models found!", RED))
        return
    
    idx = max(0, min(model_idx, len(models)-1))
    stage, name, path = models[idx]
    env_name = get_env_for_stage(stage)
    encoder = find_encoder(path)
    vecnorm = find_vecnorm(path)
    
    print(f"\n{color('Running benchmark:', YELLOW)} [{idx}] {stage}: {name}")
    print(f"  Env: {env_name}")
    print(f"  Perturbations: nominal, delay_1, delay_2, lag variants, noise variants")
    
    python = get_python()
    
    # Run multiple perturbations
    perturbations = ["nominal", "delay_1", "delay_2", "lag_1.25", "noise_0.01"]
    
    results = {}
    for pert in perturbations:
        print(f"\n  --- {pert} ---")
        
        cmd = [python, "render_episode.py",
               "--model-path", path,
               "--env-name", env_name,
               "--n-episodes", "5"]
        
        if vecnorm:
            cmd.extend(["--vecnorm-path", vecnorm])
        if encoder:
            cmd.extend(["--encoder-path", encoder])
        
        cmd.extend(["--no-render"])
        run_command(cmd, live=True)

def cmd_render(model_idx: int, args):
    """Launch MuJoCo render on model."""
    models = scan_models()
    if not models:
        print(color("No models found!", RED))
        return
    
    idx = max(0, min(model_idx, len(models)-1))
    stage, name, path = models[idx]
    env_name = get_env_for_stage(stage)
    encoder = find_encoder(path)
    vecnorm = find_vecnorm(path)
    
    print(f"\n{color('Launching MuJoCo:', YELLOW)} [{idx}] {stage}: {name}")
    print(f"  Env: {env_name}")
    print(f"  MuJoCo window should open...")
    
    python = get_python()
    cmd = [python, "render_episode.py",
           "--model-path", path,
           "--env-name", env_name,
           "--n-episodes", "3"]
    
    if vecnorm:
        cmd.extend(["--vecnorm-path", vecnorm])
    if encoder:
        cmd.extend(["--encoder-path", encoder])
    
    # Remove --no-render to allow visualization
    run_command(cmd)

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced CLI for Quadrotor NN Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py              # Interactive menu
  python cli.py list         # List all models
  python cli.py episode 0    # Run episode on model 0
  python cli.py benchmark 0  # Run benchmark on model 0
  python cli.py render 0       # Launch MuJoCo render on model 0
  python cli.py results       # View results history
        """
    )
    
    parser.add_argument("cmd", nargs="?", default="menu", help="Command: menu, list, episode, benchmark, render, results, help")
    parser.add_argument("model", nargs="?", default="0", help="Model index (default: 0)")
    parser.add_argument("--episodes", "-e", type=int, default=1, help="Number of episodes")
    
    args = parser.parse_args()
    
    # Parse model index
    try:
        model_idx = int(args.model) if args.model else 0
    except ValueError:
        model_idx = 0
    
    if args.cmd == "menu":
        cmd_menu(args)
    elif args.cmd == "list":
        cmd_list(args)
    elif args.cmd == "episode":
        cmd_episode(model_idx, args)
    elif args.cmd == "benchmark":
        cmd_benchmark(model_idx, args)
    elif args.cmd == "render":
        cmd_render(model_idx, args)
    elif args.cmd == "results":
        cmd_results(args)
    elif args.cmd in ["help", "h", "-h", "--help"]:
        print(header("HELP"))
        print("Commands:")
        print("  menu       - Interactive menu (default)")
        print("  list       - List all models and environments")  
        print("  episode <n> - Run episode on model n")
        print("  benchmark <n> - Run benchmark on model n")
        print("  render <n>  - Launch MuJoCo on model n")
        print("  results    - View results history")
    else:
        cmd_menu(args)

if __name__ == "__main__":
    main()