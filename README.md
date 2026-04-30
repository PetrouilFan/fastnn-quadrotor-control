# FastNN Quadrotor Control

Reinforcement learning research for robust quadrotor control under wind, mass perturbations, and dynamic target tracking.

**Key Result**: 100% success rate with **0.10m** tracking error on dynamic figure-8 trajectory (**47× improvement** over baseline).

## Quick Start

```bash
# Install package (editable)
pip install -e .

# Verify installation
python -c "from fastnn_quadrotor.env_rma import RMAQuadrotorEnv; print('✓ Environment loaded')"

# Run interactive visualization
python -m fastnn_quadrotor.visualize
# or
python scripts/utils/visualize.py
```

For detailed usage, training scripts, and full documentation, see **[docs/publication/README.md](docs/publication/README.md)**.

## Repository Structure

```
├── src/fastnn_quadrotor/     # Core library package
│   ├── env_rma.py            # Main MuJoCo environment
│   ├── env_wrapper*.py       # Environment wrappers
│   ├── quadrotor/            # Stage-specific environments
│   ├── utils/                # Utilities (callbacks, controllers, inference)
│   ├── training/             # Reusable training modules
│   └── models/fastnn_exported/  # Exported model weights
├── scripts/                  # Entry-point scripts
│   ├── train/                # 30+ training scripts
│   ├── eval/                 # Evaluation scripts
│   ├── test/                 # Experiment/test scripts
│   └── utils/                # CLI tools (visualize, export, etc.)
├── docs/                     # Comprehensive documentation
│   ├── papers/              Research papers and roadmaps
│   ├── experiments/          Experimental results (delay study)
│   ├── development/          Planning and audit reports
│   ├── publication/          Publication-ready READMEs
│   └── assets/               # Images and figures
├── results_*/                # JSON benchmark results
├── pyproject.toml            # Project configuration
├── uv.lock                   # Dependency lock
├── quadrotor.xml             # MuJoCo model
└── LICENSE                   # MIT License
```

Large training artifacts (`models_*`, `runs/`, `tb_logs_*`, `data/`) are excluded via `.gitignore`.

## Citation

If you use this code, please cite:

```bibtex
@article{fastnn_quadrotor_2026,
  title={FastNN Quadrotor Control: Solving the Precision-Stability Tradeoff},
  author={PetrouilFan},
  year={2026}
}
```

## License

MIT License – see LICENSE file.
