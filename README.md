# FastNN Quadrotor Control

Reinforcement learning research for robust quadrotor control under wind, mass perturbations, and dynamic target tracking.

**Key Result**: 100% success rate with **0.10m** tracking error on dynamic figure-8 trajectory (**47× improvement** over baseline).

## Research Papers

### 📄 Primary Publication
- **[FastNN Quadrotor Control with Curriculum Adaptation](docs/publication/README.md)** - Complete academic paper (Abstract, Methods, Results, Discussion)

### 📚 Detailed Research Papers
- **[Complete Research Paper](docs/papers/fastnn_quadrotor_paper.md)** (528 lines) - Full research paper with methods, results, analysis
- **[Experimental History](docs/papers/quadrotor_research_paper.md)** (882 lines) - Complete experimental history and failure analysis
- **[Technical Analysis](docs/papers/quadrotor_best_path_forward.md)** (350 lines) - Architecture decisions and future directions
- **[Research Summary](docs/papers/quadrotor_research_summary.md)** (143 lines) - Concise results and best path forward

### 🔬 Supporting Documentation
- **[Publication Package](docs/publication/)** - Academic publication with methods, results, and appendices
- **[Experiments](docs/experiments/)** - Detailed experimental results (delay robustness study)
- **[Development](docs/development/)** - Planning documents and audit reports

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
