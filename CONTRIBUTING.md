# Contributing to FastNN Quadrotor Control

We welcome contributions to this research project! This document outlines how to contribute effectively.

## Code of Conduct

- Be respectful and inclusive
- Focus on technical merit
- Provide constructive feedback
- Follow existing code style

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/fastnn-quadrotor.git
cd fastnn-quadrotor
```

### 2. Create Development Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```

## Code Style

### Python Formatting

We use `black` for code formatting:

```bash
black .
```

Configuration in `pyproject.toml`:
- Line length: 88
- Python version: 3.10+

### Linting

We use `ruff` for linting:

```bash
ruff check . --fix
```

### Type Checking

```bash
mypy .
```

## Running Tests

### Unit Tests

```bash
pytest tests/ -v
```

### Specific Test

```bash
pytest tests/test_env.py::test_reset -v
```

### Coverage

```bash
pytest --cov=env_rma --cov-report=html
```

## Training New Models

### Best Practices

1. **Use version control for models**
   ```bash
   # Save with descriptive names
   models_stage5_curriculum/stage_5/seed_0/final_v2.zip
   ```

2. **Log everything**
   - TensorBoard logs
   - Training parameters
   - Random seeds
   - Environment configuration

3. **Validate before committing**
   - Run evaluation on 100 episodes
   - Check for consistency across seeds
   - Document results

### Training Checklist

- [ ] Set random seed for reproducibility
- [ ] Configure appropriate number of environments
- [ ] Set up TensorBoard logging
- [ ] Configure checkpoint saving
- [ ] Enable evaluation callback
- [ ] Test on small run first (100K steps)

## Experiment Tracking

### Naming Convention

```
{experiment_type}_{stage}_{variation}_{timestamp}
```

Examples:
- `train_stage5_curriculum_20260429_153000`
- `eval_delay_30ms_20260429_160000`

### Metadata

Always include a `results.json` with:
- Experiment parameters
- Random seed
- Training steps
- Final metrics
- Environment configuration

## Documentation

### Code Documentation

```python
def train_model(
    stage: int,
    total_steps: int,
    n_envs: int,
    seed: int = 0,
) -> Dict[str, float]:
    """Train SAC model with curriculum learning.
    
    Args:
        stage: Curriculum stage (1-8)
        total_steps: Total training timesteps
        n_envs: Number of parallel environments
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with evaluation metrics:
        - success_rate: Percentage of successful episodes
        - mean_tracking_error: Mean distance to target
        - mean_final_dist: Final distance to target
    """
```

### Markdown Documentation

- Use clear headings
- Include code examples
- Link to related files
- Update when code changes

## Pull Request Process

### 1. Create Branch

```bash
git checkout -b feature/your-feature
git add .
git commit -m "feat: add new training script for stage 7"
```

### 2. Run Tests

```bash
pytest tests/ -v
black .
ruff check .
```

### 3. Push to Remote

```bash
git push origin feature/your-feature
```

### 4. Create Pull Request

Include in PR description:
- **Summary**: What does this change?
- **Motivation**: Why is this change needed?
- **Testing**: How was this tested?
- **Results**: What are the expected outcomes?

### 5. Review Process

- At least one reviewer required
- All tests must pass
- Code style must be consistent
- Documentation must be updated

## Experiment Results

### Publishing Results

When you have new results:

1. Update relevant markdown files
2. Add to results JSON
3. Update README if significant
4. Create visualization if helpful

### Results Format

```json
{
  "experiment": "delay_robustness",
  "stage": 5,
  "delay_ms": 30,
  "total_steps": 2000000,
  "n_envs": 8,
  "seed": 0,
  "success_rate": 96.0,
  "mean_tracking_error": 0.15,
  "timestamp": "2026-04-29_15:30:00"
}
```

## File Organization

### New Training Scripts

Place in root directory with clear naming:
- `train_<stage>_<feature>.py`
- Example: `train_stage7_yaw.py`

### New Evaluation Scripts

Place in root directory:
- `eval_<feature>.py`
- Example: `eval_robustness.py`

### New Models

Organize by experiment:
```
models_experiment_name/
├── stage_X/
│   ├── seed_0/
│   │   ├── final.zip
│   │   └── results.json
│   └── seed_1/
└── README.md
```

## Common Issues

### Import Errors

```bash
# Check Python path
export PYTHONPATH="${PYTHONPATH}:."

# Verify installation
pip install -e .
```

### CUDA Out of Memory

```bash
# Reduce batch size
# In training script:
batch_size=128  # instead of 256
```

### Training Too Slow

```bash
# Reduce number of environments
python train_stage5_curriculum.py \
  --n-envs 16 \
  ...
```

## Communication

### Questions

- Check existing documentation first
- Search closed issues
- Ask in discussions

### Bug Reports

Include:
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details
- Error messages

### Feature Requests

Include:
- Use case
- Proposed solution
- Alternatives considered

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

- Thanks to all contributors
- Built on MuJoCo, Stable-Baselines3, PyTorch
- Inspired by prior work in RL for robotics

