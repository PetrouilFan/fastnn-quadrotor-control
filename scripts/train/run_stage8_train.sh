#!/bin/bash
source .venv/bin/activate
python train_stage8_progressive.py --steps 250000 --n-envs 8 --seeds 0