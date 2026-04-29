#!/usr/bin/env python3
"""
Simple terminal HUD for motor values - works everywhere
Shows live motor commands updating on a single line
"""

import sys
import time


class TerminalHUD:
    """Live motor display in terminal using carriage return."""

    def __init__(self):
        self.last_len = 0

    def update(self, motors, pos, target=None, reward=0.0, step=0):
        """Update HUD display on single line."""
        motor_str = " ".join([f"M{i + 1}:{v:+.3f}" for i, v in enumerate(motors)])
        pos_str = f"Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
        if target is not None:
            target_str = f"Target: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]"
        else:
            target_str = ""

        line = f"\rStep {step:04d} | {motor_str} | {pos_str} {target_str} | r: {reward:+.2f}"

        # Clear previous line if longer
        if len(line) < self.last_len:
            line += " " * (self.last_len - len(line))
        self.last_len = len(line)

        print(line, end="", flush=True)


# Test
if __name__ == "__main__":
    hud = TerminalHUD()
    for i in range(100):
        motors = [0.1 * i, -0.2 * i, 0.05 * i, -0.1 * i]
        pos = [1.0, 0.5, 1.2]
        target = [1.5, 0.8, 1.0]
        hud.update(motors, pos, target, reward=0.5, step=i)
        time.sleep(0.1)
    print("\nDone")
