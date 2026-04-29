#!/usr/bin/env python3
"""
Safety Fallback for Quadrotor Neural Control

Provides hard-coded safety bounds for:
- Attitude (roll/pitch limits)
- Rate limits
- Altitude floor
- Timeout recovery
- Fallback to hover on neural network failure

This is required for hardware deployment to bound failure modes.
"""
import numpy as np
from typing import Tuple, Optional


class SafetyFallback:
    """Hard-coded safety fallback for quadrotor control."""
    
    # Safety bounds
    MAX_ROLL = np.deg2rad(60)  # 60 degrees
    MAX_PITCH = np.deg2rad(60)  # 60 degrees
    MIN_ALTITUDE = 0.15  # 15cm floor
    MAX_RATE = np.deg2rad(180)  # 180 deg/s
    
    # Fallback hover commands (stable hover)
    HOVER_ACTION = np.array([0.0, 0.0, 0.0, 0.5])  # neutral RC
    
    def __init__(self, 
                 max_roll: float = None,
                 max_pitch: float = None,
                 min_altitude: float = None,
                 max_rate: float = None):
        """Initialize with custom bounds."""
        if max_roll is not None:
            self.MAX_ROLL = max_roll
        if max_pitch is not None:
            self.MAX_PITCH = max_pitch
        if min_altitude is not None:
            self.MIN_ALTITUDE = min_altitude
        if max_rate is not None:
            self.MAX_RATE = max_rate
    
    def check_safe(self, 
                attitude: np.ndarray,
                rates: np.ndarray,
                altitude: float,
                nn_confidence: float = 1.0) -> Tuple[bool, str]:
        """
        Check if current state is safe.
        
        Args:
            attitude: [roll, pitch, yaw] in radians
            rates: [roll_rate, pitch_rate, yaw_rate] in rad/s
            altitude: altitude in meters
            nn_confidence: neural network confidence (0-1)
        
        Returns:
            (is_safe, reason)
        """
        roll, pitch, _ = attitude
        
        # Check attitude
        if abs(roll) > self.MAX_ROLL:
            return False, f"roll_exceeded {np.rad2deg(roll):.1f}deg"
        if abs(pitch) > self.MAX_PITCH:
            return False, f"pitch_exceeded {np.rad2deg(pitch):.1f}deg"
        
        # Check rates
        if np.linalg.norm(rates[:2]) > self.MAX_RATE:
            return False, "rate_exceeded"
        
        # Check altitude
        if altitude < self.MIN_ALTITUDE:
            return False, "altitude_low"
        
        # Check NN confidence
        if nn_confidence < 0.3:
            return False, "low_confidence"
        
        return True, "ok"
    
    def project_command(self, 
                    action: np.ndarray,
                    attitude: np.ndarray) -> np.ndarray:
        """
        Project neural command into safe envelope.
        
        Args:
            action: neural network action (4-dim)
            attitude: current attitude
        
        Returns:
            safely projected action
        """
        # Clamp action magnitude
        action = np.clip(action, -0.8, 0.8)
        
        # Gradually reduce aggressiveness near limits
        roll, pitch, _ = attitude
        roll_frac = abs(roll) / self.MAX_ROLL
        pitch_frac = abs(pitch) / self.MAX_PITCH
        
        max_frac = max(roll_frac, pitch_frac)
        if max_frac > 0.5:
            # Reduce action magnitude as we approach limits
            scale = 1.0 - 0.5 * (max_frac - 0.5)
            action = action * scale
        
        return action
    
    def get_fallback(self, 
                     state: str = "hover") -> np.ndarray:
        """
        Get fallback command.
        
        Args:
            state: fallback state ('hover', 'brake', 'land')
        
        Returns:
            fallback action
        """
        if state == "hover":
            return self.HOVER_ACTION.copy()
        elif state == "brake":
            return np.array([0.0, 0.0, 0.0, 0.0])
        elif state == "land":
            return np.array([0.0, 0.0, 0.0, -0.3])
        else:
            return self.HOVER_ACTION.copy()
    
    def should_reset(self,
                    episode_steps: int,
                    is_safe: bool,
                    reason: str) -> bool:
        """
        Determine if episode should reset.
        
        Args:
            episode_steps: number of steps taken
            is_safe: current safety state
            reason: reason for unsafe state
        
        Returns:
            True if should reset
        """
        # Always reset if unsafe
        if not is_safe:
            return True
        
        # Timeout after max steps (prevent infinite episodes)
        MAX_EPISODE_STEPS = 5000
        if episode_steps >= MAX_EPISODE_STEPS:
            return True
        
        return False


class FallbackMonitor:
    """Monitor neural network and switch to fallback when needed."""
    
    def __init__(self, 
                 safety_fallback: SafetyFallback = None,
                 disagreement_threshold: float = 0.5,
                 consecutive_failures: int = 3):
        """
        Initialize fallback monitor.
        
        Args:
            safety_fallback: Safety fallback instance
            disagreement_threshold: ensemble disagreement threshold
            consecutive_failures: consecutive failures before fallback
        """
        self.safety = safety_fallback or SafetyFallback()
        self.disagreement_threshold = disagreement_threshold
        self.consecutive_failures = consecutive_failures
        
        self._failure_count = 0
        self._last_action = None
    
    def update(self,
              action: np.ndarray,
              attitude: np.ndarray,
              rates: np.ndarray,
              altitude: float,
              nn_confidence: float = 1.0,
              ensemble_disagreement: float = 0.0) -> np.ndarray:
        """
        Update and get safe action.
        
        Args:
            action: neural network action
            attitude: current attitude
            rates: current rates
            altitude: current altitude
            nn_confidence: model confidence
            ensemble_disagreement: ensemble disagreement
        
        Returns:
            safe action
        """
        # Check safety
        is_safe, reason = self.safety.check_safe(
            attitude, rates, altitude, nn_confidence
        )
        
        # Check ensemble disagreement
        if ensemble_disagreement > self.disagreement_threshold:
            is_safe = False
            reason = f"disagreement {ensemble_disagreement:.2f}"
        
        if not is_safe:
            self._failure_count += 1
            if self._failure_count >= self.consecutive_failures:
                # Switch to fallback
                action = self.safety.get_fallback("hover")
        else:
            self._failure_count = 0
        
        # Project command into safe envelope
        action = self.safety.project_command(action, attitude)
        
        self._last_action = action
        return action
    
    def reset(self):
        """Reset failure count."""
        self._failure_count = 0
        self._last_action = None


def test_safety_fallback():
    """Test safety fallback."""
    print("Testing SafetyFallback...")
    
    safety = SafetyFallback()
    
    # Test 1: Normal state
    attitude = np.array([0.1, 0.1, 0.0])
    rates = np.array([0.0, 0.0, 0.0])
    altitude = 2.0
    
    is_safe, reason = safety.check_safe(attitude, rates, altitude)
    print(f"  Normal: {is_safe} ({reason})")
    assert is_safe, "Normal state should be safe"
    
    # Test 2: Roll exceeded
    attitude = np.array([1.2, 0.1, 0.0])  # ~69 degrees
    is_safe, reason = safety.check_safe(attitude, rates, altitude)
    print(f"  Roll exceeded: {is_safe} ({reason})")
    assert not is_safe, "Roll exceeded should be unsafe"
    
    # Test 3: Low altitude
    attitude = np.array([0.1, 0.1, 0.0])
    altitude = 0.05
    is_safe, reason = safety.check_safe(attitude, rates, altitude)
    print(f"  Low altitude: {is_safe} ({reason})")
    assert not is_safe, "Low altitude should be unsafe"
    
    # Test 4: Project command
    action = np.array([0.8, 0.8, 0.8, 0.8])
    attitude = np.array([0.9, 0.1, 0.0])  # Near limit
    projected = safety.project_command(action, attitude)
    print(f"  Projected: {np.linalg.norm(projected):.3f} < {np.linalg.norm(action):.3f}")
    assert np.linalg.norm(projected) < np.linalg.norm(action), "Projected should be smaller"
    
    # Test 5: Fallback
    fallback = safety.get_fallback("hover")
    print(f"  Fallback: {fallback}")
    assert np.allclose(fallback, safety.HOVER_ACTION), "Hover fallback matches"
    
    print("\nTesting FallbackMonitor...")
    
    monitor = FallbackMonitor(safety, consecutive_failures=2)
    
    # Simulate safe action
    action = np.array([0.1, 0.1, 0.1, 0.1])
    attitude = np.array([0.1, 0.1, 0.0])
    rates = np.array([0.0, 0.0, 0.0])
    altitude = 2.0
    
    safe_action = monitor.update(action, attitude, rates, altitude)
    print(f"  Safe action: {safe_action}")
    assert np.allclose(safe_action, action), "Safe action unchanged"
    
    # Simulate unsafe state (multiple times)
    for _ in range(3):
        attitude = np.array([1.2, 0.1, 0.0])  # Roll exceeded
        safe_action = monitor.update(action, attitude, rates, altitude)
    
    print(f"  After failures: {safe_action}")
    print("  All tests PASSED!")


if __name__ == "__main__":
    test_safety_fallback()