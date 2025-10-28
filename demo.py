#!/usr/bin/env python3
"""
Quick demo script for DJ Controller.
Demonstrates basic functionality without full setup.
"""

import sys
import os
import time
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import FPSCounter, Timer
from smoothing import ExponentialMovingAverage, SmoothingConfig, SmoothingMethod
from gestures import GestureRecognizer


def demo_fps_counter():
    """Demonstrate FPS counter."""
    print("=== FPS Counter Demo ===")
    
    fps_counter = FPSCounter()
    
    print("Simulating frames for 3 seconds...")
    start_time = time.time()
    
    while time.time() - start_time < 3.0:
        fps_counter.update()
        time.sleep(0.033)  # ~30 FPS
        
        if int(time.time() - start_time) != int(time.time() - start_time - 0.033):
            print(f"Current FPS: {fps_counter.get_fps():.1f}")
    
    print(f"Final FPS: {fps_counter.get_fps():.1f}")


def demo_smoothing():
    """Demonstrate signal smoothing."""
    print("\n=== Signal Smoothing Demo ===")
    
    # Create noisy signal
    import numpy as np
    np.random.seed(42)
    
    base_signal = [0.5 + 0.3 * np.sin(i * 0.2) for i in range(50)]
    noisy_signal = [val + np.random.normal(0, 0.1) for val in base_signal]
    
    # Apply EMA smoothing
    ema = ExponentialMovingAverage(alpha=0.3)
    smoothed_signal = []
    
    for value in noisy_signal:
        smoothed = ema.update(value)
        smoothed_signal.append(smoothed)
    
    # Calculate noise reduction
    original_variance = np.var(noisy_signal)
    smoothed_variance = np.var(smoothed_signal)
    reduction = (1 - smoothed_variance / original_variance) * 100
    
    print(f"Original signal variance: {original_variance:.6f}")
    print(f"Smoothed signal variance: {smoothed_variance:.6f}")
    print(f"Noise reduction: {reduction:.1f}%")


def demo_gesture_zones():
    """Demonstrate gesture zone detection."""
    print("\n=== Gesture Zones Demo ===")
    
    recognizer = GestureRecognizer()
    
    print(f"Loaded {len(recognizer.control_zones)} control zones:")
    for name, zone in recognizer.control_zones.items():
        print(f"  - {name}: {zone.zone_type.value} on MIDI channel {zone.midi_channel}")
    
    # Test point detection
    test_points = [
        (0.2, 0.275),  # Crossfader
        (0.075, 0.1),  # Volume A
        (0.1, 0.24),   # Play A button
        (0.5, 0.5)     # Outside all zones
    ]
    
    print("\nTesting point detection:")
    for point in test_points:
        zone_name = recognizer.find_active_zone(point)
        if zone_name:
            zone = recognizer.control_zones[zone_name]
            value = recognizer.calculate_zone_value(point, zone)
            print(f"  Point {point} -> Zone: {zone_name}, Value: {value:.3f}")
        else:
            print(f"  Point {point} -> No zone detected")


def demo_performance():
    """Demonstrate performance measurement."""
    print("\n=== Performance Demo ===")
    
    # Test timer
    timer = Timer()
    timer.start()
    
    # Simulate some work
    total = 0
    for i in range(100000):
        total += i * i
    
    timer.stop()
    print(f"Computation took: {timer.elapsed_ms():.2f}ms")
    
    # Test operation timing
    from utils import timed_operation
    
    with timed_operation("Matrix multiplication") as t:
        import numpy as np
        a = np.random.rand(100, 100)
        b = np.random.rand(100, 100)
        c = np.dot(a, b)
    
    print(f"Matrix operation: {t.elapsed_ms():.2f}ms")


def main():
    """Run all demos."""
    print("🎵 DJ Controller Demo 🎵\n")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        demo_fps_counter()
        demo_smoothing()
        demo_gesture_zones()
        demo_performance()
        
        print("\n✅ Demo completed successfully!")
        print("\nTo run the full application:")
        print("1. Set up your hardware (camera + ArUco markers)")
        print("2. Run: python src/main.py --config config/app_config.json")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())