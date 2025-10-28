#!/usr/bin/env python3
"""
Setup script for Computer Vision DJ Controller.
Handles installation, configuration, and system checks.
"""

import os
import sys
import subprocess
import json
import argparse
import logging
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    return True


def check_system_dependencies():
    """Check for required system dependencies."""
    dependencies = {
        'cmake': 'cmake --version',
        'pkg-config': 'pkg-config --version',
    }
    
    missing = []
    for dep, cmd in dependencies.items():
        try:
            subprocess.run(cmd.split(), capture_output=True, check=True)
            print(f"✓ {dep} found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"✗ {dep} not found")
            missing.append(dep)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Please install them using your system package manager:")
        print("  Ubuntu/Debian: sudo apt install cmake pkg-config")
        print("  macOS: brew install cmake pkg-config")
        print("  Windows: Install via Chocolatey or manually")
        return False
    
    return True


def install_python_dependencies():
    """Install Python dependencies."""
    print("Installing Python dependencies...")
    
    try:
        # Upgrade pip first
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], check=True)
        
        # Install requirements
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        
        print("✓ Python dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install Python dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    directories = [
        "config",
        "data",
        "models",
        "logs",
        "docs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def create_default_configs():
    """Create default configuration files."""
    configs = {
        "config/app_config.json": {
            "camera": {
                "device_id": 0,
                "width": 1280,
                "height": 720,
                "fps": 30
            },
            "hand_tracking": {
                "max_hands": 2,
                "detection_confidence": 0.7,
                "tracking_confidence": 0.5,
                "model_complexity": 1
            },
            "calibration": {
                "auto_save_path": "config/calibration.json",
                "marker_size": 0.05,
                "board_corners": [
                    [0.0, 0.0],
                    [0.4, 0.0],
                    [0.4, 0.3],
                    [0.0, 0.3]
                ]
            },
            "gestures": {
                "config_path": "config/zones.json",
                "parameters": {
                    "tap_max_distance": 0.02,
                    "tap_max_duration": 0.5,
                    "drag_min_distance": 0.01,
                    "rotate_min_angle": 15.0
                }
            },
            "smoothing": {
                "method": "EMA",
                "alpha": 0.3,
                "hysteresis_threshold": 0.01,
                "enabled": True
            },
            "midi": {
                "port_name": "DJ Controller",
                "config_path": "config/midi_mappings.json"
            },
            "ui": {
                "show_debug": True,
                "show_zones": True,
                "show_hands": True,
                "target_fps": 30
            },
            "logging": {
                "enabled": False,
                "output_path": "data/session_log.csv",
                "log_level": "INFO"
            }
        },
        
        "config/zones.json": {
            "zones": [
                {
                    "name": "crossfader",
                    "zone_type": "fader",
                    "bounds": [[0.15, 0.25], [0.25, 0.30]],
                    "midi_channel": 1,
                    "midi_cc": 8,
                    "orientation": "horizontal",
                    "enabled": True
                },
                {
                    "name": "volume_a",
                    "zone_type": "fader",
                    "bounds": [[0.05, 0.05], [0.10, 0.20]],
                    "midi_channel": 1,
                    "midi_cc": 7,
                    "orientation": "vertical",
                    "enabled": True
                },
                {
                    "name": "volume_b",
                    "zone_type": "fader",
                    "bounds": [[0.30, 0.05], [0.35, 0.20]],
                    "midi_channel": 2,
                    "midi_cc": 7,
                    "orientation": "vertical",
                    "enabled": True
                },
                {
                    "name": "play_a",
                    "zone_type": "button",
                    "bounds": [[0.08, 0.22], [0.12, 0.26]],
                    "midi_channel": 1,
                    "midi_note": 60,
                    "enabled": True
                },
                {
                    "name": "play_b",
                    "zone_type": "button",
                    "bounds": [[0.28, 0.22], [0.32, 0.26]],
                    "midi_channel": 2,
                    "midi_note": 60,
                    "enabled": True
                }
            ]
        },
        
        "config/midi_mappings.json": {
            "midi_mappings": [
                {
                    "control_name": "crossfader",
                    "message_type": "control_change",
                    "channel": 0,
                    "controller": 8,
                    "curve": "linear",
                    "enabled": True
                },
                {
                    "control_name": "volume_a",
                    "message_type": "control_change",
                    "channel": 0,
                    "controller": 7,
                    "curve": "linear",
                    "enabled": True
                },
                {
                    "control_name": "volume_b",
                    "message_type": "control_change",
                    "channel": 1,
                    "controller": 7,
                    "curve": "linear",
                    "enabled": True
                },
                {
                    "control_name": "play_a",
                    "message_type": "note_on",
                    "channel": 0,
                    "controller": 60,
                    "enabled": True
                },
                {
                    "control_name": "play_b",
                    "message_type": "note_on",
                    "channel": 1,
                    "controller": 60,
                    "enabled": True
                }
            ]
        }
    }
    
    for filepath, config in configs.items():
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✓ Created config: {filepath}")
        else:
            print(f"- Config exists: {filepath}")


def test_camera():
    """Test camera functionality."""
    print("Testing camera...")
    
    try:
        import cv2
        
        # Try to initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Camera not accessible (device 0)")
            return False
        
        # Try to read a frame
        ret, frame = cap.read()
        if not ret:
            print("✗ Cannot read from camera")
            cap.release()
            return False
        
        height, width = frame.shape[:2]
        print(f"✓ Camera working ({width}x{height})")
        
        cap.release()
        return True
        
    except ImportError:
        print("✗ OpenCV not installed")
        return False
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False


def test_mediapipe():
    """Test MediaPipe installation."""
    print("Testing MediaPipe...")
    
    try:
        import mediapipe as mp
        
        # Initialize hand tracking
        hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        print("✓ MediaPipe hands module working")
        hands.close()
        return True
        
    except ImportError:
        print("✗ MediaPipe not installed")
        return False
    except Exception as e:
        print(f"✗ MediaPipe test failed: {e}")
        return False


def test_midi():
    """Test MIDI functionality."""
    print("Testing MIDI...")
    
    try:
        import rtmidi
        
        # Try to create MIDI output
        midi_out = rtmidi.MidiOut()
        available_ports = midi_out.get_ports()
        
        print(f"✓ MIDI system working ({len(available_ports)} ports available)")
        midi_out.close_port()
        return True
        
    except ImportError:
        print("✗ python-rtmidi not installed")
        return False
    except Exception as e:
        print(f"✗ MIDI test failed: {e}")
        return False


def run_system_tests():
    """Run all system tests."""
    print("\n=== System Tests ===")
    
    tests = [
        ("Camera", test_camera),
        ("MediaPipe", test_mediapipe),
        ("MIDI", test_midi)
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
    
    print(f"\nTest Results:")
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    return all(results.values())


def create_run_script():
    """Create convenience run script."""
    script_content = """#!/bin/bash
# Convenience script to run DJ Controller

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the application
python src/main.py --config config/app_config.json "$@"
"""
    
    with open("run.sh", 'w') as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod("run.sh", 0o755)
    
    print("✓ Created run script: run.sh")


def main():
    parser = argparse.ArgumentParser(description="Setup DJ Controller project")
    parser.add_argument("--skip-deps", action="store_true", 
                       help="Skip Python dependency installation")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Skip system tests")
    parser.add_argument("--minimal", action="store_true",
                       help="Minimal setup (configs only)")
    
    args = parser.parse_args()
    
    print("=== DJ Controller Setup ===\n")
    
    # Check Python version
    if not check_python_version():
        return 1
    
    if not args.minimal:
        # Check system dependencies
        if not check_system_dependencies():
            return 1
        
        # Install Python dependencies
        if not args.skip_deps:
            if not install_python_dependencies():
                return 1
    
    # Create directories and configs
    create_directories()
    create_default_configs()
    create_run_script()
    
    if not args.skip_tests and not args.minimal:
        # Run system tests
        if not run_system_tests():
            print("\n⚠️  Some tests failed. The system may not work correctly.")
            print("Please check the error messages above and install missing dependencies.")
            return 1
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Print ArUco markers from docs/aruco_markers.pdf")
    print("2. Set up your DJ board with markers at corners")
    print("3. Run calibration test: python src/calibration.py")
    print("4. Start the application: python src/main.py")
    print("\nFor more information, see README.md")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())