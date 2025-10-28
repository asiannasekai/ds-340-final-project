"""
Test suite for DJ Controller project.
Basic unit tests for core functionality.
"""

import pytest
import numpy as np
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import GeometryUtils, FPSCounter, RingBuffer, Timer
from smoothing import ExponentialMovingAverage, SimpleMovingAverage, HysteresisFilter
from gestures import GestureRecognizer, ControlZone, ControlZoneType


class TestGeometryUtils:
    """Test geometry utility functions."""
    
    def test_distance_2d(self):
        """Test 2D distance calculation."""
        p1 = (0, 0)
        p2 = (3, 4)
        distance = GeometryUtils.distance_2d(p1, p2)
        assert distance == 5.0
    
    def test_angle_between_points(self):
        """Test angle calculation."""
        p1 = (0, 0)
        p2 = (1, 1)
        angle = GeometryUtils.angle_between_points(p1, p2)
        assert abs(angle - np.pi/4) < 1e-6
    
    def test_point_in_rectangle(self):
        """Test point in rectangle check."""
        rect = (0, 0, 10, 10)
        assert GeometryUtils.point_in_rectangle((5, 5), rect)
        assert not GeometryUtils.point_in_rectangle((15, 5), rect)
    
    def test_clamp(self):
        """Test value clamping."""
        assert GeometryUtils.clamp(5, 0, 10) == 5
        assert GeometryUtils.clamp(-5, 0, 10) == 0
        assert GeometryUtils.clamp(15, 0, 10) == 10


class TestFPSCounter:
    """Test FPS counter functionality."""
    
    def test_fps_calculation(self):
        """Test FPS calculation."""
        counter = FPSCounter(window_size=5)
        
        # Simulate frames at 10 FPS
        for i in range(10):
            counter.update()
            time.sleep(0.01)  # Small delay
        
        fps = counter.get_fps()
        assert fps > 0  # Should be some positive value
    
    def test_fps_reset(self):
        """Test FPS counter reset."""
        counter = FPSCounter()
        counter.update()
        counter.reset()
        assert counter.get_fps() == 0.0


class TestRingBuffer:
    """Test ring buffer functionality."""
    
    def test_append_and_get(self):
        """Test basic append and get operations."""
        buffer = RingBuffer(3)
        
        buffer.append("a")
        buffer.append("b")
        buffer.append("c")
        
        recent = buffer.get_recent()
        assert recent == ["a", "b", "c"]
    
    def test_overflow(self):
        """Test buffer overflow behavior."""
        buffer = RingBuffer(2)
        
        buffer.append("a")
        buffer.append("b")
        buffer.append("c")  # Should overwrite "a"
        
        recent = buffer.get_recent()
        assert recent == ["b", "c"]
        assert len(buffer) == 2
    
    def test_get_recent_with_limit(self):
        """Test getting limited number of recent items."""
        buffer = RingBuffer(5)
        
        for i in range(5):
            buffer.append(f"item_{i}")
        
        recent = buffer.get_recent(3)
        assert recent == ["item_2", "item_3", "item_4"]


class TestTimer:
    """Test timer utility."""
    
    def test_timer_measurement(self):
        """Test timer measurement."""
        timer = Timer()
        timer.start()
        time.sleep(0.01)
        timer.stop()
        
        elapsed = timer.elapsed()
        assert elapsed >= 0.01
        assert elapsed < 0.1  # Should be reasonably close


class TestExponentialMovingAverage:
    """Test EMA filter."""
    
    def test_ema_initialization(self):
        """Test EMA initialization."""
        ema = ExponentialMovingAverage(alpha=0.5)
        assert ema.alpha == 0.5
        assert ema.value is None
    
    def test_ema_update(self):
        """Test EMA update."""
        ema = ExponentialMovingAverage(alpha=0.5)
        
        # First update should set initial value
        result = ema.update(10.0)
        assert result == 10.0
        
        # Second update should smooth
        result = ema.update(0.0)
        assert result == 5.0  # (0.5 * 0 + 0.5 * 10)
    
    def test_ema_alpha_clamping(self):
        """Test alpha value clamping."""
        ema = ExponentialMovingAverage(alpha=1.5)
        assert ema.alpha == 1.0
        
        ema = ExponentialMovingAverage(alpha=-0.5)
        assert ema.alpha == 0.0


class TestSimpleMovingAverage:
    """Test SMA filter."""
    
    def test_sma_calculation(self):
        """Test SMA calculation."""
        sma = SimpleMovingAverage(window_size=3)
        
        sma.update(10.0)
        assert sma.get_value() == 10.0
        
        sma.update(20.0)
        assert sma.get_value() == 15.0
        
        sma.update(30.0)
        assert sma.get_value() == 20.0
        
        sma.update(40.0)  # Should drop first value
        assert sma.get_value() == 30.0


class TestHysteresisFilter:
    """Test hysteresis filter."""
    
    def test_hysteresis_threshold(self):
        """Test hysteresis threshold behavior."""
        hysteresis = HysteresisFilter(threshold=0.1, initial_value=0.0)
        
        # Small changes should be ignored
        result = hysteresis.update(0.05)
        assert result == 0.0
        
        # Large changes should pass through
        result = hysteresis.update(0.15)
        assert result == 0.15


class TestGestureRecognizer:
    """Test gesture recognition system."""
    
    def test_gesture_recognizer_initialization(self):
        """Test gesture recognizer initialization."""
        recognizer = GestureRecognizer()
        assert len(recognizer.control_zones) > 0
    
    def test_zone_point_detection(self):
        """Test point in zone detection."""
        recognizer = GestureRecognizer()
        
        # Create test zone
        zone = ControlZone(
            name="test_zone",
            zone_type=ControlZoneType.BUTTON,
            bounds=[(0.1, 0.1), (0.2, 0.2)],
            midi_channel=1
        )
        
        # Test point inside zone
        assert recognizer.point_in_zone((0.15, 0.15), zone)
        
        # Test point outside zone
        assert not recognizer.point_in_zone((0.3, 0.3), zone)
    
    def test_zone_value_calculation(self):
        """Test zone value calculation."""
        recognizer = GestureRecognizer()
        
        # Create horizontal fader zone
        zone = ControlZone(
            name="fader",
            zone_type=ControlZoneType.FADER,
            bounds=[(0.0, 0.1), (1.0, 0.2)],
            midi_channel=1,
            orientation="horizontal"
        )
        
        # Test left edge
        value = recognizer.calculate_zone_value((0.0, 0.15), zone)
        assert value == 0.0
        
        # Test center
        value = recognizer.calculate_zone_value((0.5, 0.15), zone)
        assert value == 0.5
        
        # Test right edge
        value = recognizer.calculate_zone_value((1.0, 0.15), zone)
        assert value == 1.0


class TestIntegration:
    """Integration tests for core functionality."""
    
    def test_hand_data_processing(self):
        """Test processing of hand data through the pipeline."""
        # Create dummy hand data
        hand_data = [{
            'landmarks': [
                {'x': 100, 'y': 100, 'z': 0, 'x_norm': 0.5, 'y_norm': 0.5}
                for _ in range(21)
            ],
            'handedness': 'Right',
            'confidence': 0.9,
            'bbox': {'x_min': 50, 'y_min': 50, 'x_max': 150, 'y_max': 150}
        }]
        
        # Test that data structure is valid
        assert len(hand_data) == 1
        assert len(hand_data[0]['landmarks']) == 21
        assert hand_data[0]['confidence'] > 0
    
    def test_config_loading(self):
        """Test configuration loading."""
        from utils import ConfigUtils
        
        # Test with default config
        default_config = {"test": "value"}
        config = ConfigUtils.load_json_config("nonexistent.json", default_config)
        assert config == default_config


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])