"""
Signal smoothing and filtering utilities for DJ controller.
Implements EMA filtering, hysteresis, and other noise reduction techniques.
"""

import numpy as np
import time
from typing import Optional, Dict, List, Tuple, Any
from collections import deque
import logging
from dataclasses import dataclass
from enum import Enum


class SmoothingMethod(Enum):
    """Available smoothing methods."""
    EMA = "exponential_moving_average"
    SMA = "simple_moving_average"
    MEDIAN = "median_filter"
    KALMAN = "kalman_filter"
    HYSTERESIS = "hysteresis"


@dataclass
class SmoothingConfig:
    """Configuration for a smoothing filter."""
    method: SmoothingMethod
    alpha: float = 0.3              # EMA smoothing factor (0-1)
    window_size: int = 5            # Window size for SMA/Median
    hysteresis_threshold: float = 0.02  # Hysteresis threshold
    min_change: float = 0.001       # Minimum change to register
    max_change_rate: float = 1.0    # Maximum change per second
    enabled: bool = True


class ExponentialMovingAverage:
    """
    Exponential Moving Average filter for smooth signal processing.
    """
    
    def __init__(self, alpha: float = 0.3, initial_value: Optional[float] = None):
        """
        Initialize EMA filter.
        
        Args:
            alpha: Smoothing factor (0-1). Higher = less smoothing
            initial_value: Initial filter value
        """
        self.alpha = max(0.0, min(1.0, alpha))  # Clamp to valid range
        self.value = initial_value
        self.is_initialized = initial_value is not None
        self.last_update_time = time.time()
        
    def update(self, new_value: float, timestamp: Optional[float] = None) -> float:
        """
        Update filter with new value.
        
        Args:
            new_value: New input value
            timestamp: Optional timestamp for adaptive alpha
            
        Returns:
            Filtered value
        """
        if timestamp is None:
            timestamp = time.time()
        
        if not self.is_initialized:
            self.value = new_value
            self.is_initialized = True
        else:
            # Standard EMA update
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        
        self.last_update_time = timestamp
        return self.value
    
    def get_value(self) -> Optional[float]:
        """Get current filtered value."""
        return self.value
    
    def reset(self, initial_value: Optional[float] = None):
        """Reset filter state."""
        self.value = initial_value
        self.is_initialized = initial_value is not None
        self.last_update_time = time.time()
    
    def set_alpha(self, alpha: float):
        """Update smoothing factor."""
        self.alpha = max(0.0, min(1.0, alpha))


class SimpleMovingAverage:
    """
    Simple Moving Average filter.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize SMA filter.
        
        Args:
            window_size: Number of samples to average
        """
        self.window_size = max(1, window_size)
        self.values = deque(maxlen=self.window_size)
        
    def update(self, new_value: float, timestamp: Optional[float] = None) -> float:
        """Update filter with new value."""
        self.values.append(new_value)
        return sum(self.values) / len(self.values)
    
    def get_value(self) -> Optional[float]:
        """Get current filtered value."""
        if self.values:
            return sum(self.values) / len(self.values)
        return None
    
    def reset(self):
        """Reset filter state."""
        self.values.clear()


class MedianFilter:
    """
    Median filter for noise spike removal.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize median filter.
        
        Args:
            window_size: Number of samples for median calculation
        """
        self.window_size = max(1, window_size)
        if self.window_size % 2 == 0:
            self.window_size += 1  # Ensure odd window size
        self.values = deque(maxlen=self.window_size)
        
    def update(self, new_value: float, timestamp: Optional[float] = None) -> float:
        """Update filter with new value."""
        self.values.append(new_value)
        return float(np.median(list(self.values)))
    
    def get_value(self) -> Optional[float]:
        """Get current filtered value."""
        if self.values:
            return float(np.median(list(self.values)))
        return None
    
    def reset(self):
        """Reset filter state."""
        self.values.clear()


class HysteresisFilter:
    """
    Hysteresis filter to prevent rapid oscillation around threshold values.
    """
    
    def __init__(self, threshold: float = 0.02, initial_value: Optional[float] = None):
        """
        Initialize hysteresis filter.
        
        Args:
            threshold: Minimum change required to update output
            initial_value: Initial output value
        """
        self.threshold = abs(threshold)
        self.output_value = initial_value
        self.is_initialized = initial_value is not None
        
    def update(self, new_value: float, timestamp: Optional[float] = None) -> float:
        """Update filter with new value."""
        if not self.is_initialized:
            self.output_value = new_value
            self.is_initialized = True
        else:
            # Only update if change exceeds threshold
            change = abs(new_value - self.output_value)
            if change >= self.threshold:
                self.output_value = new_value
        
        return self.output_value
    
    def get_value(self) -> Optional[float]:
        """Get current filtered value."""
        return self.output_value
    
    def reset(self, initial_value: Optional[float] = None):
        """Reset filter state."""
        self.output_value = initial_value
        self.is_initialized = initial_value is not None
    
    def set_threshold(self, threshold: float):
        """Update hysteresis threshold."""
        self.threshold = abs(threshold)


class KalmanFilter:
    """
    Simple 1D Kalman filter for position tracking.
    """
    
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        """
        Initialize Kalman filter.
        
        Args:
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
        """
        self.Q = process_noise    # Process noise
        self.R = measurement_noise  # Measurement noise
        self.P = 1.0             # Error covariance
        self.x = 0.0             # State estimate
        self.is_initialized = False
        
    def update(self, measurement: float, timestamp: Optional[float] = None) -> float:
        """Update filter with new measurement."""
        if not self.is_initialized:
            self.x = measurement
            self.is_initialized = True
            return self.x
        
        # Prediction step
        self.P += self.Q
        
        # Update step
        K = self.P / (self.P + self.R)  # Kalman gain
        self.x += K * (measurement - self.x)
        self.P *= (1 - K)
        
        return self.x
    
    def get_value(self) -> Optional[float]:
        """Get current filtered value."""
        return self.x if self.is_initialized else None
    
    def reset(self, initial_value: Optional[float] = None):
        """Reset filter state."""
        self.x = initial_value if initial_value is not None else 0.0
        self.P = 1.0
        self.is_initialized = initial_value is not None


class RateLimiter:
    """
    Limits the rate of change of a signal.
    """
    
    def __init__(self, max_rate: float = 1.0, initial_value: Optional[float] = None):
        """
        Initialize rate limiter.
        
        Args:
            max_rate: Maximum change per second
            initial_value: Initial value
        """
        self.max_rate = max_rate
        self.value = initial_value
        self.last_timestamp = time.time()
        self.is_initialized = initial_value is not None
        
    def update(self, new_value: float, timestamp: Optional[float] = None) -> float:
        """Update with rate limiting."""
        if timestamp is None:
            timestamp = time.time()
        
        if not self.is_initialized:
            self.value = new_value
            self.is_initialized = True
            self.last_timestamp = timestamp
            return self.value
        
        # Calculate time delta
        dt = timestamp - self.last_timestamp
        if dt <= 0:
            return self.value
        
        # Calculate maximum allowed change
        max_change = self.max_rate * dt
        change = new_value - self.value
        
        # Limit change
        if abs(change) > max_change:
            change = max_change if change > 0 else -max_change
        
        self.value += change
        self.last_timestamp = timestamp
        
        return self.value
    
    def get_value(self) -> Optional[float]:
        """Get current value."""
        return self.value
    
    def reset(self, initial_value: Optional[float] = None):
        """Reset rate limiter."""
        self.value = initial_value
        self.is_initialized = initial_value is not None
        self.last_timestamp = time.time()


class SignalSmoother:
    """
    High-level signal smoother that combines multiple filtering techniques.
    """
    
    def __init__(self, config: SmoothingConfig):
        """
        Initialize signal smoother.
        
        Args:
            config: Smoothing configuration
        """
        self.config = config
        self.primary_filter = self._create_filter(config.method)
        self.rate_limiter = RateLimiter(config.max_change_rate) if config.max_change_rate > 0 else None
        self.hysteresis = HysteresisFilter(config.hysteresis_threshold) if config.hysteresis_threshold > 0 else None
        
        self.raw_value = None
        self.filtered_value = None
        self.last_output = None
        self.update_count = 0
        
    def _create_filter(self, method: SmoothingMethod):
        """Create the primary filter based on method."""
        if method == SmoothingMethod.EMA:
            return ExponentialMovingAverage(self.config.alpha)
        elif method == SmoothingMethod.SMA:
            return SimpleMovingAverage(self.config.window_size)
        elif method == SmoothingMethod.MEDIAN:
            return MedianFilter(self.config.window_size)
        elif method == SmoothingMethod.KALMAN:
            return KalmanFilter()
        else:
            return ExponentialMovingAverage(0.3)  # Default fallback
    
    def update(self, raw_value: float, timestamp: Optional[float] = None) -> float:
        """
        Update smoother with new raw value.
        
        Args:
            raw_value: New input value
            timestamp: Optional timestamp
            
        Returns:
            Smoothed output value
        """
        if not self.config.enabled:
            return raw_value
        
        if timestamp is None:
            timestamp = time.time()
        
        self.raw_value = raw_value
        self.update_count += 1
        
        # Apply primary filtering
        filtered = self.primary_filter.update(raw_value, timestamp)
        
        # Apply rate limiting
        if self.rate_limiter:
            filtered = self.rate_limiter.update(filtered, timestamp)
        
        # Apply hysteresis
        if self.hysteresis:
            filtered = self.hysteresis.update(filtered, timestamp)
        
        # Check minimum change threshold
        if (self.last_output is not None and 
            abs(filtered - self.last_output) < self.config.min_change):
            filtered = self.last_output
        
        self.filtered_value = filtered
        self.last_output = filtered
        
        return filtered
    
    def get_raw_value(self) -> Optional[float]:
        """Get last raw input value."""
        return self.raw_value
    
    def get_filtered_value(self) -> Optional[float]:
        """Get current filtered value."""
        return self.filtered_value
    
    def get_smoothing_factor(self) -> float:
        """Calculate effective smoothing factor based on recent changes."""
        if self.update_count < 2 or self.raw_value is None or self.filtered_value is None:
            return 1.0
        
        raw_change = abs(self.raw_value - (self.last_output or self.raw_value))
        filtered_change = abs(self.filtered_value - (self.last_output or self.filtered_value))
        
        if raw_change == 0:
            return 1.0
        
        return filtered_change / raw_change
    
    def reset(self):
        """Reset all filter states."""
        self.primary_filter.reset()
        if self.rate_limiter:
            self.rate_limiter.reset()
        if self.hysteresis:
            self.hysteresis.reset()
        
        self.raw_value = None
        self.filtered_value = None
        self.last_output = None
        self.update_count = 0
    
    def update_config(self, config: SmoothingConfig):
        """Update smoothing configuration."""
        if config.method != self.config.method:
            # Recreate primary filter if method changed
            current_value = self.primary_filter.get_value()
            self.primary_filter = self._create_filter(config.method)
            if current_value is not None:
                self.primary_filter.update(current_value)
        
        # Update other components
        if hasattr(self.primary_filter, 'set_alpha'):
            self.primary_filter.set_alpha(config.alpha)
        
        if self.rate_limiter:
            self.rate_limiter.max_rate = config.max_change_rate
        
        if self.hysteresis:
            self.hysteresis.set_threshold(config.hysteresis_threshold)
        
        self.config = config


class MultiChannelSmoother:
    """
    Manages smoothing for multiple signal channels.
    """
    
    def __init__(self, default_config: Optional[SmoothingConfig] = None):
        """
        Initialize multi-channel smoother.
        
        Args:
            default_config: Default configuration for new channels
        """
        self.default_config = default_config or SmoothingConfig(
            method=SmoothingMethod.EMA,
            alpha=0.3,
            hysteresis_threshold=0.01
        )
        
        self.smoothers: Dict[str, SignalSmoother] = {}
        self.channel_configs: Dict[str, SmoothingConfig] = {}
        
    def add_channel(self, channel_name: str, config: Optional[SmoothingConfig] = None):
        """Add a new smoothing channel."""
        config = config or self.default_config
        self.smoothers[channel_name] = SignalSmoother(config)
        self.channel_configs[channel_name] = config
        
    def update_channel(self, channel_name: str, value: float, 
                      timestamp: Optional[float] = None) -> float:
        """Update a specific channel."""
        if channel_name not in self.smoothers:
            self.add_channel(channel_name)
        
        return self.smoothers[channel_name].update(value, timestamp)
    
    def update_multiple(self, values: Dict[str, float], 
                       timestamp: Optional[float] = None) -> Dict[str, float]:
        """Update multiple channels at once."""
        results = {}
        for channel_name, value in values.items():
            results[channel_name] = self.update_channel(channel_name, value, timestamp)
        return results
    
    def get_channel_value(self, channel_name: str) -> Optional[float]:
        """Get current filtered value for a channel."""
        if channel_name in self.smoothers:
            return self.smoothers[channel_name].get_filtered_value()
        return None
    
    def get_all_values(self) -> Dict[str, Optional[float]]:
        """Get current values for all channels."""
        return {name: smoother.get_filtered_value() 
                for name, smoother in self.smoothers.items()}
    
    def reset_channel(self, channel_name: str):
        """Reset a specific channel."""
        if channel_name in self.smoothers:
            self.smoothers[channel_name].reset()
    
    def reset_all(self):
        """Reset all channels."""
        for smoother in self.smoothers.values():
            smoother.reset()
    
    def get_channel_names(self) -> List[str]:
        """Get list of all channel names."""
        return list(self.smoothers.keys())
    
    def update_channel_config(self, channel_name: str, config: SmoothingConfig):
        """Update configuration for a specific channel."""
        if channel_name in self.smoothers:
            self.smoothers[channel_name].update_config(config)
            self.channel_configs[channel_name] = config
    
    def get_performance_stats(self) -> Dict[str, Dict]:
        """Get performance statistics for all channels."""
        stats = {}
        for name, smoother in self.smoothers.items():
            stats[name] = {
                'update_count': smoother.update_count,
                'smoothing_factor': smoother.get_smoothing_factor(),
                'raw_value': smoother.get_raw_value(),
                'filtered_value': smoother.get_filtered_value()
            }
        return stats


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test different smoothing methods
    print("Testing Signal Smoothing...")
    
    # Create test signal with noise
    np.random.seed(42)
    test_signal = []
    for i in range(100):
        # Base signal: sine wave
        base = 0.5 + 0.3 * np.sin(i * 0.1)
        # Add noise
        noise = np.random.normal(0, 0.05)
        test_signal.append(base + noise)
    
    # Test different smoothers
    configs = {
        'EMA': SmoothingConfig(SmoothingMethod.EMA, alpha=0.3),
        'SMA': SmoothingConfig(SmoothingMethod.SMA, window_size=5),
        'Median': SmoothingConfig(SmoothingMethod.MEDIAN, window_size=5),
        'Kalman': SmoothingConfig(SmoothingMethod.KALMAN),
        'Hysteresis': SmoothingConfig(SmoothingMethod.EMA, alpha=0.5, hysteresis_threshold=0.02)
    }
    
    smoothers = {name: SignalSmoother(config) for name, config in configs.items()}
    
    # Process test signal
    results = {name: [] for name in smoothers.keys()}
    
    for i, value in enumerate(test_signal):
        timestamp = i * 0.01  # 100Hz sampling
        for name, smoother in smoothers.items():
            filtered = smoother.update(value, timestamp)
            results[name].append(filtered)
    
    # Calculate smoothing effectiveness (variance reduction)
    original_variance = np.var(test_signal)
    print(f"Original signal variance: {original_variance:.6f}")
    
    for name, filtered_signal in results.items():
        filtered_variance = np.var(filtered_signal)
        reduction = (1 - filtered_variance / original_variance) * 100
        print(f"{name:12} variance: {filtered_variance:.6f} (reduction: {reduction:.1f}%)")
    
    # Test multi-channel smoother
    print("\nTesting Multi-Channel Smoother...")
    multi_smoother = MultiChannelSmoother()
    
    # Add channels for DJ controls
    multi_smoother.add_channel('crossfader')
    multi_smoother.add_channel('volume_a')
    multi_smoother.add_channel('volume_b')
    
    # Simulate control updates
    for i in range(10):
        values = {
            'crossfader': 0.5 + 0.1 * np.sin(i * 0.5) + np.random.normal(0, 0.02),
            'volume_a': 0.8 + np.random.normal(0, 0.03),
            'volume_b': 0.6 + np.random.normal(0, 0.03)
        }
        
        smoothed = multi_smoother.update_multiple(values)
        print(f"Step {i+1}: {smoothed}")
    
    print("Signal smoothing test completed!")