"""
Utility functions and helper classes for the DJ Controller project.
Common functionality shared across modules.
"""

import cv2
import numpy as np
import time
import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import threading
from contextlib import contextmanager
from dataclasses import dataclass
import math


class Timer:
    """Simple timer utility for performance measurement."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        
    def stop(self):
        """Stop the timer."""
        self.end_time = time.time()
        
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time
    
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed() * 1000


@contextmanager
def timed_operation(operation_name: str, log_result: bool = True):
    """Context manager for timing operations."""
    timer = Timer()
    timer.start()
    
    try:
        yield timer
    finally:
        timer.stop()
        if log_result:
            logging.debug(f"{operation_name} took {timer.elapsed_ms():.2f}ms")


class FPSCounter:
    """Utility class for calculating and tracking FPS."""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter.
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = []
        self.current_fps = 0.0
        self.last_update = time.time()
        
    def update(self):
        """Update FPS calculation with new frame."""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only recent frames
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        
        # Calculate FPS
        if len(self.frame_times) >= 2:
            time_span = self.frame_times[-1] - self.frame_times[0]
            if time_span > 0:
                self.current_fps = (len(self.frame_times) - 1) / time_span
        
        self.last_update = current_time
    
    def get_fps(self) -> float:
        """Get current FPS."""
        return self.current_fps
    
    def reset(self):
        """Reset FPS counter."""
        self.frame_times.clear()
        self.current_fps = 0.0


class GeometryUtils:
    """Utility functions for geometric calculations."""
    
    @staticmethod
    def distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate 2D distance between two points."""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    @staticmethod
    def angle_between_points(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate angle between two points in radians."""
        return math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    
    @staticmethod
    def point_in_rectangle(point: Tuple[float, float], 
                          rect: Tuple[float, float, float, float]) -> bool:
        """Check if point is inside rectangle."""
        x, y = point
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2
    
    @staticmethod
    def point_in_circle(point: Tuple[float, float], 
                       center: Tuple[float, float], radius: float) -> bool:
        """Check if point is inside circle."""
        distance = GeometryUtils.distance_2d(point, center)
        return distance <= radius
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max."""
        return max(min_val, min(max_val, value))
    
    @staticmethod
    def lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation between a and b."""
        return a + t * (b - a)
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-π, π] range."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


class ColorUtils:
    """Utility functions for color manipulation and visualization."""
    
    # Common colors in BGR format (OpenCV)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (128, 128, 128)
    
    @staticmethod
    def hsv_to_bgr(h: float, s: float, v: float) -> Tuple[int, int, int]:
        """Convert HSV to BGR color."""
        # Normalize HSV values
        h = int(h * 180 / 360)  # OpenCV uses 0-180 for hue
        s = int(s * 255)
        v = int(v * 255)
        
        # Create HSV image
        hsv_color = np.uint8([[[h, s, v]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
        
        return tuple(map(int, bgr_color[0, 0]))
    
    @staticmethod
    def interpolate_color(color1: Tuple[int, int, int], 
                         color2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
        """Interpolate between two colors."""
        t = GeometryUtils.clamp(t, 0.0, 1.0)
        
        r = int(GeometryUtils.lerp(color1[2], color2[2], t))
        g = int(GeometryUtils.lerp(color1[1], color2[1], t))
        b = int(GeometryUtils.lerp(color1[0], color2[0], t))
        
        return (b, g, r)  # BGR format
    
    @staticmethod
    def get_confidence_color(confidence: float) -> Tuple[int, int, int]:
        """Get color based on confidence value (0-1)."""
        if confidence < 0.5:
            # Red to yellow
            return ColorUtils.interpolate_color(ColorUtils.RED, ColorUtils.YELLOW, confidence * 2)
        else:
            # Yellow to green
            return ColorUtils.interpolate_color(ColorUtils.YELLOW, ColorUtils.GREEN, (confidence - 0.5) * 2)


class ImageUtils:
    """Utility functions for image processing and visualization."""
    
    @staticmethod
    def draw_text_with_background(image: np.ndarray, text: str, position: Tuple[int, int],
                                 font: int = cv2.FONT_HERSHEY_SIMPLEX, scale: float = 0.7,
                                 color: Tuple[int, int, int] = ColorUtils.WHITE,
                                 thickness: int = 2,
                                 bg_color: Tuple[int, int, int] = ColorUtils.BLACK,
                                 padding: int = 5) -> np.ndarray:
        """Draw text with background rectangle."""
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        
        # Calculate background rectangle
        x, y = position
        bg_x1 = x - padding
        bg_y1 = y - text_height - padding
        bg_x2 = x + text_width + padding
        bg_y2 = y + baseline + padding
        
        # Draw background
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
        
        # Draw text
        cv2.putText(image, text, position, font, scale, color, thickness)
        
        return image
    
    @staticmethod
    def draw_crosshair(image: np.ndarray, center: Tuple[int, int],
                      size: int = 10, color: Tuple[int, int, int] = ColorUtils.RED,
                      thickness: int = 2) -> np.ndarray:
        """Draw crosshair at specified position."""
        x, y = center
        
        # Horizontal line
        cv2.line(image, (x - size, y), (x + size, y), color, thickness)
        
        # Vertical line
        cv2.line(image, (x, y - size), (x, y + size), color, thickness)
        
        return image
    
    @staticmethod
    def draw_progress_bar(image: np.ndarray, position: Tuple[int, int],
                         size: Tuple[int, int], progress: float,
                         bg_color: Tuple[int, int, int] = ColorUtils.GRAY,
                         fg_color: Tuple[int, int, int] = ColorUtils.GREEN,
                         border_color: Tuple[int, int, int] = ColorUtils.WHITE) -> np.ndarray:
        """Draw progress bar."""
        x, y = position
        width, height = size
        
        # Clamp progress
        progress = GeometryUtils.clamp(progress, 0.0, 1.0)
        
        # Draw background
        cv2.rectangle(image, (x, y), (x + width, y + height), bg_color, -1)
        
        # Draw progress
        progress_width = int(width * progress)
        if progress_width > 0:
            cv2.rectangle(image, (x, y), (x + progress_width, y + height), fg_color, -1)
        
        # Draw border
        cv2.rectangle(image, (x, y), (x + width, y + height), border_color, 1)
        
        return image
    
    @staticmethod
    def resize_maintain_aspect(image: np.ndarray, target_size: Tuple[int, int],
                              fill_color: Tuple[int, int, int] = ColorUtils.BLACK) -> np.ndarray:
        """Resize image while maintaining aspect ratio."""
        target_width, target_height = target_size
        height, width = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(target_width / width, target_height / height)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height))
        
        # Create output image with target size
        result = np.full((target_height, target_width, 3), fill_color, dtype=np.uint8)
        
        # Center the resized image
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        
        result[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
        
        return result


class ConfigUtils:
    """Utility functions for configuration management."""
    
    @staticmethod
    def load_json_config(filepath: str, default_config: Optional[Dict] = None) -> Dict:
        """Load JSON configuration file with fallback to defaults."""
        if not os.path.exists(filepath):
            if default_config is not None:
                logging.warning(f"Config file {filepath} not found, using defaults")
                return default_config.copy()
            else:
                raise FileNotFoundError(f"Config file {filepath} not found")
        
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            # Merge with defaults if provided
            if default_config is not None:
                merged_config = default_config.copy()
                ConfigUtils._deep_merge(merged_config, config)
                return merged_config
            
            return config
            
        except Exception as e:
            logging.error(f"Failed to load config from {filepath}: {e}")
            if default_config is not None:
                return default_config.copy()
            raise
    
    @staticmethod
    def save_json_config(config: Dict, filepath: str, create_dirs: bool = True) -> bool:
        """Save configuration to JSON file."""
        try:
            if create_dirs:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2, sort_keys=True)
            
            logging.info(f"Configuration saved to {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save config to {filepath}: {e}")
            return False
    
    @staticmethod
    def _deep_merge(base_dict: Dict, update_dict: Dict):
        """Recursively merge dictionaries."""
        for key, value in update_dict.items():
            if (key in base_dict and 
                isinstance(base_dict[key], dict) and 
                isinstance(value, dict)):
                ConfigUtils._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value


class ThreadSafeCounter:
    """Thread-safe counter utility."""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self) -> int:
        """Increment counter and return new value."""
        with self._lock:
            self._value += 1
            return self._value
    
    def decrement(self) -> int:
        """Decrement counter and return new value."""
        with self._lock:
            self._value -= 1
            return self._value
    
    def get(self) -> int:
        """Get current value."""
        with self._lock:
            return self._value
    
    def set(self, value: int):
        """Set counter value."""
        with self._lock:
            self._value = value
    
    def reset(self):
        """Reset counter to zero."""
        with self._lock:
            self._value = 0


class RingBuffer:
    """Fixed-size ring buffer for storing recent values."""
    
    def __init__(self, size: int):
        """
        Initialize ring buffer.
        
        Args:
            size: Maximum number of items to store
        """
        self.size = size
        self.buffer = [None] * size
        self.index = 0
        self.count = 0
        self._lock = threading.Lock()
    
    def append(self, item: Any):
        """Add item to buffer."""
        with self._lock:
            self.buffer[self.index] = item
            self.index = (self.index + 1) % self.size
            self.count = min(self.count + 1, self.size)
    
    def get_recent(self, n: Optional[int] = None) -> List[Any]:
        """Get n most recent items (or all if n is None)."""
        with self._lock:
            if n is None:
                n = self.count
            else:
                n = min(n, self.count)
            
            if n == 0:
                return []
            
            items = []
            for i in range(n):
                idx = (self.index - 1 - i) % self.size
                if self.buffer[idx] is not None:
                    items.append(self.buffer[idx])
            
            return list(reversed(items))
    
    def get_oldest(self, n: Optional[int] = None) -> List[Any]:
        """Get n oldest items (or all if n is None)."""
        with self._lock:
            if n is None:
                n = self.count
            else:
                n = min(n, self.count)
            
            if n == 0:
                return []
            
            items = []
            start_idx = (self.index - self.count) % self.size
            
            for i in range(n):
                idx = (start_idx + i) % self.size
                if self.buffer[idx] is not None:
                    items.append(self.buffer[idx])
            
            return items
    
    def clear(self):
        """Clear all items from buffer."""
        with self._lock:
            self.buffer = [None] * self.size
            self.index = 0
            self.count = 0
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        with self._lock:
            return self.count == self.size
    
    def __len__(self) -> int:
        """Get number of items in buffer."""
        with self._lock:
            return self.count


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing utility functions...")
    
    # Test FPS counter
    fps_counter = FPSCounter()
    for i in range(10):
        fps_counter.update()
        time.sleep(0.01)
    print(f"FPS: {fps_counter.get_fps():.2f}")
    
    # Test geometry utils
    p1 = (0, 0)
    p2 = (3, 4)
    distance = GeometryUtils.distance_2d(p1, p2)
    print(f"Distance between {p1} and {p2}: {distance}")
    
    # Test color utils
    red_color = ColorUtils.RED
    green_color = ColorUtils.GREEN
    interpolated = ColorUtils.interpolate_color(red_color, green_color, 0.5)
    print(f"Interpolated color: {interpolated}")
    
    # Test ring buffer
    buffer = RingBuffer(5)
    for i in range(7):
        buffer.append(f"item_{i}")
    
    print(f"Buffer contents: {buffer.get_recent()}")
    print(f"Last 3 items: {buffer.get_recent(3)}")
    
    print("Utility functions test completed!")