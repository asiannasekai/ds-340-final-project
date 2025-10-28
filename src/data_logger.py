"""
Data logging system for training data collection and analysis.
Logs hand tracking data, gesture events, and MIDI messages to CSV files.
"""

import csv
import json
import time
import os
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np
import threading
import queue


@dataclass
class HandDataPoint:
    """Single hand tracking data point for logging."""
    timestamp: float
    frame_id: int
    hand_id: int
    handedness: str
    confidence: float
    landmarks: List[Dict[str, float]]  # List of {x, y, z, x_norm, y_norm}
    bbox: Dict[str, int]  # {x_min, y_min, x_max, y_max}
    board_landmarks: Optional[List[Dict[str, float]]] = None


@dataclass
class GestureDataPoint:
    """Single gesture event data point for logging."""
    timestamp: float
    frame_id: int
    hand_id: int
    gesture_type: str
    zone_name: str
    position: List[float]  # [x, y] board coordinates
    value: float
    raw_value: float
    velocity: float
    confidence: float
    smoothed_value: Optional[float] = None


@dataclass
class MidiDataPoint:
    """Single MIDI message data point for logging."""
    timestamp: float
    frame_id: int
    control_name: str
    message_type: str
    channel: int
    controller: Optional[int]
    midi_value: int
    normalized_value: float
    latency_ms: Optional[float] = None


@dataclass
class SessionMetadata:
    """Session-level metadata for logging."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    camera_config: Dict
    calibration_quality: Optional[float]
    total_frames: int
    total_gestures: int
    total_midi_messages: int
    average_fps: float
    configuration_files: Dict[str, str]


class CSVLogger:
    """
    Thread-safe CSV logger for different data types.
    """
    
    def __init__(self, filepath: str, fieldnames: List[str], 
                 buffer_size: int = 1000, auto_flush_interval: float = 5.0):
        """
        Initialize CSV logger.
        
        Args:
            filepath: Output CSV file path
            fieldnames: Column names for CSV
            buffer_size: Number of rows to buffer before writing
            auto_flush_interval: Seconds between automatic flushes
        """
        self.filepath = filepath
        self.fieldnames = fieldnames
        self.buffer_size = buffer_size
        self.auto_flush_interval = auto_flush_interval
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Data buffer
        self.buffer: List[Dict] = []
        self.buffer_lock = threading.Lock()
        
        # File writing
        self.file_handle = None
        self.csv_writer = None
        self.header_written = False
        
        # Auto-flush timer
        self.last_flush_time = time.time()
        self.auto_flush_timer = None
        
        # Statistics
        self.rows_written = 0
        self.flushes_performed = 0
        
        self._initialize_file()
        self._start_auto_flush()
    
    def _initialize_file(self):
        """Initialize CSV file and writer."""
        try:
            self.file_handle = open(self.filepath, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.DictWriter(self.file_handle, fieldnames=self.fieldnames)
            
            # Write header
            self.csv_writer.writeheader()
            self.header_written = True
            
            logging.info(f"CSV logger initialized: {self.filepath}")
            
        except Exception as e:
            logging.error(f"Failed to initialize CSV logger {self.filepath}: {e}")
            raise
    
    def _start_auto_flush(self):
        """Start auto-flush timer."""
        if self.auto_flush_interval > 0:
            self.auto_flush_timer = threading.Timer(self.auto_flush_interval, self._auto_flush)
            self.auto_flush_timer.daemon = True
            self.auto_flush_timer.start()
    
    def _auto_flush(self):
        """Automatic flush callback."""
        self.flush()
        
        # Restart timer
        if self.auto_flush_interval > 0:
            self.auto_flush_timer = threading.Timer(self.auto_flush_interval, self._auto_flush)
            self.auto_flush_timer.daemon = True
            self.auto_flush_timer.start()
    
    def log_row(self, data: Dict):
        """Add a row to the buffer."""
        with self.buffer_lock:
            self.buffer.append(data)
            
            # Auto-flush if buffer is full
            if len(self.buffer) >= self.buffer_size:
                self._flush_buffer()
    
    def log_rows(self, data_list: List[Dict]):
        """Add multiple rows to the buffer."""
        with self.buffer_lock:
            self.buffer.extend(data_list)
            
            # Auto-flush if buffer is full
            if len(self.buffer) >= self.buffer_size:
                self._flush_buffer()
    
    def flush(self):
        """Manually flush buffer to file."""
        with self.buffer_lock:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Internal flush method (assumes lock is held)."""
        if not self.buffer or not self.csv_writer:
            return
        
        try:
            # Write all buffered rows
            for row in self.buffer:
                # Ensure all required fields are present
                filtered_row = {field: row.get(field, '') for field in self.fieldnames}
                self.csv_writer.writerow(filtered_row)
            
            self.file_handle.flush()
            
            # Update statistics
            self.rows_written += len(self.buffer)
            self.flushes_performed += 1
            self.last_flush_time = time.time()
            
            # Clear buffer
            self.buffer.clear()
            
        except Exception as e:
            logging.error(f"Error flushing CSV buffer: {e}")
    
    def close(self):
        """Close logger and cleanup."""
        # Cancel auto-flush timer
        if self.auto_flush_timer:
            self.auto_flush_timer.cancel()
        
        # Flush remaining data
        self.flush()
        
        # Close file
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
            self.csv_writer = None
        
        logging.info(f"CSV logger closed: {self.filepath} ({self.rows_written} rows written)")
    
    def get_stats(self) -> Dict:
        """Get logger statistics."""
        return {
            'filepath': self.filepath,
            'rows_written': self.rows_written,
            'flushes_performed': self.flushes_performed,
            'buffer_size': len(self.buffer),
            'last_flush_age': time.time() - self.last_flush_time
        }


class DataLogger:
    """
    Main data logging system for DJ Controller.
    """
    
    def __init__(self, base_output_dir: str = "data", session_name: Optional[str] = None):
        """
        Initialize data logger.
        
        Args:
            base_output_dir: Base directory for log files
            session_name: Optional session name (defaults to timestamp)
        """
        self.base_output_dir = base_output_dir
        
        # Generate session name if not provided
        if session_name is None:
            session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        
        self.session_name = session_name
        self.session_dir = os.path.join(base_output_dir, session_name)
        
        # Create session directory
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Initialize CSV loggers
        self.hand_logger = None
        self.gesture_logger = None
        self.midi_logger = None
        
        # Session tracking
        self.session_metadata = SessionMetadata(
            session_id=session_name,
            start_time=datetime.now(),
            end_time=None,
            camera_config={},
            calibration_quality=None,
            total_frames=0,
            total_gestures=0,
            total_midi_messages=0,
            average_fps=0.0,
            configuration_files={}
        )
        
        self.frame_counter = 0
        self.start_time = time.time()
        
        # Enable/disable flags
        self.log_hand_data = True
        self.log_gesture_events = True
        self.log_midi_messages = True
        
        self._initialize_loggers()
        
        logging.info(f"DataLogger initialized for session: {session_name}")
    
    def _initialize_loggers(self):
        """Initialize CSV loggers for different data types."""
        
        # Hand tracking data logger
        if self.log_hand_data:
            hand_fieldnames = [
                'timestamp', 'frame_id', 'hand_id', 'handedness', 'confidence',
                'landmark_0_x', 'landmark_0_y', 'landmark_0_z',
                'landmark_1_x', 'landmark_1_y', 'landmark_1_z',
                'landmark_2_x', 'landmark_2_y', 'landmark_2_z',
                'landmark_3_x', 'landmark_3_y', 'landmark_3_z',
                'landmark_4_x', 'landmark_4_y', 'landmark_4_z',
                'landmark_5_x', 'landmark_5_y', 'landmark_5_z',
                'landmark_6_x', 'landmark_6_y', 'landmark_6_z',
                'landmark_7_x', 'landmark_7_y', 'landmark_7_z',
                'landmark_8_x', 'landmark_8_y', 'landmark_8_z',
                'landmark_9_x', 'landmark_9_y', 'landmark_9_z',
                'landmark_10_x', 'landmark_10_y', 'landmark_10_z',
                'landmark_11_x', 'landmark_11_y', 'landmark_11_z',
                'landmark_12_x', 'landmark_12_y', 'landmark_12_z',
                'landmark_13_x', 'landmark_13_y', 'landmark_13_z',
                'landmark_14_x', 'landmark_14_y', 'landmark_14_z',
                'landmark_15_x', 'landmark_15_y', 'landmark_15_z',
                'landmark_16_x', 'landmark_16_y', 'landmark_16_z',
                'landmark_17_x', 'landmark_17_y', 'landmark_17_z',
                'landmark_18_x', 'landmark_18_y', 'landmark_18_z',
                'landmark_19_x', 'landmark_19_y', 'landmark_19_z',
                'landmark_20_x', 'landmark_20_y', 'landmark_20_z',
                'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max',
                'board_landmark_8_x', 'board_landmark_8_y'  # Index finger tip in board coords
            ]
            
            hand_log_path = os.path.join(self.session_dir, "hand_tracking.csv")
            self.hand_logger = CSVLogger(hand_log_path, hand_fieldnames)
        
        # Gesture events logger
        if self.log_gesture_events:
            gesture_fieldnames = [
                'timestamp', 'frame_id', 'hand_id', 'gesture_type', 'zone_name',
                'position_x', 'position_y', 'value', 'raw_value', 'velocity',
                'confidence', 'smoothed_value'
            ]
            
            gesture_log_path = os.path.join(self.session_dir, "gesture_events.csv")
            self.gesture_logger = CSVLogger(gesture_log_path, gesture_fieldnames)
        
        # MIDI messages logger
        if self.log_midi_messages:
            midi_fieldnames = [
                'timestamp', 'frame_id', 'control_name', 'message_type',
                'channel', 'controller', 'midi_value', 'normalized_value', 'latency_ms'
            ]
            
            midi_log_path = os.path.join(self.session_dir, "midi_messages.csv")
            self.midi_logger = CSVLogger(midi_log_path, midi_fieldnames)
    
    def log_hand_frame(self, hand_data: List[Dict], board_hand_data: Optional[List[Dict]] = None):
        """Log hand tracking data for current frame."""
        if not self.hand_logger or not hand_data:
            return
        
        timestamp = time.time()
        self.frame_counter += 1
        
        for i, hand_info in enumerate(hand_data):
            # Create flattened row for CSV
            row = {
                'timestamp': timestamp,
                'frame_id': self.frame_counter,
                'hand_id': i,
                'handedness': hand_info.get('handedness', 'Unknown'),
                'confidence': hand_info.get('confidence', 0.0)
            }
            
            # Add landmark data
            landmarks = hand_info.get('landmarks', [])
            for j in range(21):  # MediaPipe has 21 hand landmarks
                if j < len(landmarks):
                    landmark = landmarks[j]
                    row[f'landmark_{j}_x'] = landmark.get('x_norm', 0.0)
                    row[f'landmark_{j}_y'] = landmark.get('y_norm', 0.0)
                    row[f'landmark_{j}_z'] = landmark.get('z', 0.0)
                else:
                    row[f'landmark_{j}_x'] = 0.0
                    row[f'landmark_{j}_y'] = 0.0
                    row[f'landmark_{j}_z'] = 0.0
            
            # Add bounding box
            bbox = hand_info.get('bbox', {})
            row['bbox_x_min'] = bbox.get('x_min', 0)
            row['bbox_y_min'] = bbox.get('y_min', 0)
            row['bbox_x_max'] = bbox.get('x_max', 0)
            row['bbox_y_max'] = bbox.get('y_max', 0)
            
            # Add board coordinates for index finger tip (landmark 8)
            if board_hand_data and i < len(board_hand_data):
                board_landmarks = board_hand_data[i].get('landmarks', [])
                if len(board_landmarks) > 8:
                    row['board_landmark_8_x'] = board_landmarks[8].get('x', 0.0)
                    row['board_landmark_8_y'] = board_landmarks[8].get('y', 0.0)
                else:
                    row['board_landmark_8_x'] = 0.0
                    row['board_landmark_8_y'] = 0.0
            else:
                row['board_landmark_8_x'] = 0.0
                row['board_landmark_8_y'] = 0.0
            
            self.hand_logger.log_row(row)
        
        self.session_metadata.total_frames = self.frame_counter
    
    def log_gesture_event(self, gesture_event, smoothed_value: Optional[float] = None):
        """Log a gesture event."""
        if not self.gesture_logger:
            return
        
        row = {
            'timestamp': gesture_event.timestamp,
            'frame_id': self.frame_counter,
            'hand_id': gesture_event.hand_id,
            'gesture_type': gesture_event.gesture_type.value,
            'zone_name': gesture_event.zone_name,
            'position_x': gesture_event.position[0],
            'position_y': gesture_event.position[1],
            'value': gesture_event.value,
            'raw_value': gesture_event.raw_value,
            'velocity': gesture_event.velocity,
            'confidence': gesture_event.confidence,
            'smoothed_value': smoothed_value
        }
        
        self.gesture_logger.log_row(row)
        self.session_metadata.total_gestures += 1
    
    def log_midi_message(self, control_name: str, message_type: str, 
                        channel: int, controller: Optional[int], 
                        midi_value: int, normalized_value: float,
                        latency_ms: Optional[float] = None):
        """Log a MIDI message."""
        if not self.midi_logger:
            return
        
        row = {
            'timestamp': time.time(),
            'frame_id': self.frame_counter,
            'control_name': control_name,
            'message_type': message_type,
            'channel': channel,
            'controller': controller,
            'midi_value': midi_value,
            'normalized_value': normalized_value,
            'latency_ms': latency_ms
        }
        
        self.midi_logger.log_row(row)
        self.session_metadata.total_midi_messages += 1
    
    def set_camera_config(self, config: Dict):
        """Set camera configuration metadata."""
        self.session_metadata.camera_config = config
    
    def set_calibration_quality(self, quality: float):
        """Set calibration quality metadata."""
        self.session_metadata.calibration_quality = quality
    
    def set_configuration_files(self, config_files: Dict[str, str]):
        """Set configuration file paths."""
        self.session_metadata.configuration_files = config_files
    
    def save_session_metadata(self):
        """Save session metadata to JSON file."""
        # Calculate final statistics
        total_time = time.time() - self.start_time
        self.session_metadata.average_fps = self.frame_counter / total_time if total_time > 0 else 0.0
        self.session_metadata.end_time = datetime.now()
        
        # Convert to serializable format
        metadata_dict = asdict(self.session_metadata)
        metadata_dict['start_time'] = self.session_metadata.start_time.isoformat()
        metadata_dict['end_time'] = self.session_metadata.end_time.isoformat()
        
        # Save to JSON file
        metadata_path = os.path.join(self.session_dir, "session_metadata.json")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            logging.info(f"Session metadata saved: {metadata_path}")
            
        except Exception as e:
            logging.error(f"Failed to save session metadata: {e}")
    
    def get_session_stats(self) -> Dict:
        """Get current session statistics."""
        total_time = time.time() - self.start_time
        current_fps = self.frame_counter / total_time if total_time > 0 else 0.0
        
        stats = {
            'session_name': self.session_name,
            'session_dir': self.session_dir,
            'total_frames': self.frame_counter,
            'total_gestures': self.session_metadata.total_gestures,
            'total_midi_messages': self.session_metadata.total_midi_messages,
            'session_duration': total_time,
            'average_fps': current_fps,
            'loggers': {}
        }
        
        # Add logger statistics
        if self.hand_logger:
            stats['loggers']['hand_tracking'] = self.hand_logger.get_stats()
        if self.gesture_logger:
            stats['loggers']['gesture_events'] = self.gesture_logger.get_stats()
        if self.midi_logger:
            stats['loggers']['midi_messages'] = self.midi_logger.get_stats()
        
        return stats
    
    def flush_all(self):
        """Flush all loggers."""
        if self.hand_logger:
            self.hand_logger.flush()
        if self.gesture_logger:
            self.gesture_logger.flush()
        if self.midi_logger:
            self.midi_logger.flush()
    
    def close(self):
        """Close all loggers and save metadata."""
        logging.info("Closing data logger...")
        
        # Flush and close all loggers
        if self.hand_logger:
            self.hand_logger.close()
        if self.gesture_logger:
            self.gesture_logger.close()
        if self.midi_logger:
            self.midi_logger.close()
        
        # Save session metadata
        self.save_session_metadata()
        
        logging.info(f"Data logging session completed: {self.session_name}")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Data Logger...")
    
    # Create data logger
    logger = DataLogger(session_name="test_session")
    
    try:
        # Simulate some logging
        print("Logging test data...")
        
        # Log hand tracking data
        for frame in range(10):
            hand_data = [{
                'handedness': 'Right',
                'confidence': 0.9,
                'landmarks': [{'x_norm': 0.5 + 0.1 * frame, 'y_norm': 0.5, 'z': 0.0} for _ in range(21)],
                'bbox': {'x_min': 100, 'y_min': 100, 'x_max': 200, 'y_max': 200}
            }]
            
            logger.log_hand_frame(hand_data)
            
            # Log some MIDI messages
            if frame % 3 == 0:
                logger.log_midi_message("crossfader", "control_change", 0, 8, frame * 10, frame * 0.1)
        
        # Get statistics
        stats = logger.get_session_stats()
        print(f"Session stats: {stats}")
        
        print("Data logging test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
    
    finally:
        logger.close()