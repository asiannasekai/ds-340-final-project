"""
Main application for Computer Vision DJ Controller.
Combines all modules into a real-time DJ control interface.
"""

import cv2
import numpy as np
import time
import logging
import argparse
import json
import sys
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict
import threading

# Import our modules
from capture import CaptureManager
from calibration import CalibrationManager
from gestures import GestureRecognizer, GestureEvent, GestureType
from smoothing import MultiChannelSmoother, SmoothingConfig, SmoothingMethod
from midi_controller import DJMidiController


class DJControllerApp:
    """
    Main DJ Controller application class.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize DJ Controller application.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()
        
        # Initialize components
        self.capture_manager = None
        self.calibration_manager = None
        self.gesture_recognizer = None
        self.smoother = None
        self.midi_controller = None
        
        # Application state
        self.running = False
        self.paused = False
        self.show_debug = True
        self.show_zones = True
        self.show_hands = True
        self.show_overlay = True
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
        
        # UI state
        self.window_name = "DJ Controller - Computer Vision Interface"
        self.window_size = (1280, 720)
        
        # Data logging
        self.log_data = False
        self.logged_frames = 0
        
        logging.info("DJControllerApp initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from file."""
        default_config = {
            "camera": {
                "device_id": 0,
                "width": 1280,
                "height": 720,
                "fps": 30
            },
            "hand_tracking": {
                "max_hands": 2,
                "detection_confidence": 0.7,
                "tracking_confidence": 0.5
            },
            "calibration": {
                "auto_save_path": "config/calibration.json",
                "marker_size": 0.05,
                "enable_dj_board_detection": True,
                "dj_detection_config_path": "config/dj_board_detection.json"
            },
            "gestures": {
                "config_path": "config/zones.json"
            },
            "smoothing": {
                "method": "EMA",
                "alpha": 0.3,
                "hysteresis_threshold": 0.01
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
                "output_path": "data/session_log.csv"
            }
        }
        
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                
                # Merge with default config
                self._deep_update(default_config, loaded_config)
                logging.info(f"Configuration loaded from {self.config_file}")
                
            except Exception as e:
                logging.error(f"Failed to load config file: {e}")
        
        return default_config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def initialize_components(self) -> bool:
        """Initialize all system components."""
        try:
            # Initialize camera capture
            camera_config = self.config["camera"]
            hand_config = self.config["hand_tracking"]
            
            self.capture_manager = CaptureManager(
                camera_id=camera_config["device_id"],
                camera_width=camera_config["width"],
                camera_height=camera_config["height"],
                camera_fps=camera_config["fps"],
                max_hands=hand_config["max_hands"],
                detection_confidence=hand_config["detection_confidence"],
                tracking_confidence=hand_config["tracking_confidence"]
            )
            
            if not self.capture_manager.camera.is_opened:
                logging.error("Failed to initialize camera")
                return False
            
            # Initialize calibration
            calib_config = self.config["calibration"]
            self.calibration_manager = CalibrationManager(
                auto_save_path=calib_config["auto_save_path"],
                enable_dj_board_detection=calib_config.get("enable_dj_board_detection", True)
            )
            
            # Load DJ board detection config if available
            dj_config_path = calib_config.get("dj_detection_config_path")
            if dj_config_path:
                self.calibration_manager.load_dj_detection_config(dj_config_path)
            
            # Initialize gesture recognition
            gesture_config = self.config["gestures"]
            self.gesture_recognizer = GestureRecognizer(
                config_path=gesture_config.get("config_path")
            )
            
            # Initialize smoothing
            smooth_config = self.config["smoothing"]
            smoothing_cfg = SmoothingConfig(
                method=SmoothingMethod(smooth_config["method"]),
                alpha=smooth_config["alpha"],
                hysteresis_threshold=smooth_config["hysteresis_threshold"]
            )
            self.smoother = MultiChannelSmoother(default_config=smoothing_cfg)
            
            # Initialize MIDI controller
            midi_config = self.config["midi"]
            self.midi_controller = DJMidiController(
                port_name=midi_config["port_name"],
                config_file=midi_config.get("config_path")
            )
            
            if not self.midi_controller.midi_out.is_connected:
                logging.warning("MIDI controller not connected - continuing without MIDI")
            
            logging.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize components: {e}")
            return False
    
    def run(self):
        """Main application loop."""
        if not self.initialize_components():
            logging.error("Failed to initialize components")
            return
        
        self.running = True
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_RESIZABLE)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        
        # Set up keyboard callbacks
        self._setup_ui()
        
        logging.info("Starting main loop...")
        
        try:
            while self.running:
                if not self.paused:
                    success = self._process_frame()
                    if not success:
                        logging.error("Frame processing failed")
                        break
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_keyboard(key):
                    break
                
                # Update FPS counter
                self._update_fps()
        
        except KeyboardInterrupt:
            logging.info("Application interrupted by user")
        
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}")
        
        finally:
            self._cleanup()
    
    def _process_frame(self) -> bool:
        """Process a single frame."""
        timestamp = time.time()
        
        # Capture frame and detect hands
        success, frame, hand_data = self.capture_manager.get_frame_with_hands()
        
        if not success or frame is None:
            return False
        
        # Process calibration
        frame, calib_info = self.calibration_manager.process_frame(
            frame, 
            show_markers=self.show_debug,
            show_overlay=self.show_overlay,
            auto_calibrate=True
        )
        
        # Transform hand positions to board coordinates if calibrated
        board_hand_data = None
        if hand_data and self.calibration_manager.is_calibrated():
            board_hand_data = self._transform_hand_data(hand_data)
            
            # Update control zones from calibration if using DJ board detection
            if (calib_info.get('calibration_source') == 'dj_board' and 
                calib_info.get('dj_board_detected')):
                self.gesture_recognizer.update_zones_from_calibration(self.calibration_manager)
            
            # Update gesture recognition
            self.gesture_recognizer.update_hand_states(board_hand_data, timestamp)
            gesture_events = self.gesture_recognizer.recognize_gestures(timestamp)
            
            # Process gesture events
            self._process_gesture_events(gesture_events, timestamp)
        
        # Draw visualization
        frame = self._draw_visualization(frame, hand_data, board_hand_data, calib_info)
        
        # Display frame
        cv2.imshow(self.window_name, frame)
        
        self.frame_count += 1
        return True
    
    def _transform_hand_data(self, hand_data: List[Dict]) -> List[Dict]:
        """Transform hand data to board coordinates."""
        board_hand_data = []
        
        for hand_info in hand_data:
            # Transform landmark positions
            landmarks = hand_info['landmarks']
            board_landmarks = []
            
            for landmark in landmarks:
                image_point = np.array([[landmark['x_norm'], landmark['y_norm']]], dtype=np.float32)
                board_point = self.calibration_manager.transform_points_to_board(image_point)
                
                if board_point is not None:
                    board_landmarks.append({
                        'x': board_point[0][0],
                        'y': board_point[0][1],
                        'z': landmark['z'],
                        'x_norm': landmark['x_norm'],
                        'y_norm': landmark['y_norm']
                    })
                else:
                    board_landmarks.append(landmark)  # Keep original if transform fails
            
            board_hand_info = hand_info.copy()
            board_hand_info['landmarks'] = board_landmarks
            board_hand_data.append(board_hand_info)
        
        return board_hand_data
    
    def _process_gesture_events(self, events: List[GestureEvent], timestamp: float):
        """Process gesture events and send MIDI messages."""
        for event in events:
            # Apply smoothing to continuous controls
            if event.gesture_type in [GestureType.DRAG, GestureType.ROTATE]:
                smoothed_value = self.smoother.update_channel(
                    event.zone_name, event.value, timestamp
                )
            else:
                smoothed_value = event.value
            
            # Send MIDI message
            if self.midi_controller and self.midi_controller.midi_out.is_connected:
                if event.gesture_type == GestureType.TAP:
                    self.midi_controller.trigger_button(event.zone_name, True)
                    # Schedule button release
                    threading.Timer(0.1, lambda: self.midi_controller.trigger_button(event.zone_name, False)).start()
                else:
                    self.midi_controller.update_control(event.zone_name, smoothed_value)
            
            # Log event if enabled
            if self.log_data:
                self._log_gesture_event(event, smoothed_value, timestamp)
    
    def _draw_visualization(self, frame: np.ndarray, hand_data: Optional[List[Dict]], 
                          board_hand_data: Optional[List[Dict]], calib_info: Dict) -> np.ndarray:
        """Draw visualization overlay on frame."""
        
        # Draw control zones if calibrated
        if self.show_zones and self.calibration_manager.is_calibrated():
            frame = self.gesture_recognizer.draw_zones(frame, self.calibration_manager)
        
        # Draw hand information
        if self.show_hands and hand_data:
            self._draw_hand_info(frame, hand_data)
        
        # Draw debug information
        if self.show_debug:
            self._draw_debug_info(frame, calib_info)
        
        return frame
    
    def _draw_hand_info(self, frame: np.ndarray, hand_data: List[Dict]):
        """Draw hand information overlay."""
        for i, hand_info in enumerate(hand_data):
            # Draw hand ID and confidence
            bbox = hand_info['bbox']
            label = f"Hand {i+1} ({hand_info['handedness']}) - {hand_info['confidence']:.2f}"
            
            cv2.putText(frame, label, 
                       (bbox['x_min'], bbox['y_min'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw bounding box
            cv2.rectangle(frame, 
                         (bbox['x_min'], bbox['y_min']),
                         (bbox['x_max'], bbox['y_max']),
                         (0, 255, 0), 1)
    
    def _draw_debug_info(self, frame: np.ndarray, calib_info: Dict):
        """Draw debug information overlay."""
        height, width = frame.shape[:2]
        
        # FPS and performance info
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Calibration status
        calib_status = "CALIBRATED" if self.calibration_manager.is_calibrated() else "NOT CALIBRATED"
        calib_color = (0, 255, 0) if self.calibration_manager.is_calibrated() else (0, 0, 255)
        cv2.putText(frame, f"Calibration: {calib_status}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, calib_color, 2)
        
        # Detected markers and boards
        marker_text = f"Markers: {calib_info['detected_markers']}"
        cv2.putText(frame, marker_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # DJ board detection status
        dj_detected = calib_info.get('dj_board_detected', False)
        dj_text = f"DJ Board: {'DETECTED' if dj_detected else 'NOT DETECTED'}"
        dj_color = (0, 255, 0) if dj_detected else (128, 128, 128)
        cv2.putText(frame, dj_text, (10, 115), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, dj_color, 1)
        
        # Calibration source
        source = calib_info.get('calibration_source', 'unknown')
        cv2.putText(frame, f"Source: {source}", (10, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # MIDI status
        midi_status = "CONNECTED" if (self.midi_controller and self.midi_controller.midi_out.is_connected) else "DISCONNECTED"
        midi_color = (0, 255, 0) if midi_status == "CONNECTED" else (0, 0, 255)
        cv2.putText(frame, f"MIDI: {midi_status}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, midi_color, 2)
        
        # Control instructions
        instructions = [
            "Controls:",
            "Q - Quit",
            "SPACE - Pause/Resume",
            "D - Toggle Debug",
            "Z - Toggle Zones", 
            "H - Toggle Hands",
            "O - Toggle Overlay",
            "C - Force Calibration",
            "B - Detect DJ Board",
            "R - Reset MIDI",
            "L - Toggle Logging"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = height - (len(instructions) - i) * 25
            cv2.putText(frame, instruction, (width - 300, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Performance stats (if available)
        if self.midi_controller:
            stats = self.midi_controller.get_performance_stats()
            queue_size = stats.get('queue_size', 0)
            cv2.putText(frame, f"MIDI Queue: {queue_size}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _handle_keyboard(self, key: int) -> bool:
        """Handle keyboard input."""
        if key == ord('q') or key == 27:  # Q or ESC
            return False
        
        elif key == ord(' '):  # SPACE - pause/resume
            self.paused = not self.paused
            logging.info(f"Application {'paused' if self.paused else 'resumed'}")
        
        elif key == ord('d'):  # D - toggle debug
            self.show_debug = not self.show_debug
        
        elif key == ord('z'):  # Z - toggle zones
            self.show_zones = not self.show_zones
        
        elif key == ord('h'):  # H - toggle hands
            self.show_hands = not self.show_hands
        
        elif key == ord('o'):  # O - toggle overlay
            self.show_overlay = not self.show_overlay
        
        elif key == ord('c'):  # C - force calibration
            if hasattr(self, 'capture_manager') and self.capture_manager:
                success, frame, _ = self.capture_manager.get_frame_with_hands()
                if success and frame is not None:
                    # Try ArUco first
                    corners, ids, _ = self.calibration_manager.calibrator.detect_markers(frame)
                    if self.calibration_manager.calibrator.calculate_homography(corners, ids):
                        logging.info("Manual ArUco calibration successful")
                    else:
                        # Try DJ board detection
                        if self.calibration_manager.force_dj_board_detection(frame):
                            logging.info("Manual DJ board detection successful")
                        else:
                            logging.warning("Manual calibration failed - no markers or DJ boards detected")
        
        elif key == ord('r'):  # R - reset MIDI
            if self.midi_controller:
                self.midi_controller.reset_all_controls()
                logging.info("MIDI controls reset")
        
        elif key == ord('l'):  # L - toggle logging
            self.log_data = not self.log_data
            logging.info(f"Data logging {'enabled' if self.log_data else 'disabled'}")
        
        elif key == ord('b'):  # B - force DJ board detection
            if hasattr(self, 'capture_manager') and self.capture_manager:
                success, frame, _ = self.capture_manager.get_frame_with_hands()
                if success and frame is not None:
                    if self.calibration_manager.force_dj_board_detection(frame):
                        logging.info("DJ board detection successful")
                        # Auto-save detection config
                        dj_config_path = self.config["calibration"].get("dj_detection_config_path")
                        if dj_config_path:
                            self.calibration_manager.save_dj_detection_config(dj_config_path)
                    else:
                        logging.warning("DJ board detection failed")
        
        elif key == ord('s'):  # S - save configuration
            self._save_current_config()
        
        return True
    
    def _log_gesture_event(self, event: GestureEvent, smoothed_value: float, timestamp: float):
        """Log gesture event to file."""
        # This is a placeholder for data logging functionality
        self.logged_frames += 1
    
    def _save_current_config(self):
        """Save current configuration to file."""
        try:
            config_path = self.config_file or "config/app_config.json"
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logging.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")
    
    def _setup_ui(self):
        """Set up UI elements and callbacks."""
        # This could be extended with trackbars or other UI elements
        pass
    
    def _update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        if self.fps_counter >= 30:  # Update every 30 frames
            current_time = time.time()
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def _cleanup(self):
        """Clean up resources."""
        logging.info("Cleaning up resources...")
        
        if self.midi_controller:
            self.midi_controller.close()
        
        if self.capture_manager:
            self.capture_manager.close()
        
        cv2.destroyAllWindows()
        
        # Print final statistics
        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time if total_time > 0 else 0
        
        logging.info(f"Session complete:")
        logging.info(f"  Total frames: {self.frame_count}")
        logging.info(f"  Total time: {total_time:.2f}s")
        logging.info(f"  Average FPS: {avg_fps:.2f}")
        
        if self.log_data:
            logging.info(f"  Logged frames: {self.logged_frames}")


def create_default_config():
    """Create default configuration files."""
    # Create directories
    os.makedirs("config", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Create default app config
    app_config = {
        "camera": {
            "device_id": 0,
            "width": 1280,
            "height": 720,
            "fps": 30
        },
        "hand_tracking": {
            "max_hands": 2,
            "detection_confidence": 0.7,
            "tracking_confidence": 0.5
        },
        "calibration": {
            "auto_save_path": "config/calibration.json",
            "marker_size": 0.05,
            "enable_dj_board_detection": True,
            "dj_detection_config_path": "config/dj_board_detection.json"
        },
        "gestures": {
            "config_path": "config/zones.json"
        },
        "smoothing": {
            "method": "EMA",
            "alpha": 0.3,
            "hysteresis_threshold": 0.01
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
            "output_path": "data/session_log.csv"
        }
    }
    
    with open("config/app_config.json", 'w') as f:
        json.dump(app_config, f, indent=2)
    
    print("Default configuration created in config/app_config.json")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Computer Vision DJ Controller")
    parser.add_argument("--config", "-c", type=str, default="config/app_config.json",
                       help="Configuration file path")
    parser.add_argument("--create-config", action="store_true",
                       help="Create default configuration files")
    parser.add_argument("--debug", "-d", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--no-midi", action="store_true",
                       help="Disable MIDI output")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("dj_controller.log")
        ]
    )
    
    # Create default config if requested
    if args.create_config:
        create_default_config()
        return
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Configuration file {args.config} not found.")
        print("Run with --create-config to create default configuration.")
        return
    
    try:
        # Create and run application
        app = DJControllerApp(config_file=args.config)
        app.run()
        
    except Exception as e:
        logging.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()