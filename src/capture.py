"""
Camera capture and MediaPipe hand tracking pipeline.
Handles webcam initialization and real-time hand keypoint detection.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Optional, Tuple, List, Dict
import logging


class HandTracker:
    """
    Manages MediaPipe hand tracking with configurable parameters.
    """
    
    def __init__(self, 
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 1):
        """
        Initialize MediaPipe hand tracking.
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
            model_complexity: Model complexity (0=lite, 1=full)
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity
        )
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        logging.info(f"HandTracker initialized with {max_num_hands} max hands")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List[Dict]]]:
        """
        Process a single frame and extract hand landmarks.
        
        Args:
            frame: Input BGR image from camera
            
        Returns:
            Tuple of (processed_frame, hand_data)
            hand_data: List of dicts with 'landmarks', 'handedness', 'bbox'
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Convert back to BGR for OpenCV
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        hand_data = []
        
        if results.multi_hand_landmarks:
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # Extract landmark coordinates
                landmarks = []
                h, w, _ = frame.shape
                
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    z = landmark.z  # Depth (relative)
                    landmarks.append({'x': x, 'y': y, 'z': z, 
                                    'x_norm': landmark.x, 'y_norm': landmark.y})
                
                # Calculate bounding box
                x_coords = [lm['x'] for lm in landmarks]
                y_coords = [lm['y'] for lm in landmarks]
                bbox = {
                    'x_min': min(x_coords),
                    'y_min': min(y_coords),
                    'x_max': max(x_coords),
                    'y_max': max(y_coords)
                }
                
                hand_info = {
                    'landmarks': landmarks,
                    'handedness': handedness.classification[0].label,
                    'confidence': handedness.classification[0].score,
                    'bbox': bbox
                }
                
                hand_data.append(hand_info)
                
                # Draw landmarks on frame
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Update FPS counter
        self._update_fps()
        
        return frame, hand_data if hand_data else None
    
    def _update_fps(self):
        """Update FPS calculation."""
        self.fps_counter += 1
        if self.fps_counter >= 30:  # Update every 30 frames
            current_time = time.time()
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def get_fps(self) -> float:
        """Get current FPS."""
        return self.current_fps
    
    def close(self):
        """Clean up MediaPipe resources."""
        self.hands.close()


class CameraCapture:
    """
    Manages camera initialization and frame capture with error handling.
    """
    
    def __init__(self, 
                 camera_id: int = 0,
                 width: int = 1280,
                 height: int = 720,
                 fps: int = 30):
        """
        Initialize camera capture.
        
        Args:
            camera_id: Camera device ID
            width: Frame width
            height: Frame height  
            fps: Target FPS
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.target_fps = fps
        self.cap = None
        self.is_opened = False
        
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize camera with specified parameters."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self.camera_id}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logging.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            self.is_opened = True
            
        except Exception as e:
            logging.error(f"Camera initialization failed: {e}")
            self.is_opened = False
            raise
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.is_opened or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if not ret:
            logging.warning("Failed to read frame from camera")
            return False, None
        
        return True, frame
    
    def get_camera_info(self) -> Dict:
        """Get camera properties."""
        if not self.is_opened:
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'codec': int(self.cap.get(cv2.CAP_PROP_FOURCC)),
            'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.cap.get(cv2.CAP_PROP_SATURATION),
            'hue': self.cap.get(cv2.CAP_PROP_HUE)
        }
    
    def close(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            logging.info("Camera released")


class CaptureManager:
    """
    High-level manager combining camera capture and hand tracking.
    """
    
    def __init__(self,
                 camera_id: int = 0,
                 camera_width: int = 1280,
                 camera_height: int = 720,
                 camera_fps: int = 30,
                 max_hands: int = 2,
                 detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.5):
        """
        Initialize capture manager.
        
        Args:
            camera_id: Camera device ID
            camera_width: Camera frame width
            camera_height: Camera frame height
            camera_fps: Target camera FPS
            max_hands: Maximum hands to track
            detection_confidence: Hand detection confidence threshold
            tracking_confidence: Hand tracking confidence threshold
        """
        # Initialize camera
        self.camera = CameraCapture(
            camera_id=camera_id,
            width=camera_width,
            height=camera_height,
            fps=camera_fps
        )
        
        # Initialize hand tracker
        self.hand_tracker = HandTracker(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        
        # Performance metrics
        self.total_frames = 0
        self.start_time = time.time()
        
        logging.info("CaptureManager initialized successfully")
    
    def get_frame_with_hands(self) -> Tuple[bool, Optional[np.ndarray], Optional[List[Dict]]]:
        """
        Capture frame and detect hands in one call.
        
        Returns:
            Tuple of (success, processed_frame, hand_data)
        """
        # Read frame from camera
        success, frame = self.camera.read_frame()
        
        if not success or frame is None:
            return False, None, None
        
        # Process frame for hand detection
        processed_frame, hand_data = self.hand_tracker.process_frame(frame)
        
        self.total_frames += 1
        
        return True, processed_frame, hand_data
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        elapsed_time = time.time() - self.start_time
        avg_fps = self.total_frames / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'total_frames': self.total_frames,
            'elapsed_time': elapsed_time,
            'average_fps': avg_fps,
            'current_fps': self.hand_tracker.get_fps(),
            'camera_info': self.camera.get_camera_info()
        }
    
    def close(self):
        """Clean up all resources."""
        self.camera.close()
        self.hand_tracker.close()
        logging.info("CaptureManager closed")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the capture manager
    capture_manager = CaptureManager()
    
    try:
        print("Starting camera capture test...")
        print("Press 'q' to quit, 's' to show stats")
        
        while True:
            success, frame, hand_data = capture_manager.get_frame_with_hands()
            
            if not success:
                print("Failed to capture frame")
                break
            
            # Add FPS display to frame
            fps = capture_manager.hand_tracker.get_fps()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add hand count
            hand_count = len(hand_data) if hand_data else 0
            cv2.putText(frame, f"Hands: {hand_count}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('DJ Controller - Hand Tracking', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                stats = capture_manager.get_performance_stats()
                print(f"Performance Stats: {stats}")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        capture_manager.close()
        cv2.destroyAllWindows()
        print("Camera test completed")