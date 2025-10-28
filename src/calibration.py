"""
ArUco marker detection and camera calibration for DJ board surface mapping.
Handles homography calculation to map camera coordinates to board space.
Enhanced with DJ board detection for physical controllers and paper layouts.
"""

import cv2
import numpy as np
import json
import logging
from typing import Optional, Tuple, Dict, List
import time
from dj_board_detector import DJBoardDetector, DJBoardType


class ArUcoCalibrator:
    """
    Handles ArUco marker detection and homography calculation for board calibration.
    """
    
    def __init__(self, 
                 marker_dict_id: int = cv2.aruco.DICT_6X6_250,
                 marker_size: float = 0.05,  # 5cm markers
                 board_corners: Optional[List[Tuple[float, float]]] = None):
        """
        Initialize ArUco calibrator.
        
        Args:
            marker_dict_id: ArUco dictionary ID
            marker_size: Physical size of markers in meters
            board_corners: Real-world coordinates of board corners in meters
                          Default: [(0,0), (0.4,0), (0.4,0.3), (0,0.3)] for 40x30cm board
        """
        # ArUco setup
        self.aruco_dict = cv2.aruco.Dictionary_get(marker_dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # Improve detection parameters
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        self.aruco_params.minMarkerPerimeterRate = 0.03
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        
        # Board configuration
        self.marker_size = marker_size
        
        # Default board corners (40cm x 30cm DJ board)
        if board_corners is None:
            self.board_corners = np.array([
                [0.0, 0.0],      # Top-left (marker ID 0)
                [0.4, 0.0],      # Top-right (marker ID 1)  
                [0.4, 0.3],      # Bottom-right (marker ID 2)
                [0.0, 0.3]       # Bottom-left (marker ID 3)
            ], dtype=np.float32)
        else:
            self.board_corners = np.array(board_corners, dtype=np.float32)
        
        # Expected marker IDs for calibration
        self.expected_marker_ids = [0, 1, 2, 3]
        
        # Calibration state
        self.homography_matrix = None
        self.is_calibrated = False
        self.calibration_timestamp = None
        self.detection_quality = 0.0
        
        logging.info("ArUcoCalibrator initialized")
    
    def detect_markers(self, frame: np.ndarray) -> Tuple[List, List, List]:
        """
        Detect ArUco markers in frame.
        
        Args:
            frame: Input image
            
        Returns:
            Tuple of (corners, ids, rejected_points)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )
        
        return corners, ids, rejected
    
    def draw_markers(self, frame: np.ndarray, corners: List, ids: List) -> np.ndarray:
        """
        Draw detected markers on frame.
        
        Args:
            frame: Input image
            corners: Detected marker corners
            ids: Detected marker IDs
            
        Returns:
            Frame with markers drawn
        """
        if ids is not None and len(ids) > 0:
            # Draw markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Add marker ID labels
            for i, marker_id in enumerate(ids):
                corner = corners[i][0]
                center = np.mean(corner, axis=0).astype(int)
                
                cv2.putText(frame, f"ID:{marker_id[0]}", 
                           (center[0]-20, center[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def calculate_homography(self, corners: List, ids: List) -> bool:
        """
        Calculate homography matrix from detected markers.
        
        Args:
            corners: Detected marker corners
            ids: Detected marker IDs
            
        Returns:
            True if calibration successful
        """
        if ids is None or len(ids) < 4:
            return False
        
        # Convert ids to flat list
        ids_flat = [id_item[0] for id_item in ids]
        
        # Check if we have all expected markers
        if not all(marker_id in ids_flat for marker_id in self.expected_marker_ids):
            missing_ids = [id for id in self.expected_marker_ids if id not in ids_flat]
            logging.warning(f"Missing markers: {missing_ids}")
            return False
        
        # Extract marker centers in image coordinates
        image_points = []
        world_points = []
        
        for i, marker_id in enumerate(ids_flat):
            if marker_id in self.expected_marker_ids:
                # Get marker center
                marker_corners = corners[i][0]
                center = np.mean(marker_corners, axis=0)
                image_points.append(center)
                
                # Get corresponding world point
                world_point = self.board_corners[marker_id]
                world_points.append(world_point)
        
        if len(image_points) < 4:
            return False
        
        # Convert to numpy arrays
        image_points = np.array(image_points, dtype=np.float32)
        world_points = np.array(world_points, dtype=np.float32)
        
        # Calculate homography
        self.homography_matrix, mask = cv2.findHomography(
            image_points, world_points, 
            cv2.RANSAC, 5.0
        )
        
        if self.homography_matrix is not None:
            # Calculate quality metric (reprojection error)
            self.detection_quality = self._calculate_reprojection_error(
                image_points, world_points
            )
            
            self.is_calibrated = True
            self.calibration_timestamp = time.time()
            
            logging.info(f"Calibration successful! Quality: {self.detection_quality:.3f}")
            return True
        
        return False
    
    def _calculate_reprojection_error(self, image_points: np.ndarray, 
                                    world_points: np.ndarray) -> float:
        """Calculate average reprojection error."""
        if self.homography_matrix is None:
            return float('inf')
        
        # Project world points back to image
        projected_points = cv2.perspectiveTransform(
            world_points.reshape(-1, 1, 2), 
            np.linalg.inv(self.homography_matrix)
        ).reshape(-1, 2)
        
        # Calculate average error
        errors = np.linalg.norm(image_points - projected_points, axis=1)
        return np.mean(errors)
    
    def image_to_board_coordinates(self, image_points: np.ndarray) -> Optional[np.ndarray]:
        """
        Transform image coordinates to board coordinates.
        
        Args:
            image_points: Array of (x, y) points in image space
            
        Returns:
            Array of (x, y) points in board space, or None if not calibrated
        """
        if not self.is_calibrated or self.homography_matrix is None:
            return None
        
        # Ensure correct shape
        if image_points.ndim == 1:
            image_points = image_points.reshape(1, -1)
        
        # Transform points
        board_points = cv2.perspectiveTransform(
            image_points.reshape(-1, 1, 2).astype(np.float32),
            self.homography_matrix
        )
        
        return board_points.reshape(-1, 2)
    
    def board_to_image_coordinates(self, board_points: np.ndarray) -> Optional[np.ndarray]:
        """
        Transform board coordinates to image coordinates.
        
        Args:
            board_points: Array of (x, y) points in board space
            
        Returns:
            Array of (x, y) points in image space, or None if not calibrated
        """
        if not self.is_calibrated or self.homography_matrix is None:
            return None
        
        # Ensure correct shape
        if board_points.ndim == 1:
            board_points = board_points.reshape(1, -1)
        
        # Transform points using inverse homography
        image_points = cv2.perspectiveTransform(
            board_points.reshape(-1, 1, 2).astype(np.float32),
            np.linalg.inv(self.homography_matrix)
        )
        
        return image_points.reshape(-1, 2)
    
    def draw_board_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw board coordinate system overlay.
        
        Args:
            frame: Input image
            
        Returns:
            Frame with overlay drawn
        """
        if not self.is_calibrated:
            return frame
        
        # Draw board boundary
        board_boundary = np.array([
            [0, 0], [0.4, 0], [0.4, 0.3], [0, 0.3], [0, 0]
        ], dtype=np.float32)
        
        image_boundary = self.board_to_image_coordinates(board_boundary)
        
        if image_boundary is not None:
            pts = image_boundary.astype(np.int32)
            cv2.polylines(frame, [pts], False, (0, 255, 255), 2)
            
            # Draw coordinate axes
            origin = self.board_to_image_coordinates(np.array([[0, 0]]))[0].astype(int)
            x_axis = self.board_to_image_coordinates(np.array([[0.05, 0]]))[0].astype(int)
            y_axis = self.board_to_image_coordinates(np.array([[0, 0.05]]))[0].astype(int)
            
            cv2.arrowedLine(frame, tuple(origin), tuple(x_axis), (0, 0, 255), 3)  # X-axis (red)
            cv2.arrowedLine(frame, tuple(origin), tuple(y_axis), (0, 255, 0), 3)  # Y-axis (green)
            
            # Labels
            cv2.putText(frame, "X", tuple(x_axis + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "Y", tuple(y_axis + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def save_calibration(self, filepath: str) -> bool:
        """Save calibration data to file."""
        if not self.is_calibrated:
            return False
        
        calibration_data = {
            'homography_matrix': self.homography_matrix.tolist(),
            'board_corners': self.board_corners.tolist(),
            'detection_quality': self.detection_quality,
            'timestamp': self.calibration_timestamp,
            'marker_size': self.marker_size
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            logging.info(f"Calibration saved to {filepath}")
            return True
        except Exception as e:
            logging.error(f"Failed to save calibration: {e}")
            return False
    
    def load_calibration(self, filepath: str) -> bool:
        """Load calibration data from file."""
        try:
            with open(filepath, 'r') as f:
                calibration_data = json.load(f)
            
            self.homography_matrix = np.array(calibration_data['homography_matrix'])
            self.board_corners = np.array(calibration_data['board_corners'])
            self.detection_quality = calibration_data['detection_quality']
            self.calibration_timestamp = calibration_data['timestamp']
            self.marker_size = calibration_data['marker_size']
            self.is_calibrated = True
            
            logging.info(f"Calibration loaded from {filepath}")
            return True
        except Exception as e:
            logging.error(f"Failed to load calibration: {e}")
            return False
    
    def get_calibration_status(self) -> Dict:
        """Get current calibration status."""
        return {
            'is_calibrated': self.is_calibrated,
            'detection_quality': self.detection_quality,
            'timestamp': self.calibration_timestamp,
            'age_seconds': time.time() - self.calibration_timestamp if self.calibration_timestamp else None
        }


class CalibrationManager:
    """
    High-level manager for camera calibration workflow.
    Enhanced with DJ board detection capabilities.
    """
    
    def __init__(self, auto_save_path: Optional[str] = None, 
                 enable_dj_board_detection: bool = True):
        """
        Initialize calibration manager.
        
        Args:
            auto_save_path: Path to automatically save/load calibration
            enable_dj_board_detection: Enable DJ board detection alongside ArUco
        """
        self.calibrator = ArUcoCalibrator()
        self.auto_save_path = auto_save_path
        
        # DJ Board detection
        self.enable_dj_detection = enable_dj_board_detection
        self.dj_detector = DJBoardDetector() if enable_dj_board_detection else None
        self.calibration_source = "aruco"  # "aruco", "dj_board", or "hybrid"
        
        # Auto-load existing calibration
        if auto_save_path and self._file_exists(auto_save_path):
            self.calibrator.load_calibration(auto_save_path)
    
    def _file_exists(self, filepath: str) -> bool:
        """Check if file exists."""
        try:
            with open(filepath, 'r'):
                return True
        except FileNotFoundError:
            return False
    
    def process_frame(self, frame: np.ndarray, 
                     show_markers: bool = True,
                     show_overlay: bool = True,
                     auto_calibrate: bool = True,
                     detect_dj_boards: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Process frame for calibration with enhanced DJ board detection.
        
        Args:
            frame: Input image
            show_markers: Draw detected markers/boards
            show_overlay: Draw board overlay if calibrated
            auto_calibrate: Automatically calibrate when markers/boards detected
            detect_dj_boards: Enable DJ board detection
            
        Returns:
            Tuple of (processed_frame, calibration_info)
        """
        calibration_success = False
        calibration_info = {
            'detected_markers': 0,
            'marker_ids': [],
            'dj_board_detected': False,
            'dj_board_info': {},
            'calibration_source': self.calibration_source,
            'status': {}
        }
        
        # Try ArUco marker detection first
        corners, ids, rejected = self.calibrator.detect_markers(frame)
        
        # Draw ArUco markers if requested
        if show_markers and ids is not None:
            frame = self.calibrator.draw_markers(frame, corners, ids)
        
        # Auto-calibrate with ArUco if enabled and markers detected
        if auto_calibrate and ids is not None and len(ids) >= 4:
            if self.calibrator.calculate_homography(corners, ids):
                self.calibration_source = "aruco"
                calibration_success = True
                # Auto-save if path provided
                if self.auto_save_path:
                    self.calibrator.save_calibration(self.auto_save_path)
        
        # Try DJ board detection if enabled and ArUco not successful
        dj_detection_info = {}
        if (detect_dj_boards and self.enable_dj_detection and 
            (not calibration_success or not self.calibrator.is_calibrated)):
            
            dj_detected, dj_detection_info = self.dj_detector.detect_board(frame)
            
            if dj_detected and auto_calibrate:
                # Use DJ board detection for calibration
                self.calibration_source = "dj_board"
                
                # Copy homography from DJ detector to ArUco calibrator
                if dj_detection_info.get('homography') is not None:
                    self.calibrator.homography_matrix = dj_detection_info['homography']
                    self.calibrator.is_calibrated = True
                    self.calibrator.detection_quality = 1.0 - dj_detection_info.get('confidence', 0.5)
                    self.calibrator.calibration_timestamp = time.time()
                    calibration_success = True
        
        # Draw DJ board overlay if detected
        if show_markers and dj_detection_info.get('detected', False):
            frame = self.dj_detector.draw_detection_overlay(frame, dj_detection_info)
            if show_overlay:
                frame = self.dj_detector.draw_control_zones(frame)
        
        # Draw ArUco overlay if calibrated and requested (and not overridden by DJ board)
        elif show_overlay and self.calibrator.is_calibrated and self.calibration_source == "aruco":
            frame = self.calibrator.draw_board_overlay(frame)
        
        # Prepare calibration info
        calibration_info.update({
            'detected_markers': len(ids) if ids is not None else 0,
            'marker_ids': [id[0] for id in ids] if ids is not None else [],
            'dj_board_detected': dj_detection_info.get('detected', False),
            'dj_board_info': dj_detection_info,
            'calibration_source': self.calibration_source,
            'status': self.calibrator.get_calibration_status()
        })
        
        return frame, calibration_info
    
    def transform_points_to_board(self, image_points: np.ndarray) -> Optional[np.ndarray]:
        """Transform image points to board coordinates."""
        if self.calibration_source == "dj_board" and self.dj_detector:
            return self.dj_detector.transform_points_to_board(image_points)
        else:
            return self.calibrator.image_to_board_coordinates(image_points)
    
    def is_calibrated(self) -> bool:
        """Check if calibrated (either ArUco or DJ board detection)."""
        aruco_calibrated = self.calibrator.is_calibrated
        dj_calibrated = (self.dj_detector and self.dj_detector.is_detected) if self.enable_dj_detection else False
        return aruco_calibrated or dj_calibrated
    
    def get_quality(self) -> float:
        """Get calibration quality."""
        if self.calibration_source == "dj_board" and self.dj_detector:
            return 1.0 - self.dj_detector.detection_confidence
        else:
            return self.calibrator.detection_quality
    
    def get_control_zones(self) -> List[Dict]:
        """Get control zones from detected DJ board."""
        if self.calibration_source == "dj_board" and self.dj_detector:
            return self.dj_detector.get_control_zones_for_gestures()
        else:
            return []  # ArUco markers don't have predefined control zones
    
    def force_dj_board_detection(self, frame: np.ndarray, 
                                board_type: Optional[DJBoardType] = None) -> bool:
        """Force DJ board detection on current frame."""
        if not self.enable_dj_detection or not self.dj_detector:
            return False
        
        detected, detection_info = self.dj_detector.detect_board(frame, board_type)
        
        if detected:
            self.calibration_source = "dj_board"
            # Copy homography to ArUco calibrator for compatibility
            if detection_info.get('homography') is not None:
                self.calibrator.homography_matrix = detection_info['homography']
                self.calibrator.is_calibrated = True
                self.calibrator.detection_quality = 1.0 - detection_info.get('confidence', 0.5)
                self.calibrator.calibration_timestamp = time.time()
        
        return detected
    
    def save_dj_detection_config(self, filepath: str) -> bool:
        """Save DJ board detection configuration."""
        if self.dj_detector:
            return self.dj_detector.save_detection_config(filepath)
        return False
    
    def load_dj_detection_config(self, filepath: str) -> bool:
        """Load DJ board detection configuration."""
        if self.dj_detector:
            success = self.dj_detector.load_detection_config(filepath)
            if success:
                self.calibration_source = "dj_board"
                # Copy homography for compatibility
                if self.dj_detector.homography_matrix is not None:
                    self.calibrator.homography_matrix = self.dj_detector.homography_matrix
                    self.calibrator.is_calibrated = True
                    self.calibrator.detection_quality = 1.0 - self.dj_detector.detection_confidence
                    self.calibrator.calibration_timestamp = time.time()
            return success
        return False


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test calibration with camera
    import sys
    sys.path.append('.')
    from capture import CameraCapture
    
    camera = CameraCapture()
    calibration_manager = CalibrationManager("config/calibration.json")
    
    try:
        print("Starting calibration test...")
        print("Place ArUco markers (IDs 0,1,2,3) at board corners")
        print("Press 'c' to force calibration, 's' to save, 'l' to load, 'q' to quit")
        
        while True:
            success, frame = camera.read_frame()
            
            if not success:
                print("Failed to capture frame")
                break
            
            # Process frame
            processed_frame, calib_info = calibration_manager.process_frame(frame)
            
            # Add status text
            status = "CALIBRATED" if calibration_manager.is_calibrated() else "NOT CALIBRATED"
            quality = calibration_manager.get_quality()
            
            cv2.putText(processed_frame, f"Status: {status}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Quality: {quality:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Markers: {calib_info['detected_markers']}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('DJ Controller - Calibration', processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                print("Force calibration...")
                corners, ids, _ = calibration_manager.calibrator.detect_markers(frame)
                if calibration_manager.calibrator.calculate_homography(corners, ids):
                    print("Calibration successful!")
                else:
                    print("Calibration failed!")
            elif key == ord('s'):
                if calibration_manager.calibrator.save_calibration("config/calibration.json"):
                    print("Calibration saved!")
            elif key == ord('l'):
                if calibration_manager.calibrator.load_calibration("config/calibration.json"):
                    print("Calibration loaded!")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        camera.close()
        cv2.destroyAllWindows()
        print("Calibration test completed")