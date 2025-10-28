"""
DJ Board Detection System - Detects traditional DJ controllers and paper layouts.
Supports both physical DJ hardware and printed paper templates.
"""

import cv2
import numpy as np
import json
import logging
from typing import Optional, Tuple, Dict, List, Any
from enum import Enum
import time
from dataclasses import dataclass


class DJBoardType(Enum):
    """Types of DJ boards that can be detected."""
    PHYSICAL_CONTROLLER = "physical_controller"
    PAPER_TEMPLATE = "paper_template"
    ARUCO_MARKERS = "aruco_markers"
    AUTO_DETECT = "auto_detect"


class DetectionMethod(Enum):
    """Detection methods for different board types."""
    CONTOUR_DETECTION = "contour_detection"
    TEMPLATE_MATCHING = "template_matching"
    CORNER_DETECTION = "corner_detection"
    EDGE_DETECTION = "edge_detection"
    COLOR_SEGMENTATION = "color_segmentation"
    FEATURE_MATCHING = "feature_matching"


@dataclass
class DJBoardLayout:
    """Standard DJ board layout configuration."""
    name: str
    board_type: DJBoardType
    expected_dimensions: Tuple[float, float]  # Width x Height in cm
    control_zones: List[Dict[str, Any]]
    reference_points: List[Tuple[float, float]]  # Key reference points for detection
    detection_config: Dict[str, Any]


class DJBoardDetector:
    """
    Detects various types of DJ boards and controllers.
    """
    
    def __init__(self):
        """Initialize DJ board detector."""
        # Detection parameters
        self.detection_method = DetectionMethod.AUTO_DETECT
        self.board_type = DJBoardType.AUTO_DETECT
        
        # Detection state
        self.is_detected = False
        self.detected_board = None
        self.detection_confidence = 0.0
        self.homography_matrix = None
        self.board_corners = None
        
        # Load predefined layouts
        self.board_layouts = self._load_standard_layouts()
        
        # Detection parameters
        self.min_contour_area = 5000
        self.max_contour_area = 100000
        self.corner_detection_quality = 0.01
        self.corner_detection_min_distance = 50
        
        # Template matching parameters
        self.template_threshold = 0.7
        self.template_images = {}
        
        # Feature matching parameters  
        self.feature_detector = cv2.SIFT_create()
        self.feature_matcher = cv2.BFMatcher()
        self.min_match_count = 10
        
        logging.info("DJ Board Detector initialized")
    
    def _load_standard_layouts(self) -> Dict[str, DJBoardLayout]:
        """Load standard DJ board layouts."""
        layouts = {}
        
        # Pioneer DDJ-SB3 style layout
        layouts["pioneer_ddj_sb3"] = DJBoardLayout(
            name="Pioneer DDJ-SB3",
            board_type=DJBoardType.PHYSICAL_CONTROLLER,
            expected_dimensions=(48.3, 27.2),  # cm
            control_zones=[
                # Left deck
                {"name": "jog_left", "type": "circular", "center": [12.0, 8.0], "radius": 4.5},
                {"name": "volume_left", "type": "fader", "bounds": [[2.0, 15.0], [3.0, 22.0]]},
                {"name": "eq_high_left", "type": "knob", "center": [4.5, 12.0], "radius": 1.0},
                {"name": "eq_mid_left", "type": "knob", "center": [4.5, 15.0], "radius": 1.0},
                {"name": "eq_low_left", "type": "knob", "center": [4.5, 18.0], "radius": 1.0},
                {"name": "play_left", "type": "button", "center": [8.0, 20.0], "radius": 1.5},
                {"name": "cue_left", "type": "button", "center": [12.0, 20.0], "radius": 1.5},
                
                # Right deck  
                {"name": "jog_right", "type": "circular", "center": [36.3, 8.0], "radius": 4.5},
                {"name": "volume_right", "type": "fader", "bounds": [[45.3, 15.0], [46.3, 22.0]]},
                {"name": "eq_high_right", "type": "knob", "center": [43.8, 12.0], "radius": 1.0},
                {"name": "eq_mid_right", "type": "knob", "center": [43.8, 15.0], "radius": 1.0},
                {"name": "eq_low_right", "type": "knob", "center": [43.8, 18.0], "radius": 1.0},
                {"name": "play_right", "type": "button", "center": [40.3, 20.0], "radius": 1.5},
                {"name": "cue_right", "type": "button", "center": [36.3, 20.0], "radius": 1.5},
                
                # Center
                {"name": "crossfader", "type": "fader", "bounds": [[20.0, 23.0], [28.3, 24.0]]},
                {"name": "browse", "type": "knob", "center": [24.15, 15.0], "radius": 1.2},
            ],
            reference_points=[
                (0.0, 0.0), (48.3, 0.0), (48.3, 27.2), (0.0, 27.2)  # Corners
            ],
            detection_config={
                "edge_threshold_low": 50,
                "edge_threshold_high": 150,
                "corner_quality": 0.01,
                "corner_min_distance": 50,
                "contour_approx_epsilon": 0.02
            }
        )
        
        # Numark Party Mix style layout
        layouts["numark_party_mix"] = DJBoardLayout(
            name="Numark Party Mix",
            board_type=DJBoardType.PHYSICAL_CONTROLLER,
            expected_dimensions=(31.0, 19.5),  # cm
            control_zones=[
                # Left deck
                {"name": "jog_left", "type": "circular", "center": [7.5, 6.0], "radius": 3.0},
                {"name": "volume_left", "type": "fader", "bounds": [[2.0, 10.0], [3.0, 16.0]]},
                {"name": "gain_left", "type": "knob", "center": [4.0, 8.0], "radius": 0.8},
                {"name": "eq_high_left", "type": "knob", "center": [6.0, 10.0], "radius": 0.8},
                {"name": "eq_low_left", "type": "knob", "center": [9.0, 10.0], "radius": 0.8},
                {"name": "play_left", "type": "button", "center": [6.0, 13.5], "radius": 1.0},
                {"name": "cue_left", "type": "button", "center": [9.0, 13.5], "radius": 1.0},
                
                # Right deck
                {"name": "jog_right", "type": "circular", "center": [23.5, 6.0], "radius": 3.0},
                {"name": "volume_right", "type": "fader", "bounds": [[28.0, 10.0], [29.0, 16.0]]},
                {"name": "gain_right", "type": "knob", "center": [27.0, 8.0], "radius": 0.8},
                {"name": "eq_high_right", "type": "knob", "center": [25.0, 10.0], "radius": 0.8},
                {"name": "eq_low_right", "type": "knob", "center": [22.0, 10.0], "radius": 0.8},
                {"name": "play_right", "type": "button", "center": [25.0, 13.5], "radius": 1.0},
                {"name": "cue_right", "type": "button", "center": [22.0, 13.5], "radius": 1.0},
                
                # Center
                {"name": "crossfader", "type": "fader", "bounds": [[12.0, 17.0], [19.0, 17.8]]},
                {"name": "master_gain", "type": "knob", "center": [15.5, 14.0], "radius": 0.8},
            ],
            reference_points=[
                (0.0, 0.0), (31.0, 0.0), (31.0, 19.5), (0.0, 19.5)  # Corners
            ],
            detection_config={
                "edge_threshold_low": 40,
                "edge_threshold_high": 120,
                "corner_quality": 0.02,
                "corner_min_distance": 40,
                "contour_approx_epsilon": 0.025
            }
        )
        
        # Paper template layout - Generic 2-deck
        layouts["paper_template_2deck"] = DJBoardLayout(
            name="Paper Template 2-Deck",
            board_type=DJBoardType.PAPER_TEMPLATE,
            expected_dimensions=(40.0, 30.0),  # cm - A3 size
            control_zones=[
                # Left deck
                {"name": "jog_left", "type": "circular", "center": [10.0, 8.0], "radius": 4.0},
                {"name": "volume_left", "type": "fader", "bounds": [[3.0, 15.0], [4.0, 25.0]]},
                {"name": "eq_high_left", "type": "knob", "center": [6.0, 12.0], "radius": 1.2},
                {"name": "eq_mid_left", "type": "knob", "center": [6.0, 15.0], "radius": 1.2},
                {"name": "eq_low_left", "type": "knob", "center": [6.0, 18.0], "radius": 1.2},
                {"name": "play_left", "type": "button", "center": [8.0, 22.0], "radius": 1.5},
                {"name": "cue_left", "type": "button", "center": [12.0, 22.0], "radius": 1.5},
                {"name": "sync_left", "type": "button", "center": [10.0, 25.0], "radius": 1.0},
                
                # Right deck
                {"name": "jog_right", "type": "circular", "center": [30.0, 8.0], "radius": 4.0},
                {"name": "volume_right", "type": "fader", "bounds": [[36.0, 15.0], [37.0, 25.0]]},
                {"name": "eq_high_right", "type": "knob", "center": [34.0, 12.0], "radius": 1.2},
                {"name": "eq_mid_right", "type": "knob", "center": [34.0, 15.0], "radius": 1.2},
                {"name": "eq_low_right", "type": "knob", "center": [34.0, 18.0], "radius": 1.2},
                {"name": "play_right", "type": "button", "center": [32.0, 22.0], "radius": 1.5},
                {"name": "cue_right", "type": "button", "center": [28.0, 22.0], "radius": 1.5},
                {"name": "sync_right", "type": "button", "center": [30.0, 25.0], "radius": 1.0},
                
                # Center
                {"name": "crossfader", "type": "fader", "bounds": [[16.0, 26.0], [24.0, 27.0]]},
                {"name": "browse_knob", "type": "knob", "center": [20.0, 15.0], "radius": 1.5},
                {"name": "master_gain", "type": "fader", "bounds": [[38.5, 5.0], [39.0, 12.0]]},
                {"name": "headphone_gain", "type": "knob", "center": [20.0, 5.0], "radius": 1.0},
            ],
            reference_points=[
                (0.0, 0.0), (40.0, 0.0), (40.0, 30.0), (0.0, 30.0)  # Corners
            ],
            detection_config={
                "edge_threshold_low": 100,
                "edge_threshold_high": 200,
                "corner_quality": 0.05,
                "corner_min_distance": 30,
                "contour_approx_epsilon": 0.01,
                "paper_color_range": {
                    "white_low": [200, 200, 200],
                    "white_high": [255, 255, 255]
                }
            }
        )
        
        return layouts
    
    def detect_board(self, frame: np.ndarray, 
                    board_type: Optional[DJBoardType] = None,
                    layout_name: Optional[str] = None) -> Tuple[bool, Dict]:
        """
        Detect DJ board in frame.
        
        Args:
            frame: Input image
            board_type: Specific board type to detect
            layout_name: Specific layout name to use
            
        Returns:
            Tuple of (success, detection_info)
        """
        detection_info = {
            'detected': False,
            'board_type': None,
            'layout_name': None,
            'confidence': 0.0,
            'corners': None,
            'method': None,
            'homography': None
        }
        
        # Auto-detect if no specific type given
        if board_type is None:
            board_type = DJBoardType.AUTO_DETECT
        
        # Try different detection methods based on board type
        if board_type == DJBoardType.AUTO_DETECT:
            # Try all detection methods
            success, info = self._auto_detect(frame)
            if success:
                detection_info.update(info)
        
        elif board_type == DJBoardType.PHYSICAL_CONTROLLER:
            success, info = self._detect_physical_controller(frame, layout_name)
            if success:
                detection_info.update(info)
        
        elif board_type == DJBoardType.PAPER_TEMPLATE:
            success, info = self._detect_paper_template(frame, layout_name)
            if success:
                detection_info.update(info)
        
        # Update internal state if detection successful
        if detection_info['detected']:
            self.is_detected = True
            self.detected_board = detection_info['layout_name']
            self.detection_confidence = detection_info['confidence']
            self.homography_matrix = detection_info['homography']
            self.board_corners = detection_info['corners']
        else:
            self.is_detected = False
        
        return detection_info['detected'], detection_info
    
    def _auto_detect(self, frame: np.ndarray) -> Tuple[bool, Dict]:
        """Auto-detect board type and layout."""
        # Try paper template first (usually easier to detect)
        success, info = self._detect_paper_template(frame)
        if success and info['confidence'] > 0.6:
            return True, info
        
        # Try physical controllers
        success, info = self._detect_physical_controller(frame)
        if success and info['confidence'] > 0.5:
            return True, info
        
        return False, {}
    
    def _detect_physical_controller(self, frame: np.ndarray, 
                                  layout_name: Optional[str] = None) -> Tuple[bool, Dict]:
        """Detect physical DJ controllers."""
        # If specific layout requested, try that first
        if layout_name and layout_name in self.board_layouts:
            layouts_to_try = [layout_name]
        else:
            # Try all physical controller layouts
            layouts_to_try = [name for name, layout in self.board_layouts.items() 
                            if layout.board_type == DJBoardType.PHYSICAL_CONTROLLER]
        
        best_detection = None
        best_confidence = 0.0
        
        for layout_name in layouts_to_try:
            layout = self.board_layouts[layout_name]
            
            # Try multiple detection methods
            methods = [
                self._detect_by_contours,
                self._detect_by_corners,
                self._detect_by_edges
            ]
            
            for method in methods:
                success, info = method(frame, layout)
                if success and info['confidence'] > best_confidence:
                    best_confidence = info['confidence']
                    best_detection = info
                    best_detection['layout_name'] = layout_name
                    best_detection['board_type'] = DJBoardType.PHYSICAL_CONTROLLER
        
        if best_detection and best_confidence > 0.3:
            return True, best_detection
        
        return False, {}
    
    def _detect_paper_template(self, frame: np.ndarray,
                             layout_name: Optional[str] = None) -> Tuple[bool, Dict]:
        """Detect paper template layouts."""
        # If specific layout requested, try that first
        if layout_name and layout_name in self.board_layouts:
            layouts_to_try = [layout_name]
        else:
            # Try all paper template layouts
            layouts_to_try = [name for name, layout in self.board_layouts.items() 
                            if layout.board_type == DJBoardType.PAPER_TEMPLATE]
        
        best_detection = None
        best_confidence = 0.0
        
        for layout_name in layouts_to_try:
            layout = self.board_layouts[layout_name]
            
            # Paper-specific detection methods
            methods = [
                self._detect_paper_by_color,
                self._detect_by_contours,
                self._detect_by_corners
            ]
            
            for method in methods:
                success, info = method(frame, layout)
                if success and info['confidence'] > best_confidence:
                    best_confidence = info['confidence']
                    best_detection = info
                    best_detection['layout_name'] = layout_name
                    best_detection['board_type'] = DJBoardType.PAPER_TEMPLATE
        
        if best_detection and best_confidence > 0.4:
            return True, best_detection
        
        return False, {}
    
    def _detect_by_contours(self, frame: np.ndarray, layout: DJBoardLayout) -> Tuple[bool, Dict]:
        """Detect board using contour detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        config = layout.detection_config
        edges = cv2.Canny(blurred, 
                         config.get('edge_threshold_low', 50),
                         config.get('edge_threshold_high', 150))
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_contour_area or area > self.max_contour_area:
                continue
            
            # Approximate contour to polygon
            epsilon = config.get('contour_approx_epsilon', 0.02) * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Look for rectangular shapes (4 corners)
            if len(approx) == 4:
                # Check if it's roughly rectangular
                corners = approx.reshape(-1, 2).astype(np.float32)
                
                # Calculate aspect ratio
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                aspect_ratio = max(width, height) / min(width, height)
                
                # Expected aspect ratio for this layout
                expected_w, expected_h = layout.expected_dimensions
                expected_ratio = expected_w / expected_h
                
                # Check if aspect ratio matches (within tolerance)
                ratio_diff = abs(aspect_ratio - expected_ratio) / expected_ratio
                if ratio_diff < 0.3:  # 30% tolerance
                    # Calculate homography
                    board_corners = np.array(layout.reference_points, dtype=np.float32)
                    homography, _ = cv2.findHomography(corners, board_corners)
                    
                    confidence = 1.0 - ratio_diff  # Higher confidence for better aspect ratio match
                    
                    return True, {
                        'detected': True,
                        'confidence': confidence,
                        'corners': corners,
                        'method': DetectionMethod.CONTOUR_DETECTION,
                        'homography': homography,
                        'area': area
                    }
        
        return False, {}
    
    def _detect_by_corners(self, frame: np.ndarray, layout: DJBoardLayout) -> Tuple[bool, Dict]:
        """Detect board using corner detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        config = layout.detection_config
        
        # Harris corner detection
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=100,
            qualityLevel=config.get('corner_quality', 0.01),
            minDistance=config.get('corner_min_distance', 50)
        )
        
        if corners is None or len(corners) < 4:
            return False, {}
        
        corners = corners.reshape(-1, 2)
        
        # Try to find rectangular patterns
        # This is a simplified approach - you could use more sophisticated algorithms
        hull = cv2.convexHull(corners.astype(np.int32))
        
        if len(hull) >= 4:
            # Approximate to rectangle
            epsilon = 0.02 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            
            if len(approx) == 4:
                rect_corners = approx.reshape(-1, 2).astype(np.float32)
                
                # Calculate homography
                board_corners = np.array(layout.reference_points, dtype=np.float32)
                homography, _ = cv2.findHomography(rect_corners, board_corners)
                
                confidence = 0.6  # Medium confidence for corner detection
                
                return True, {
                    'detected': True,
                    'confidence': confidence,
                    'corners': rect_corners,
                    'method': DetectionMethod.CORNER_DETECTION,
                    'homography': homography
                }
        
        return False, {}
    
    def _detect_by_edges(self, frame: np.ndarray, layout: DJBoardLayout) -> Tuple[bool, Dict]:
        """Detect board using edge detection and Hough transforms."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        config = layout.detection_config
        
        # Edge detection
        edges = cv2.Canny(gray,
                         config.get('edge_threshold_low', 50),
                         config.get('edge_threshold_high', 150))
        
        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                               minLineLength=100, maxLineGap=20)
        
        if lines is None or len(lines) < 4:
            return False, {}
        
        # This is a simplified implementation
        # In practice, you'd want to group lines, find intersections, etc.
        # For now, return basic detection
        
        return False, {}  # Not implemented in this simplified version
    
    def _detect_paper_by_color(self, frame: np.ndarray, layout: DJBoardLayout) -> Tuple[bool, Dict]:
        """Detect paper template using color segmentation."""
        config = layout.detection_config
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for white paper
        if 'paper_color_range' in config:
            lower_white = np.array([0, 0, config['paper_color_range']['white_low'][2]])
            upper_white = np.array([180, 30, 255])
        else:
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
        
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (paper should be reasonably large)
            if area < 10000:
                continue
            
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                corners = approx.reshape(-1, 2).astype(np.float32)
                
                # Check aspect ratio
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                aspect_ratio = max(width, height) / min(width, height)
                
                # A3/A4 paper ratios
                expected_ratios = [1.414, 1.33]  # A4, 4:3
                ratio_match = any(abs(aspect_ratio - ratio) < 0.2 for ratio in expected_ratios)
                
                if ratio_match:
                    # Calculate homography
                    board_corners = np.array(layout.reference_points, dtype=np.float32)
                    homography, _ = cv2.findHomography(corners, board_corners)
                    
                    confidence = 0.8  # High confidence for color-based paper detection
                    
                    return True, {
                        'detected': True,
                        'confidence': confidence,
                        'corners': corners,
                        'method': DetectionMethod.COLOR_SEGMENTATION,
                        'homography': homography,
                        'area': area
                    }
        
        return False, {}
    
    def transform_points_to_board(self, image_points: np.ndarray) -> Optional[np.ndarray]:
        """Transform image points to board coordinates."""
        if not self.is_detected or self.homography_matrix is None:
            return None
        
        if image_points.ndim == 1:
            image_points = image_points.reshape(1, -1)
        
        board_points = cv2.perspectiveTransform(
            image_points.reshape(-1, 1, 2).astype(np.float32),
            self.homography_matrix
        )
        
        return board_points.reshape(-1, 2)
    
    def draw_detection_overlay(self, frame: np.ndarray, detection_info: Dict) -> np.ndarray:
        """Draw detection overlay on frame."""
        if not detection_info.get('detected', False):
            return frame
        
        corners = detection_info.get('corners')
        if corners is not None:
            # Draw board boundary
            pts = corners.astype(np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
            
            # Draw corner markers
            for i, corner in enumerate(corners):
                center = tuple(corner.astype(int))
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cv2.putText(frame, str(i), (center[0]+10, center[1]+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw detection info
        layout_name = detection_info.get('layout_name', 'Unknown')
        confidence = detection_info.get('confidence', 0.0)
        method = detection_info.get('method', DetectionMethod.CONTOUR_DETECTION)
        
        cv2.putText(frame, f"Board: {layout_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Method: {method.value}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def draw_control_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw control zones overlay."""
        if not self.is_detected or self.detected_board is None:
            return frame
        
        layout = self.board_layouts[self.detected_board]
        
        for zone in layout.control_zones:
            if zone['type'] == 'circular':
                center_board = np.array([zone['center']], dtype=np.float32)
                center_image = cv2.perspectiveTransform(
                    center_board.reshape(-1, 1, 2),
                    np.linalg.inv(self.homography_matrix)
                ).reshape(-1, 2)[0]
                
                # Transform radius (approximate)
                radius_board = zone['radius']
                radius_point = center_board + [[radius_board, 0]]
                radius_image = cv2.perspectiveTransform(
                    radius_point.reshape(-1, 1, 2),
                    np.linalg.inv(self.homography_matrix)
                ).reshape(-1, 2)[0]
                
                radius_pixels = int(np.linalg.norm(radius_image - center_image))
                
                cv2.circle(frame, tuple(center_image.astype(int)), radius_pixels, (255, 0, 0), 2)
                cv2.putText(frame, zone['name'], tuple((center_image - [10, -5]).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            elif zone['type'] == 'fader':
                bounds_board = np.array(zone['bounds'], dtype=np.float32)
                bounds_image = cv2.perspectiveTransform(
                    bounds_board.reshape(-1, 1, 2),
                    np.linalg.inv(self.homography_matrix)
                ).reshape(-1, 2)
                
                cv2.rectangle(frame, 
                            tuple(bounds_image[0].astype(int)),
                            tuple(bounds_image[1].astype(int)),
                            (0, 255, 255), 2)
                cv2.putText(frame, zone['name'], 
                           tuple((bounds_image[0] - [10, -5]).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            elif zone['type'] == 'knob':
                center_board = np.array([zone['center']], dtype=np.float32)
                center_image = cv2.perspectiveTransform(
                    center_board.reshape(-1, 1, 2),
                    np.linalg.inv(self.homography_matrix)
                ).reshape(-1, 2)[0]
                
                radius_board = zone['radius']
                radius_point = center_board + [[radius_board, 0]]
                radius_image = cv2.perspectiveTransform(
                    radius_point.reshape(-1, 1, 2),
                    np.linalg.inv(self.homography_matrix)
                ).reshape(-1, 2)[0]
                
                radius_pixels = int(np.linalg.norm(radius_image - center_image))
                
                cv2.circle(frame, tuple(center_image.astype(int)), radius_pixels, (255, 255, 0), 2)
                cv2.putText(frame, zone['name'], tuple((center_image - [10, -5]).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            elif zone['type'] == 'button':
                center_board = np.array([zone['center']], dtype=np.float32)
                center_image = cv2.perspectiveTransform(
                    center_board.reshape(-1, 1, 2),
                    np.linalg.inv(self.homography_matrix)
                ).reshape(-1, 2)[0]
                
                radius_board = zone['radius']
                radius_point = center_board + [[radius_board, 0]]
                radius_image = cv2.perspectiveTransform(
                    radius_point.reshape(-1, 1, 2),
                    np.linalg.inv(self.homography_matrix)
                ).reshape(-1, 2)[0]
                
                radius_pixels = int(np.linalg.norm(radius_image - center_image))
                
                cv2.circle(frame, tuple(center_image.astype(int)), radius_pixels, (0, 0, 255), 2)
                cv2.putText(frame, zone['name'], tuple((center_image - [10, -5]).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return frame
    
    def get_control_zones_for_gestures(self) -> List[Dict]:
        """Get control zones in format compatible with gesture recognizer."""
        if not self.is_detected or self.detected_board is None:
            return []
        
        layout = self.board_layouts[self.detected_board]
        gesture_zones = []
        
        for zone in layout.control_zones:
            # Convert to gesture recognizer format
            if zone['type'] == 'fader':
                gesture_zone = {
                    'name': zone['name'],
                    'zone_type': 'fader',
                    'bounds': zone['bounds'],
                    'orientation': 'vertical' if abs(zone['bounds'][1][1] - zone['bounds'][0][1]) > abs(zone['bounds'][1][0] - zone['bounds'][0][0]) else 'horizontal'
                }
            elif zone['type'] in ['knob', 'circular']:
                # Convert circular to rectangular bounds for gesture detection
                center = zone['center']
                radius = zone.get('radius', 1.0)
                gesture_zone = {
                    'name': zone['name'],
                    'zone_type': 'knob',
                    'bounds': [
                        [center[0] - radius, center[1] - radius],
                        [center[0] + radius, center[1] + radius]
                    ],
                    'orientation': 'radial'
                }
            elif zone['type'] == 'button':
                center = zone['center']
                radius = zone.get('radius', 1.0)
                gesture_zone = {
                    'name': zone['name'],
                    'zone_type': 'button',
                    'bounds': [
                        [center[0] - radius, center[1] - radius],
                        [center[0] + radius, center[1] + radius]
                    ]
                }
            else:
                continue
            
            gesture_zones.append(gesture_zone)
        
        return gesture_zones
    
    def save_detection_config(self, filepath: str) -> bool:
        """Save current detection configuration."""
        if not self.is_detected:
            return False
        
        config = {
            'detected_board': self.detected_board,
            'board_type': self.board_type.value if self.board_type else None,
            'detection_confidence': self.detection_confidence,
            'homography_matrix': self.homography_matrix.tolist() if self.homography_matrix is not None else None,
            'board_corners': self.board_corners.tolist() if self.board_corners is not None else None,
            'timestamp': time.time()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            logging.info(f"Detection config saved to {filepath}")
            return True
        except Exception as e:
            logging.error(f"Failed to save detection config: {e}")
            return False
    
    def load_detection_config(self, filepath: str) -> bool:
        """Load detection configuration."""
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            self.detected_board = config['detected_board']
            self.board_type = DJBoardType(config['board_type']) if config['board_type'] else None
            self.detection_confidence = config['detection_confidence']
            self.homography_matrix = np.array(config['homography_matrix']) if config['homography_matrix'] else None
            self.board_corners = np.array(config['board_corners']) if config['board_corners'] else None
            self.is_detected = True
            
            logging.info(f"Detection config loaded from {filepath}")
            return True
        except Exception as e:
            logging.error(f"Failed to load detection config: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test DJ board detection
    import sys
    sys.path.append('.')
    from capture import CameraCapture
    
    camera = CameraCapture()
    detector = DJBoardDetector()
    
    try:
        print("Starting DJ board detection test...")
        print("Controls:")
        print("  1 - Detect physical controllers")
        print("  2 - Detect paper templates")
        print("  3 - Auto-detect")
        print("  z - Toggle control zones overlay")
        print("  s - Save detection config")
        print("  l - Load detection config")
        print("  q - Quit")
        
        show_zones = False
        
        while True:
            success, frame = camera.read_frame()
            
            if not success:
                print("Failed to capture frame")
                break
            
            # Handle keyboard input first
            key = cv2.waitKey(1) & 0xFF
            
            detection_performed = False
            
            if key == ord('1'):
                print("Detecting physical controllers...")
                success, detection_info = detector.detect_board(frame, DJBoardType.PHYSICAL_CONTROLLER)
                detection_performed = True
            elif key == ord('2'):
                print("Detecting paper templates...")
                success, detection_info = detector.detect_board(frame, DJBoardType.PAPER_TEMPLATE)
                detection_performed = True
            elif key == ord('3'):
                print("Auto-detecting...")
                success, detection_info = detector.detect_board(frame, DJBoardType.AUTO_DETECT)
                detection_performed = True
            elif key == ord('z'):
                show_zones = not show_zones
                print(f"Control zones overlay: {'ON' if show_zones else 'OFF'}")
            elif key == ord('s'):
                if detector.save_detection_config("config/dj_board_detection.json"):
                    print("Detection config saved!")
            elif key == ord('l'):
                if detector.load_detection_config("config/dj_board_detection.json"):
                    print("Detection config loaded!")
            elif key == ord('q'):
                break
            
            # Continuous detection for auto mode (less frequent)
            if not detection_performed and cv2.getTickCount() % 30 == 0:  # Every 30 frames
                success, detection_info = detector.detect_board(frame, DJBoardType.AUTO_DETECT)
            
            # Draw detection overlay
            if detector.is_detected:
                # Get the latest detection info
                detection_info = {
                    'detected': True,
                    'layout_name': detector.detected_board,
                    'confidence': detector.detection_confidence,
                    'corners': detector.board_corners,
                    'method': 'loaded'
                }
                
                frame = detector.draw_detection_overlay(frame, detection_info)
                
                if show_zones:
                    frame = detector.draw_control_zones(frame)
            
            # Add instructions
            cv2.putText(frame, "Press 1-3 to detect, Z for zones, S/L to save/load, Q to quit", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('DJ Board Detection Test', frame)
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        camera.close()
        cv2.destroyAllWindows()
        print("DJ board detection test completed")