"""
Gesture recognition and control zone detection for DJ controller.
Handles drag, tap, and rotate gestures within defined control zones.
"""

import numpy as np
import cv2
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import json
from dataclasses import dataclass, asdict


class GestureType(Enum):
    """Enumeration of supported gesture types."""
    NONE = "none"
    TAP = "tap"
    DRAG = "drag"
    ROTATE = "rotate"
    HOLD = "hold"


class ControlZoneType(Enum):
    """Enumeration of control zone types."""
    FADER = "fader"           # Linear slider (crossfader, volume)
    BUTTON = "button"         # Toggle/momentary button
    KNOB = "knob"            # Rotary control
    XY_PAD = "xy_pad"        # 2D control surface
    CUSTOM = "custom"        # Custom zone


@dataclass
class ControlZone:
    """Definition of a control zone on the DJ board."""
    name: str
    zone_type: ControlZoneType
    bounds: List[Tuple[float, float]]  # Board coordinates [(x1,y1), (x2,y2), ...]
    midi_channel: int
    midi_cc: Optional[int] = None
    midi_note: Optional[int] = None
    min_value: float = 0.0
    max_value: float = 127.0
    orientation: str = "horizontal"  # "horizontal", "vertical", "radial"
    sensitivity: float = 1.0
    enabled: bool = True


@dataclass
class GestureEvent:
    """Represents a detected gesture event."""
    gesture_type: GestureType
    zone_name: str
    position: Tuple[float, float]  # Board coordinates
    value: float  # Normalized value 0-1
    raw_value: float  # Raw sensor value
    velocity: float  # Change rate
    confidence: float  # Detection confidence
    timestamp: float
    hand_id: int = 0
    additional_data: Optional[Dict] = None


class HandState:
    """Tracks state of a single hand for gesture recognition."""
    
    def __init__(self, hand_id: int, max_history: int = 10):
        self.hand_id = hand_id
        self.max_history = max_history
        
        # Position tracking
        self.position_history: List[Tuple[float, float, float]] = []  # (x, y, timestamp)
        self.current_position: Optional[Tuple[float, float]] = None
        
        # Gesture state
        self.current_gesture = GestureType.NONE
        self.gesture_start_time: Optional[float] = None
        self.gesture_start_position: Optional[Tuple[float, float]] = None
        
        # Zone interaction
        self.current_zone: Optional[str] = None
        self.zone_entry_time: Optional[float] = None
        self.last_zone_value: Optional[float] = None
        
        # Finger tracking for detailed gestures
        self.fingertip_positions: Dict[str, Tuple[float, float]] = {}
        self.finger_states: Dict[str, bool] = {}  # Extended/retracted
        
        # Tap detection
        self.tap_candidate_start: Optional[float] = None
        self.tap_candidate_position: Optional[Tuple[float, float]] = None
        
        # Rotation tracking
        self.rotation_center: Optional[Tuple[float, float]] = None
        self.rotation_angle_history: List[float] = []
    
    def update_position(self, position: Tuple[float, float], timestamp: float):
        """Update hand position and maintain history."""
        self.current_position = position
        self.position_history.append((position[0], position[1], timestamp))
        
        # Limit history size
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
    
    def get_velocity(self) -> Tuple[float, float]:
        """Calculate current velocity vector."""
        if len(self.position_history) < 2:
            return (0.0, 0.0)
        
        recent = self.position_history[-2:]
        dt = recent[1][2] - recent[0][2]
        
        if dt <= 0:
            return (0.0, 0.0)
        
        dx = recent[1][0] - recent[0][0]
        dy = recent[1][1] - recent[0][1]
        
        return (dx / dt, dy / dt)
    
    def get_speed(self) -> float:
        """Calculate current speed (magnitude of velocity)."""
        vx, vy = self.get_velocity()
        return np.sqrt(vx**2 + vy**2)
    
    def get_movement_distance(self, time_window: float = 0.5) -> float:
        """Calculate total movement distance in time window."""
        current_time = time.time()
        total_distance = 0.0
        
        for i in range(len(self.position_history) - 1):
            t1 = self.position_history[i][2]
            t2 = self.position_history[i + 1][2]
            
            if current_time - t1 > time_window:
                continue
            
            x1, y1 = self.position_history[i][:2]
            x2, y2 = self.position_history[i + 1][:2]
            
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_distance += distance
        
        return total_distance


class GestureRecognizer:
    """
    Main gesture recognition engine.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize gesture recognizer.
        
        Args:
            config_path: Path to configuration file with zones and parameters
        """
        # Gesture detection parameters
        self.tap_max_distance = 0.02  # 2cm
        self.tap_max_duration = 0.5   # 0.5 seconds
        self.tap_min_duration = 0.05  # 50ms
        
        self.drag_min_distance = 0.01  # 1cm
        self.drag_min_duration = 0.1   # 100ms
        
        self.rotate_min_angle = 15.0   # 15 degrees
        self.rotate_center_radius = 0.03  # 3cm
        
        self.hold_min_duration = 1.0   # 1 second
        
        # Control zones
        self.control_zones: Dict[str, ControlZone] = {}
        self.load_default_zones()
        
        # Load configuration if provided
        if config_path:
            self.load_zones_from_file(config_path)
        
        # Hand tracking
        self.hand_states: Dict[int, HandState] = {}
        
        # Event history
        self.recent_events: List[GestureEvent] = []
        self.max_event_history = 100
        
        # Load configuration if provided
        if config_path:
            self.load_configuration(config_path)
        
        logging.info("GestureRecognizer initialized")
    
    def load_default_zones(self):
        """Load default DJ control zones."""
        # Crossfader (center bottom)
        self.control_zones["crossfader"] = ControlZone(
            name="crossfader",
            zone_type=ControlZoneType.FADER,
            bounds=[(0.15, 0.25), (0.25, 0.30)],
            midi_channel=1,
            midi_cc=8,  # Standard crossfader CC
            orientation="horizontal"
        )
        
        # Volume A (left side)
        self.control_zones["volume_a"] = ControlZone(
            name="volume_a",
            zone_type=ControlZoneType.FADER,
            bounds=[(0.05, 0.05), (0.10, 0.20)],
            midi_channel=1,
            midi_cc=7,  # Volume CC
            orientation="vertical"
        )
        
        # Volume B (right side)
        self.control_zones["volume_b"] = ControlZone(
            name="volume_b",
            zone_type=ControlZoneType.FADER,
            bounds=[(0.30, 0.05), (0.35, 0.20)],
            midi_channel=2,
            midi_cc=7,
            orientation="vertical"
        )
        
        # Play button A
        self.control_zones["play_a"] = ControlZone(
            name="play_a",
            zone_type=ControlZoneType.BUTTON,
            bounds=[(0.08, 0.22), (0.12, 0.26)],
            midi_channel=1,
            midi_note=60  # Middle C
        )
        
        # Play button B
        self.control_zones["play_b"] = ControlZone(
            name="play_b",
            zone_type=ControlZoneType.BUTTON,
            bounds=[(0.28, 0.22), (0.32, 0.26)],
            midi_channel=2,
            midi_note=60
        )
        
        # Filter knob A
        self.control_zones["filter_a"] = ControlZone(
            name="filter_a",
            zone_type=ControlZoneType.KNOB,
            bounds=[(0.02, 0.02), (0.08, 0.08)],
            midi_channel=1,
            midi_cc=74,  # Filter CC
            orientation="radial"
        )
        
        # Filter knob B
        self.control_zones["filter_b"] = ControlZone(
            name="filter_b",
            zone_type=ControlZoneType.KNOB,
            bounds=[(0.32, 0.02), (0.38, 0.08)],
            midi_channel=2,
            midi_cc=74,
            orientation="radial"
        )
        
        # XY Performance pad (center)
        self.control_zones["xy_pad"] = ControlZone(
            name="xy_pad",
            zone_type=ControlZoneType.XY_PAD,
            bounds=[(0.15, 0.10), (0.25, 0.20)],
            midi_channel=1,
            midi_cc=16,  # X-axis
            additional_data={"y_cc": 17}  # Y-axis
        )
    
    def load_configuration(self, config_path: str) -> bool:
        """Load zones and parameters from configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load zones
            if 'zones' in config:
                self.control_zones.clear()
                for zone_data in config['zones']:
                    zone = ControlZone(**zone_data)
                    self.control_zones[zone.name] = zone
            
            # Load parameters
            if 'parameters' in config:
                params = config['parameters']
                self.tap_max_distance = params.get('tap_max_distance', self.tap_max_distance)
                self.tap_max_duration = params.get('tap_max_duration', self.tap_max_duration)
                self.drag_min_distance = params.get('drag_min_distance', self.drag_min_distance)
                # ... load other parameters
            
            logging.info(f"Configuration loaded from {config_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            return False
    
    def save_configuration(self, config_path: str) -> bool:
        """Save current zones and parameters to configuration file."""
        try:
            config = {
                'zones': [asdict(zone) for zone in self.control_zones.values()],
                'parameters': {
                    'tap_max_distance': self.tap_max_distance,
                    'tap_max_duration': self.tap_max_duration,
                    'tap_min_duration': self.tap_min_duration,
                    'drag_min_distance': self.drag_min_distance,
                    'drag_min_duration': self.drag_min_duration,
                    'rotate_min_angle': self.rotate_min_angle,
                    'rotate_center_radius': self.rotate_center_radius,
                    'hold_min_duration': self.hold_min_duration
                }
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logging.info(f"Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")
            return False
    
    def point_in_zone(self, point: Tuple[float, float], zone: ControlZone) -> bool:
        """Check if point is inside zone bounds."""
        x, y = point
        
        # Simple rectangular bounds check
        if len(zone.bounds) == 2:
            x1, y1 = zone.bounds[0]
            x2, y2 = zone.bounds[1]
            return x1 <= x <= x2 and y1 <= y <= y2
        
        # Polygon bounds check
        elif len(zone.bounds) > 2:
            return self._point_in_polygon(point, zone.bounds)
        
        return False
    
    def _point_in_polygon(self, point: Tuple[float, float], 
                         polygon: List[Tuple[float, float]]) -> bool:
        """Check if point is inside polygon using ray casting."""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def find_active_zone(self, position: Tuple[float, float]) -> Optional[str]:
        """Find which zone contains the given position."""
        for zone_name, zone in self.control_zones.items():
            if zone.enabled and self.point_in_zone(position, zone):
                return zone_name
        return None
    
    def calculate_zone_value(self, position: Tuple[float, float], 
                           zone: ControlZone) -> float:
        """Calculate normalized value (0-1) for position within zone."""
        x, y = position
        
        if zone.zone_type == ControlZoneType.FADER:
            if zone.orientation == "horizontal":
                x1, _ = zone.bounds[0]
                x2, _ = zone.bounds[1]
                value = (x - x1) / (x2 - x1) if x2 != x1 else 0.5
            else:  # vertical
                _, y1 = zone.bounds[0]
                _, y2 = zone.bounds[1]
                value = (y - y1) / (y2 - y1) if y2 != y1 else 0.5
        
        elif zone.zone_type == ControlZoneType.KNOB:
            # Calculate angle from center
            center_x = (zone.bounds[0][0] + zone.bounds[1][0]) / 2
            center_y = (zone.bounds[0][1] + zone.bounds[1][1]) / 2
            
            angle = np.arctan2(y - center_y, x - center_x)
            angle = (angle + np.pi) / (2 * np.pi)  # Normalize to 0-1
            value = angle
        
        elif zone.zone_type == ControlZoneType.XY_PAD:
            x1, y1 = zone.bounds[0]
            x2, y2 = zone.bounds[1]
            value_x = (x - x1) / (x2 - x1) if x2 != x1 else 0.5
            value_y = (y - y1) / (y2 - y1) if y2 != y1 else 0.5
            value = (value_x + value_y) / 2  # Average for single value
        
        else:
            value = 0.5  # Default for buttons
        
        # Clamp to 0-1 range
        return max(0.0, min(1.0, value))
    
    def update_hand_states(self, hand_data: List[Dict], timestamp: float):
        """Update hand states from detection data."""
        current_hand_positions = {}
        
        # Process detected hands
        if hand_data:
            for i, hand_info in enumerate(hand_data):
                hand_id = i  # Simple ID assignment
                landmarks = hand_info['landmarks']
                
                # Use index finger tip as primary position
                if len(landmarks) > 8:  # INDEX_FINGER_TIP
                    position = (landmarks[8]['x_norm'], landmarks[8]['y_norm'])
                    current_hand_positions[hand_id] = position
                    
                    # Update or create hand state
                    if hand_id not in self.hand_states:
                        self.hand_states[hand_id] = HandState(hand_id)
                    
                    self.hand_states[hand_id].update_position(position, timestamp)
                    
                    # Store finger positions for advanced gestures
                    finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
                    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
                    
                    for finger_idx, finger_name in zip(finger_tips, finger_names):
                        if len(landmarks) > finger_idx:
                            self.hand_states[hand_id].fingertip_positions[finger_name] = (
                                landmarks[finger_idx]['x_norm'], 
                                landmarks[finger_idx]['y_norm']
                            )
        
        # Remove inactive hands
        inactive_hands = [hid for hid in self.hand_states.keys() 
                         if hid not in current_hand_positions]
        for hand_id in inactive_hands:
            del self.hand_states[hand_id]
    
    def recognize_gestures(self, timestamp: float) -> List[GestureEvent]:
        """Recognize gestures from current hand states."""
        events = []
        
        for hand_id, hand_state in self.hand_states.items():
            if hand_state.current_position is None:
                continue
            
            # Find active zone
            active_zone_name = self.find_active_zone(hand_state.current_position)
            
            # Zone entry/exit detection
            if active_zone_name != hand_state.current_zone:
                hand_state.current_zone = active_zone_name
                hand_state.zone_entry_time = timestamp
                hand_state.last_zone_value = None
            
            if active_zone_name:
                zone = self.control_zones[active_zone_name]
                events.extend(self._recognize_zone_gestures(hand_state, zone, timestamp))
        
        # Add events to history
        self.recent_events.extend(events)
        if len(self.recent_events) > self.max_event_history:
            self.recent_events = self.recent_events[-self.max_event_history:]
        
        return events
    
    def _recognize_zone_gestures(self, hand_state: HandState, 
                               zone: ControlZone, timestamp: float) -> List[GestureEvent]:
        """Recognize gestures within a specific zone."""
        events = []
        current_value = self.calculate_zone_value(hand_state.current_position, zone)
        
        # Detect gesture type based on zone and movement
        speed = hand_state.get_speed()
        movement_distance = hand_state.get_movement_distance()
        
        if zone.zone_type == ControlZoneType.BUTTON:
            # Button tap detection
            if self._detect_tap(hand_state, timestamp):
                event = GestureEvent(
                    gesture_type=GestureType.TAP,
                    zone_name=zone.name,
                    position=hand_state.current_position,
                    value=1.0,  # Button press
                    raw_value=current_value,
                    velocity=speed,
                    confidence=0.9,
                    timestamp=timestamp,
                    hand_id=hand_state.hand_id
                )
                events.append(event)
        
        elif zone.zone_type in [ControlZoneType.FADER, ControlZoneType.XY_PAD]:
            # Fader drag detection
            if movement_distance > self.drag_min_distance:
                event = GestureEvent(
                    gesture_type=GestureType.DRAG,
                    zone_name=zone.name,
                    position=hand_state.current_position,
                    value=current_value,
                    raw_value=current_value,
                    velocity=speed,
                    confidence=0.8,
                    timestamp=timestamp,
                    hand_id=hand_state.hand_id
                )
                events.append(event)
        
        elif zone.zone_type == ControlZoneType.KNOB:
            # Knob rotation detection
            if self._detect_rotation(hand_state, zone, timestamp):
                event = GestureEvent(
                    gesture_type=GestureType.ROTATE,
                    zone_name=zone.name,
                    position=hand_state.current_position,
                    value=current_value,
                    raw_value=current_value,
                    velocity=speed,
                    confidence=0.7,
                    timestamp=timestamp,
                    hand_id=hand_state.hand_id
                )
                events.append(event)
        
        # Update last zone value
        hand_state.last_zone_value = current_value
        
        return events
    
    def _detect_tap(self, hand_state: HandState, timestamp: float) -> bool:
        """Detect tap gesture."""
        if len(hand_state.position_history) < 3:
            return False
        
        # Check for brief stationary period (tap candidate)
        current_speed = hand_state.get_speed()
        
        if current_speed < 0.1:  # Nearly stationary
            if hand_state.tap_candidate_start is None:
                hand_state.tap_candidate_start = timestamp
                hand_state.tap_candidate_position = hand_state.current_position
        else:
            # Check if we just finished a tap
            if hand_state.tap_candidate_start is not None:
                duration = timestamp - hand_state.tap_candidate_start
                
                if (self.tap_min_duration <= duration <= self.tap_max_duration):
                    # Calculate movement during tap
                    if hand_state.tap_candidate_position:
                        x1, y1 = hand_state.tap_candidate_position
                        x2, y2 = hand_state.current_position
                        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        
                        if distance <= self.tap_max_distance:
                            hand_state.tap_candidate_start = None
                            return True
                
                hand_state.tap_candidate_start = None
        
        return False
    
    def _detect_rotation(self, hand_state: HandState, zone: ControlZone, 
                        timestamp: float) -> bool:
        """Detect rotation gesture for knobs."""
        if len(hand_state.position_history) < 3:
            return False
        
        # Calculate center of knob
        center_x = (zone.bounds[0][0] + zone.bounds[1][0]) / 2
        center_y = (zone.bounds[0][1] + zone.bounds[1][1]) / 2
        
        # Track angle changes
        recent_positions = hand_state.position_history[-3:]
        angles = []
        
        for pos_x, pos_y, _ in recent_positions:
            angle = np.arctan2(pos_y - center_y, pos_x - center_x)
            angles.append(np.degrees(angle))
        
        # Check for significant rotation
        if len(angles) >= 2:
            angle_change = abs(angles[-1] - angles[0])
            # Handle angle wrap-around
            if angle_change > 180:
                angle_change = 360 - angle_change
            
            return angle_change >= self.rotate_min_angle
        
        return False
    
    def draw_zones(self, frame: np.ndarray, 
                   calibration_manager) -> np.ndarray:
        """Draw control zones on frame."""
        if not calibration_manager.is_calibrated():
            return frame
        
        for zone_name, zone in self.control_zones.items():
            if not zone.enabled:
                continue
            
            # Convert zone bounds to image coordinates
            zone_points = np.array(zone.bounds, dtype=np.float32)
            image_points = calibration_manager.calibrator.board_to_image_coordinates(zone_points)
            
            if image_points is not None:
                # Choose color based on zone type
                color_map = {
                    ControlZoneType.FADER: (255, 0, 0),    # Blue
                    ControlZoneType.BUTTON: (0, 255, 0),   # Green  
                    ControlZoneType.KNOB: (0, 0, 255),     # Red
                    ControlZoneType.XY_PAD: (255, 255, 0)  # Cyan
                }
                
                color = color_map.get(zone.zone_type, (128, 128, 128))
                pts = image_points.astype(np.int32)
                
                # Draw zone boundary
                cv2.rectangle(frame, tuple(pts[0]), tuple(pts[1]), color, 2)
                
                # Add zone label
                label_pos = (pts[0][0], pts[0][1] - 10)
                cv2.putText(frame, zone_name, label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def get_zone_info(self) -> Dict:
        """Get information about all zones."""
        return {
            name: {
                'type': zone.zone_type.value,
                'midi_channel': zone.midi_channel,
                'midi_cc': zone.midi_cc,
                'midi_note': zone.midi_note,
                'enabled': zone.enabled
            }
            for name, zone in self.control_zones.items()
        }
    
    def update_zones_from_calibration(self, calibration_manager):
        """Update control zones from calibration manager (DJ board detection)."""
        try:
            dj_zones = calibration_manager.get_control_zones()
            if dj_zones:
                logging.info(f"Loading {len(dj_zones)} control zones from DJ board detection")
                
                # Clear existing zones
                self.control_zones.clear()
                
                # Convert DJ board zones to ControlZone objects
                for dj_zone in dj_zones:
                    try:
                        zone_type = ControlZoneType(dj_zone['zone_type'])
                        
                        # Create ControlZone object
                        control_zone = ControlZone(
                            name=dj_zone['name'],
                            zone_type=zone_type,
                            bounds=dj_zone['bounds'],
                            midi_channel=dj_zone.get('midi_channel', 1),
                            midi_cc=dj_zone.get('midi_cc'),
                            midi_note=dj_zone.get('midi_note'),
                            min_value=dj_zone.get('min_value', 0.0),
                            max_value=dj_zone.get('max_value', 127.0),
                            orientation=dj_zone.get('orientation', 'horizontal'),
                            sensitivity=dj_zone.get('sensitivity', 1.0),
                            enabled=dj_zone.get('enabled', True)
                        )
                        
                        self.control_zones[control_zone.name] = control_zone
                        
                    except Exception as e:
                        logging.warning(f"Failed to create control zone {dj_zone.get('name', 'unknown')}: {e}")
                
                logging.info(f"Successfully loaded {len(self.control_zones)} control zones from DJ board")
            else:
                logging.info("No DJ board zones available, using default zones")
                
        except Exception as e:
            logging.error(f"Failed to update zones from calibration: {e}")
            # Fall back to default zones if something goes wrong
            self.load_default_zones()


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test gesture recognition
    gesture_recognizer = GestureRecognizer()
    
    print("Gesture Recognition Test")
    print("Available zones:", list(gesture_recognizer.control_zones.keys()))
    
    # Simulate hand data
    test_positions = [
        (0.2, 0.27),  # Crossfader area
        (0.07, 0.15), # Volume A area
        (0.10, 0.24)  # Play A area
    ]
    
    for i, pos in enumerate(test_positions):
        active_zone = gesture_recognizer.find_active_zone(pos)
        print(f"Position {pos} -> Zone: {active_zone}")
        
        if active_zone:
            zone = gesture_recognizer.control_zones[active_zone]
            value = gesture_recognizer.calculate_zone_value(pos, zone)
            print(f"  Value: {value:.3f}")
    
    print("Gesture recognition test completed")