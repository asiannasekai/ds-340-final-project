"""
MIDI controller for DJ interface.
Handles virtual MIDI output, CC messages, note events, and DJ software integration.
"""

import rtmidi
import mido
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum
from dataclasses import dataclass
import json
import queue


class MidiMessageType(Enum):
    """MIDI message types."""
    NOTE_ON = "note_on"
    NOTE_OFF = "note_off"
    CONTROL_CHANGE = "control_change"
    PITCH_BEND = "pitch_bend"
    PROGRAM_CHANGE = "program_change"


@dataclass
class MidiMapping:
    """MIDI mapping configuration."""
    control_name: str
    message_type: MidiMessageType
    channel: int
    controller: Optional[int] = None  # CC number or note number
    min_value: int = 0
    max_value: int = 127
    curve: str = "linear"  # "linear", "exponential", "logarithmic"
    enabled: bool = True


class MidiMessageQueue:
    """Thread-safe MIDI message queue."""
    
    def __init__(self, maxsize: int = 1000):
        self.queue = queue.Queue(maxsize=maxsize)
        self.dropped_messages = 0
        
    def put(self, message: Dict, block: bool = False):
        """Add message to queue."""
        try:
            self.queue.put(message, block=block)
        except queue.Full:
            self.dropped_messages += 1
            logging.warning(f"MIDI queue full, dropped message. Total dropped: {self.dropped_messages}")
    
    def get(self, timeout: Optional[float] = None) -> Optional[Dict]:
        """Get message from queue."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_all(self) -> List[Dict]:
        """Get all pending messages."""
        messages = []
        while not self.queue.empty():
            try:
                messages.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return messages
    
    def size(self) -> int:
        """Get queue size."""
        return self.queue.qsize()
    
    def clear(self):
        """Clear all messages."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break


class VirtualMidiOutput:
    """
    Virtual MIDI output port manager using python-rtmidi.
    """
    
    def __init__(self, port_name: str = "DJ Controller"):
        """
        Initialize virtual MIDI output.
        
        Args:
            port_name: Name of the virtual MIDI port
        """
        self.port_name = port_name
        self.midi_out = None
        self.is_connected = False
        self.message_count = 0
        self.error_count = 0
        
        # Performance tracking
        self.last_message_time = 0.0
        self.message_times = []
        self.max_message_history = 100
        
        self._initialize_port()
    
    def _initialize_port(self):
        """Initialize the virtual MIDI port."""
        try:
            self.midi_out = rtmidi.MidiOut()
            
            # Create virtual port
            self.midi_out.open_virtual_port(self.port_name)
            self.is_connected = True
            
            logging.info(f"Virtual MIDI port '{self.port_name}' created successfully")
            
        except Exception as e:
            logging.error(f"Failed to create MIDI port: {e}")
            self.is_connected = False
            
            # Try alternative initialization with mido
            try:
                self._initialize_with_mido()
            except Exception as e2:
                logging.error(f"Mido initialization also failed: {e2}")
    
    def _initialize_with_mido(self):
        """Alternative initialization using mido."""
        try:
            # List available ports
            available_ports = mido.get_output_names()
            logging.info(f"Available MIDI ports: {available_ports}")
            
            # For now, we'll create a simple fallback
            self.is_connected = True
            logging.info("Using mido fallback mode")
            
        except Exception as e:
            logging.error(f"Mido initialization failed: {e}")
            raise
    
    def send_message(self, message_type: MidiMessageType, 
                    channel: int, controller: int, value: int) -> bool:
        """
        Send MIDI message.
        
        Args:
            message_type: Type of MIDI message
            channel: MIDI channel (0-15)
            controller: Controller/note number
            value: Value (0-127)
            
        Returns:
            True if message sent successfully
        """
        if not self.is_connected or self.midi_out is None:
            return False
        
        try:
            # Clamp values to valid MIDI ranges
            channel = max(0, min(15, channel))
            controller = max(0, min(127, controller))
            value = max(0, min(127, value))
            
            # Create MIDI message
            if message_type == MidiMessageType.CONTROL_CHANGE:
                message = [0xB0 + channel, controller, value]
            elif message_type == MidiMessageType.NOTE_ON:
                message = [0x90 + channel, controller, value]
            elif message_type == MidiMessageType.NOTE_OFF:
                message = [0x80 + channel, controller, value]
            elif message_type == MidiMessageType.PITCH_BEND:
                # Pitch bend uses 14-bit value
                lsb = value & 0x7F
                msb = (value >> 7) & 0x7F
                message = [0xE0 + channel, lsb, msb]
            else:
                logging.warning(f"Unsupported message type: {message_type}")
                return False
            
            # Send message
            self.midi_out.send_message(message)
            
            # Update statistics
            self.message_count += 1
            current_time = time.time()
            self.message_times.append(current_time)
            
            # Limit message history
            if len(self.message_times) > self.max_message_history:
                self.message_times.pop(0)
            
            self.last_message_time = current_time
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to send MIDI message: {e}")
            self.error_count += 1
            return False
    
    def send_cc(self, channel: int, controller: int, value: int) -> bool:
        """Send Control Change message."""
        return self.send_message(MidiMessageType.CONTROL_CHANGE, channel, controller, value)
    
    def send_note_on(self, channel: int, note: int, velocity: int = 127) -> bool:
        """Send Note On message."""
        return self.send_message(MidiMessageType.NOTE_ON, channel, note, velocity)
    
    def send_note_off(self, channel: int, note: int, velocity: int = 0) -> bool:
        """Send Note Off message."""
        return self.send_message(MidiMessageType.NOTE_OFF, channel, note, velocity)
    
    def send_pitch_bend(self, channel: int, value: int) -> bool:
        """Send Pitch Bend message."""
        return self.send_message(MidiMessageType.PITCH_BEND, channel, 0, value)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        current_time = time.time()
        
        # Calculate recent message rate
        recent_messages = [t for t in self.message_times if current_time - t <= 1.0]
        messages_per_second = len(recent_messages)
        
        # Calculate average latency (rough estimate)
        if len(self.message_times) >= 2:
            intervals = [self.message_times[i] - self.message_times[i-1] 
                        for i in range(1, len(self.message_times))]
            avg_interval = sum(intervals) / len(intervals) * 1000  # ms
        else:
            avg_interval = 0.0
        
        return {
            'is_connected': self.is_connected,
            'total_messages': self.message_count,
            'error_count': self.error_count,
            'messages_per_second': messages_per_second,
            'avg_interval_ms': avg_interval,
            'last_message_age': current_time - self.last_message_time
        }
    
    def close(self):
        """Close MIDI port."""
        if self.midi_out is not None:
            try:
                self.midi_out.close_port()
                logging.info(f"MIDI port '{self.port_name}' closed")
            except Exception as e:
                logging.error(f"Error closing MIDI port: {e}")
            finally:
                self.midi_out = None
                self.is_connected = False


class DJMidiController:
    """
    High-level DJ MIDI controller with gesture-to-MIDI mapping.
    """
    
    def __init__(self, port_name: str = "DJ Controller", config_file: Optional[str] = None):
        """
        Initialize DJ MIDI controller.
        
        Args:
            port_name: MIDI port name
            config_file: Configuration file path
        """
        self.midi_out = VirtualMidiOutput(port_name)
        self.mappings: Dict[str, MidiMapping] = {}
        self.message_queue = MidiMessageQueue()
        
        # Control state tracking
        self.control_states: Dict[str, float] = {}
        self.button_states: Dict[str, bool] = {}
        self.last_values: Dict[str, int] = {}
        
        # Performance settings
        self.max_messages_per_frame = 10
        self.min_change_threshold = 1  # Minimum value change to send update
        
        # Load default mappings
        self._load_default_mappings()
        
        # Load configuration if provided
        if config_file:
            self.load_configuration(config_file)
        
        # Start message processing thread
        self.processing_thread = threading.Thread(target=self._process_messages, daemon=True)
        self.running = True
        self.processing_thread.start()
        
        logging.info("DJMidiController initialized")
    
    def _load_default_mappings(self):
        """Load default MIDI mappings for DJ controls."""
        default_mappings = [
            # Crossfader
            MidiMapping("crossfader", MidiMessageType.CONTROL_CHANGE, 0, 8),
            
            # Volume faders
            MidiMapping("volume_a", MidiMessageType.CONTROL_CHANGE, 0, 7),
            MidiMapping("volume_b", MidiMessageType.CONTROL_CHANGE, 1, 7),
            
            # Play buttons
            MidiMapping("play_a", MidiMessageType.NOTE_ON, 0, 60),
            MidiMapping("play_b", MidiMessageType.NOTE_ON, 1, 60),
            
            # Cue buttons
            MidiMapping("cue_a", MidiMessageType.NOTE_ON, 0, 61),
            MidiMapping("cue_b", MidiMessageType.NOTE_ON, 1, 61),
            
            # Filter knobs
            MidiMapping("filter_a", MidiMessageType.CONTROL_CHANGE, 0, 74),
            MidiMapping("filter_b", MidiMessageType.CONTROL_CHANGE, 1, 74),
            
            # EQ controls
            MidiMapping("eq_high_a", MidiMessageType.CONTROL_CHANGE, 0, 16),
            MidiMapping("eq_mid_a", MidiMessageType.CONTROL_CHANGE, 0, 17),
            MidiMapping("eq_low_a", MidiMessageType.CONTROL_CHANGE, 0, 18),
            MidiMapping("eq_high_b", MidiMessageType.CONTROL_CHANGE, 1, 16),
            MidiMapping("eq_mid_b", MidiMessageType.CONTROL_CHANGE, 1, 17),
            MidiMapping("eq_low_b", MidiMessageType.CONTROL_CHANGE, 1, 18),
            
            # Pitch bend
            MidiMapping("pitch_a", MidiMessageType.PITCH_BEND, 0),
            MidiMapping("pitch_b", MidiMessageType.PITCH_BEND, 1),
            
            # XY Pad
            MidiMapping("xy_pad_x", MidiMessageType.CONTROL_CHANGE, 0, 20),
            MidiMapping("xy_pad_y", MidiMessageType.CONTROL_CHANGE, 0, 21),
        ]
        
        for mapping in default_mappings:
            self.mappings[mapping.control_name] = mapping
    
    def load_configuration(self, config_file: str) -> bool:
        """Load MIDI mappings from configuration file."""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            if 'midi_mappings' in config:
                self.mappings.clear()
                for mapping_data in config['midi_mappings']:
                    mapping = MidiMapping(**mapping_data)
                    self.mappings[mapping.control_name] = mapping
            
            logging.info(f"MIDI configuration loaded from {config_file}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load MIDI configuration: {e}")
            return False
    
    def save_configuration(self, config_file: str) -> bool:
        """Save current MIDI mappings to configuration file."""
        try:
            config = {
                'midi_mappings': [
                    {
                        'control_name': mapping.control_name,
                        'message_type': mapping.message_type.value,
                        'channel': mapping.channel,
                        'controller': mapping.controller,
                        'min_value': mapping.min_value,
                        'max_value': mapping.max_value,
                        'curve': mapping.curve,
                        'enabled': mapping.enabled
                    }
                    for mapping in self.mappings.values()
                ]
            }
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logging.info(f"MIDI configuration saved to {config_file}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save MIDI configuration: {e}")
            return False
    
    def _apply_curve(self, value: float, curve: str) -> float:
        """Apply response curve to control value."""
        if curve == "exponential":
            return value ** 2
        elif curve == "logarithmic":
            return np.sqrt(value) if value >= 0 else -np.sqrt(-value)
        elif curve == "inverse":
            return 1.0 - value
        else:  # linear
            return value
    
    def _normalize_value(self, value: float, mapping: MidiMapping) -> int:
        """Convert normalized value (0-1) to MIDI value range."""
        # Apply curve
        curved_value = self._apply_curve(value, mapping.curve)
        
        # Scale to MIDI range
        midi_value = int(curved_value * (mapping.max_value - mapping.min_value) + mapping.min_value)
        
        # Clamp to valid range
        return max(mapping.min_value, min(mapping.max_value, midi_value))
    
    def update_control(self, control_name: str, value: float, force_send: bool = False):
        """
        Update a control value and queue MIDI message if needed.
        
        Args:
            control_name: Name of the control
            value: Normalized value (0.0 to 1.0)
            force_send: Force sending even if value hasn't changed much
        """
        if control_name not in self.mappings:
            return
        
        mapping = self.mappings[control_name]
        if not mapping.enabled:
            return
        
        # Clamp value to valid range
        value = max(0.0, min(1.0, value))
        
        # Convert to MIDI value
        midi_value = self._normalize_value(value, mapping)
        
        # Check if we should send update
        last_value = self.last_values.get(control_name, -1)
        value_changed = abs(midi_value - last_value) >= self.min_change_threshold
        
        if force_send or value_changed:
            # Queue MIDI message
            message = {
                'control_name': control_name,
                'mapping': mapping,
                'midi_value': midi_value,
                'normalized_value': value,
                'timestamp': time.time()
            }
            
            self.message_queue.put(message)
            self.last_values[control_name] = midi_value
            self.control_states[control_name] = value
    
    def trigger_button(self, button_name: str, pressed: bool = True):
        """
        Trigger a button press/release.
        
        Args:
            button_name: Name of the button
            pressed: True for press, False for release
        """
        if button_name not in self.mappings:
            return
        
        mapping = self.mappings[button_name]
        if not mapping.enabled:
            return
        
        # Queue button message
        message = {
            'control_name': button_name,
            'mapping': mapping,
            'midi_value': 127 if pressed else 0,
            'normalized_value': 1.0 if pressed else 0.0,
            'timestamp': time.time(),
            'is_button': True,
            'pressed': pressed
        }
        
        self.message_queue.put(message)
        self.button_states[button_name] = pressed
    
    def _process_messages(self):
        """Process queued MIDI messages in background thread."""
        while self.running:
            try:
                # Process up to max_messages_per_frame messages
                processed = 0
                while processed < self.max_messages_per_frame:
                    message = self.message_queue.get(timeout=0.01)
                    if message is None:
                        break
                    
                    self._send_midi_message(message)
                    processed += 1
                
                # Small delay to prevent excessive CPU usage
                if processed == 0:
                    time.sleep(0.001)
                    
            except Exception as e:
                logging.error(f"Error in MIDI message processing: {e}")
                time.sleep(0.01)
    
    def _send_midi_message(self, message: Dict):
        """Send a single MIDI message."""
        mapping = message['mapping']
        midi_value = message['midi_value']
        
        try:
            if mapping.message_type == MidiMessageType.CONTROL_CHANGE:
                self.midi_out.send_cc(mapping.channel, mapping.controller, midi_value)
            
            elif mapping.message_type == MidiMessageType.NOTE_ON:
                if message.get('is_button', False):
                    if message.get('pressed', True):
                        self.midi_out.send_note_on(mapping.channel, mapping.controller, midi_value)
                    else:
                        self.midi_out.send_note_off(mapping.channel, mapping.controller, 0)
                else:
                    self.midi_out.send_note_on(mapping.channel, mapping.controller, midi_value)
            
            elif mapping.message_type == MidiMessageType.PITCH_BEND:
                # Convert 0-127 to 14-bit pitch bend value
                pitch_value = int((midi_value / 127.0) * 16383)
                self.midi_out.send_pitch_bend(mapping.channel, pitch_value)
            
        except Exception as e:
            logging.error(f"Error sending MIDI message for {mapping.control_name}: {e}")
    
    def get_control_state(self, control_name: str) -> Optional[float]:
        """Get current state of a control."""
        return self.control_states.get(control_name)
    
    def get_button_state(self, button_name: str) -> bool:
        """Get current state of a button."""
        return self.button_states.get(button_name, False)
    
    def get_all_states(self) -> Dict[str, Any]:
        """Get all current control and button states."""
        return {
            'controls': self.control_states.copy(),
            'buttons': self.button_states.copy()
        }
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        midi_stats = self.midi_out.get_performance_stats()
        
        return {
            'midi_output': midi_stats,
            'queue_size': self.message_queue.size(),
            'dropped_messages': self.message_queue.dropped_messages,
            'active_controls': len(self.control_states),
            'active_buttons': sum(self.button_states.values()),
            'total_mappings': len(self.mappings),
            'enabled_mappings': sum(1 for m in self.mappings.values() if m.enabled)
        }
    
    def reset_all_controls(self):
        """Reset all controls to default values."""
        for control_name in self.mappings:
            self.update_control(control_name, 0.0, force_send=True)
        
        for button_name in self.button_states:
            self.trigger_button(button_name, False)
    
    def close(self):
        """Close MIDI controller and cleanup resources."""
        self.running = False
        
        # Wait for processing thread to finish
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        # Send all remaining messages
        remaining_messages = self.message_queue.get_all()
        for message in remaining_messages:
            self._send_midi_message(message)
        
        # Close MIDI output
        self.midi_out.close()
        
        logging.info("DJMidiController closed")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing MIDI Controller...")
    
    # Create MIDI controller
    midi_controller = DJMidiController("DJ Controller Test")
    
    try:
        if midi_controller.midi_out.is_connected:
            print("MIDI port created successfully!")
            
            # Test control updates
            print("Testing control updates...")
            
            # Crossfader sweep
            for value in [0.0, 0.25, 0.5, 0.75, 1.0, 0.5]:
                midi_controller.update_control("crossfader", value)
                time.sleep(0.1)
            
            # Volume controls
            midi_controller.update_control("volume_a", 0.8)
            midi_controller.update_control("volume_b", 0.6)
            
            # Button presses
            midi_controller.trigger_button("play_a", True)
            time.sleep(0.1)
            midi_controller.trigger_button("play_a", False)
            
            # Performance stats
            time.sleep(0.5)  # Let messages process
            stats = midi_controller.get_performance_stats()
            print(f"Performance stats: {stats}")
            
            print("MIDI test completed successfully!")
        else:
            print("Failed to create MIDI port")
    
    except KeyboardInterrupt:
        print("Test interrupted by user")
    
    finally:
        midi_controller.close()
        print("MIDI controller closed")