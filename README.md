# Computer Vision DJ Controller

A real-time hand tracking DJ controller that uses computer vision to control DJ software via MIDI. Uses webcam-based hand tracking with MediaPipe to detect gestures on a calibrated DJ board surface and translates them into MIDI control messages.

![DJ Controller Demo](docs/demo.gif)

## Features

### Core Functionality
- **Real-time Hand Tracking**: MediaPipe-based hand detection and landmark tracking
- **Multiple Calibration Methods**: 
  - ArUco marker-based homography for custom surfaces
  - **Physical DJ Controller Detection**: Automatic recognition of popular DJ controllers
  - **Paper Template Detection**: Printable PDF templates with automatic layout detection
- **Gesture Recognition**: Detect tap, drag, rotate, and hold gestures
- **MIDI Output**: Virtual MIDI port with configurable CC and Note mappings
- **Signal Smoothing**: EMA filtering and hysteresis to reduce jitter
- **Real-time Visualization**: Live overlay showing hands, zones, and control values

### Advanced Features
- **Machine Learning Integration**: PyTorch model support for gesture classification
- **Data Logging**: CSV export for training data collection
- **Configurable Zones**: JSON-based control zone layout
- **Performance Optimization**: <70ms latency target
- **Multi-hand Support**: Track up to 2 hands simultaneously

### Supported DJ Controllers
- **Pioneer DDJ-SB3**: Full layout with jog wheels, faders, EQ knobs, and buttons
- **Numark Party Mix**: Compact 2-deck layout detection  
- **Paper Templates**: Customizable printable layouts (A3/A4 sizes)
- **Custom Controllers**: Extensible detection system for new layouts

## System Requirements

- Python 3.8+
- Webcam (USB or built-in)
- 4 ArUco markers (6x6 dictionary, IDs 0-3)
- DJ software with MIDI input support

### Supported Platforms
- Linux (tested on Ubuntu 20.04+)
- Windows 10/11
- macOS 10.15+

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/asiannasekai/ds-340-final-project.git
cd ds-340-final-project
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Create Configuration Files
```bash
python src/main.py --create-config
```

## Hardware Setup

### 1. Print ArUco Markers
Print the 4 ArUco markers from `docs/aruco_markers.pdf`:
- Marker ID 0: Top-left corner
- Marker ID 1: Top-right corner  
- Marker ID 2: Bottom-right corner
- Marker ID 3: Bottom-left corner

### 2. Create DJ Board Surface

**Option A: ArUco Markers (Traditional)**
- Recommended size: 40cm x 30cm
- Place markers at corners on a flat surface
- Ensure good lighting and minimal shadows
- Position camera 30-50cm above the surface

**Option B: Physical DJ Controller**
- Compatible with popular controllers (Pioneer DDJ-SB3, Numark Party Mix, etc.)
- Ensure controller is well-lit and visible to camera
- Position camera to capture entire controller surface
- Automatic detection and zone mapping

**Option C: Paper Template**
- Print provided PDF templates on A3/A4 paper
- Use `python generate_paper_templates.py` to create templates
- Place on flat surface with good lighting
- Automatic detection of printed layout

### 3. Camera Setup
- Mount camera directly above the DJ board
- Ensure all 4 markers are visible in the frame
- Use good lighting to avoid shadows
- Test different angles for optimal marker detection

## Quick Start

### 1. Run Calibration Test
```bash
cd src
python calibration.py
```
- **ArUco Method**: Place markers at board corners, wait for "CALIBRATED" status
- **DJ Board Method**: Position physical controller or paper template in view
- **Auto-Detection**: System automatically detects available calibration methods
- Press 'q' to quit when satisfied

### 1b. Test DJ Board Detection (New!)
```bash
python demo_dj_board_detection.py
```
- Test detection of physical controllers and paper templates
- Press 1-3 for different detection modes
- Press 'z' to toggle control zone overlay
- Verify detection accuracy before main usage

### 2. Test Hand Tracking
```bash
python capture.py
```
- Verify hand detection works properly
- Check FPS performance (should be >20 FPS)
- Press 'q' to quit

### 3. Run Full Application
```bash
python main.py --config ../config/app_config.json
```

### 4. Connect to DJ Software
- Open your DJ software (e.g., Virtual DJ, Serato, Traktor)
- Look for "DJ Controller" in MIDI input devices
- Map controls as needed

## Usage

### Basic Controls
- **Q**: Quit application
- **SPACE**: Pause/resume processing
- **D**: Toggle debug info display
- **Z**: Toggle control zones display
- **H**: Toggle hand tracking display
- **O**: Toggle board overlay
- **C**: Force recalibration (ArUco + DJ board detection)
- **B**: Force DJ board detection
- **R**: Reset all MIDI controls
- **L**: Toggle data logging
- **S**: Save current configuration

### Control Zones (Default Layout)
- **Crossfader**: Center bottom (horizontal drag)
- **Volume A/B**: Left/right sides (vertical drag)
- **Play A/B**: Left/right buttons (tap)
- **Cue A/B**: Left/right buttons (tap)
- **Filter A/B**: Top corners (rotate)
- **EQ Controls**: Side knobs (rotate)
- **XY Pad**: Center area (2D drag)

### Gesture Types
- **Tap**: Quick touch and release for buttons
- **Drag**: Sustained movement for faders
- **Rotate**: Circular motion for knobs
- **Hold**: Stationary contact for toggles

## Configuration

### Main Configuration (`config/app_config.json`)
```json
{
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
  "smoothing": {
    "method": "EMA",
    "alpha": 0.3,
    "hysteresis_threshold": 0.01
  }
}
```

### Control Zones (`config/zones.json`)
Define custom control layouts:
```json
{
  "zones": [
    {
      "name": "crossfader",
      "zone_type": "fader",
      "bounds": [[0.15, 0.25], [0.25, 0.30]],
      "midi_channel": 1,
      "midi_cc": 8,
      "orientation": "horizontal"
    }
  ]
}
```

### MIDI Mappings (`config/midi_mappings.json`)
Customize MIDI output:
```json
{
  "midi_mappings": [
    {
      "control_name": "crossfader",
      "message_type": "control_change",
      "channel": 0,
      "controller": 8,
      "curve": "linear"
    }
  ]
}
```

## Data Collection & Training

### Enable Data Logging
```bash
python main.py --config config/app_config.json
# Press 'L' to start logging
```

### Training Data Format
Logged data includes:
- Hand landmark positions (CSV)
- Gesture events with labels (CSV)
- MIDI messages with timing (CSV)
- Session metadata (JSON)

### Machine Learning Integration
```python
from src.model_inference import ModelManager, ModelConfig

# Load trained gesture classifier
model_manager = ModelManager()
config = ModelConfig(
    model_type=ModelType.GESTURE_CLASSIFIER,
    model_path="models/gesture_classifier.pth",
    class_names=['tap', 'drag', 'rotate', 'hold']
)
model_manager.load_model("gestures", config)

# Use in main loop
prediction = model_manager.predict_gesture("gestures", hand_landmarks)
```

## Troubleshooting

### Common Issues

**Camera not detected:**
- Check camera permissions
- Try different device IDs (0, 1, 2)
- Ensure camera isn't used by other applications

**Poor calibration quality:**
- Improve lighting conditions
- Clean camera lens
- Print markers at correct size (5cm recommended)
- Ensure markers are flat and clearly visible

**High latency:**
- Reduce camera resolution
- Lower hand tracking complexity
- Disable unnecessary features
- Close other applications

**MIDI not working:**
- Check virtual MIDI port creation
- Verify DJ software MIDI settings
- Try restarting the application
- Check firewall/security settings

### Performance Optimization

**For better FPS:**
```json
{
  "camera": {
    "width": 640,
    "height": 480,
    "fps": 30
  },
  "hand_tracking": {
    "model_complexity": 0,
    "max_hands": 1
  }
}
```

**For lower latency:**
```json
{
  "smoothing": {
    "alpha": 0.8,
    "hysteresis_threshold": 0.005
  },
  "midi": {
    "max_messages_per_frame": 20
  }
}
```

## Development

### Project Structure
```
ds-340-final-project/
├── src/
│   ├── main.py              # Main application
│   ├── capture.py           # Camera & hand tracking
│   ├── calibration.py       # ArUco calibration
│   ├── gestures.py          # Gesture recognition
│   ├── smoothing.py         # Signal filtering
│   ├── midi_controller.py   # MIDI output
│   ├── data_logger.py       # Data collection
│   ├── model_inference.py   # ML integration
│   └── utils.py            # Utility functions
├── config/                  # Configuration files
├── data/                   # Logged session data
├── models/                 # Trained ML models
└── docs/                  # Documentation
```

### Adding New Gesture Types
1. Define gesture in `gestures.py`:
```python
class GestureType(Enum):
    MY_GESTURE = "my_gesture"
```

2. Implement detection logic:
```python
def _detect_my_gesture(self, hand_state, timestamp):
    # Detection logic here
    return is_detected
```

3. Add to recognition pipeline:
```python
if self._detect_my_gesture(hand_state, timestamp):
    # Create gesture event
    pass
```

### Adding New Control Zones
1. Update `config/zones.json`:
```json
{
  "name": "new_control",
  "zone_type": "knob",
  "bounds": [[x1, y1], [x2, y2]],
  "midi_channel": 1,
  "midi_cc": 75
}
```

2. Add MIDI mapping in `config/midi_mappings.json`

### Custom ML Models
1. Train model using logged data
2. Export to PyTorch format
3. Create model config:
```python
config = ModelConfig(
    model_type=ModelType.GESTURE_CLASSIFIER,
    model_path="models/my_model.pth",
    input_size=feature_size,
    output_size=num_classes
)
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 src/
black src/

# Generate documentation
sphinx-build -b html docs/ docs/_build/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for hand tracking
- [OpenCV](https://opencv.org/) for computer vision
- [PyTorch](https://pytorch.org/) for machine learning
- [python-rtmidi](https://pypi.org/project/python-rtmidi/) for MIDI support

## Citation

If you use this project in academic work, please cite:
```bibtex
@software{dj_controller_cv,
  title={Computer Vision DJ Controller},
  author={Your Name},
  year={2025},
  url={https://github.com/asiannasekai/ds-340-final-project}
}
```

## Support

- 📧 Email: support@example.com
- 💬 Discord: [Join Server](https://discord.gg/example)
- 📖 Wiki: [Project Wiki](https://github.com/asiannasekai/ds-340-final-project/wiki)
- 🐛 Issues: [Report Bug](https://github.com/asiannasekai/ds-340-final-project/issues)