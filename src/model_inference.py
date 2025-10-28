"""
PyTorch model integration for gesture recognition and control prediction.
Supports loading trained models for gesture classification and fader position estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import time
from enum import Enum


class ModelType(Enum):
    """Supported model types."""
    GESTURE_CLASSIFIER = "gesture_classifier"
    FADER_REGRESSOR = "fader_regressor"
    HAND_POSE_ESTIMATOR = "hand_pose_estimator"
    ZONE_DETECTOR = "zone_detector"


@dataclass
class ModelConfig:
    """Configuration for a trained model."""
    model_type: ModelType
    model_path: str
    input_size: int
    output_size: int
    class_names: Optional[List[str]] = None
    normalization_params: Optional[Dict] = None
    preprocessing_config: Optional[Dict] = None
    confidence_threshold: float = 0.5
    enabled: bool = True


class HandLandmarkFeatureExtractor:
    """
    Extract features from MediaPipe hand landmarks for ML models.
    """
    
    def __init__(self, normalize: bool = True, include_velocity: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            normalize: Whether to normalize coordinates
            include_velocity: Whether to include velocity features
        """
        self.normalize = normalize
        self.include_velocity = include_velocity
        self.previous_landmarks = None
        self.previous_timestamp = None
        
        # Important landmark indices for gesture recognition
        self.key_landmarks = [
            0,   # WRIST
            4,   # THUMB_TIP
            8,   # INDEX_FINGER_TIP
            12,  # MIDDLE_FINGER_TIP
            16,  # RING_FINGER_TIP
            20   # PINKY_TIP
        ]
    
    def extract_features(self, landmarks: List[Dict], timestamp: Optional[float] = None) -> np.ndarray:
        """
        Extract feature vector from hand landmarks.
        
        Args:
            landmarks: List of landmark dictionaries with x, y, z coordinates
            timestamp: Optional timestamp for velocity calculation
            
        Returns:
            Feature vector as numpy array
        """
        if len(landmarks) < 21:
            # Pad with zeros if insufficient landmarks
            padded_landmarks = landmarks + [{'x_norm': 0, 'y_norm': 0, 'z': 0}] * (21 - len(landmarks))
            landmarks = padded_landmarks
        
        features = []
        
        # Basic coordinate features
        for i, landmark in enumerate(landmarks):
            x = landmark.get('x_norm', 0.0)
            y = landmark.get('y_norm', 0.0)
            z = landmark.get('z', 0.0)
            
            features.extend([x, y, z])
        
        # Relative positions (distances from wrist)
        if len(landmarks) > 0:
            wrist_x = landmarks[0].get('x_norm', 0.0)
            wrist_y = landmarks[0].get('y_norm', 0.0)
            wrist_z = landmarks[0].get('z', 0.0)
            
            for landmark in landmarks[1:]:  # Skip wrist itself
                rel_x = landmark.get('x_norm', 0.0) - wrist_x
                rel_y = landmark.get('y_norm', 0.0) - wrist_y
                rel_z = landmark.get('z', 0.0) - wrist_z
                
                # Distance from wrist
                distance = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2)
                features.append(distance)
        
        # Finger angles and distances
        finger_tips = [4, 8, 12, 16, 20]
        finger_bases = [2, 5, 9, 13, 17]
        
        for tip_idx, base_idx in zip(finger_tips, finger_bases):
            if tip_idx < len(landmarks) and base_idx < len(landmarks):
                tip = landmarks[tip_idx]
                base = landmarks[base_idx]
                
                # Vector from base to tip
                dx = tip.get('x_norm', 0.0) - base.get('x_norm', 0.0)
                dy = tip.get('y_norm', 0.0) - base.get('y_norm', 0.0)
                
                # Angle
                angle = np.arctan2(dy, dx)
                features.append(angle)
                
                # Length
                length = np.sqrt(dx**2 + dy**2)
                features.append(length)
        
        # Velocity features
        if self.include_velocity and timestamp is not None:
            velocities = self._calculate_velocities(landmarks, timestamp)
            features.extend(velocities)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_velocities(self, landmarks: List[Dict], timestamp: float) -> List[float]:
        """Calculate velocity features."""
        velocities = []
        
        if self.previous_landmarks is not None and self.previous_timestamp is not None:
            dt = timestamp - self.previous_timestamp
            if dt > 0:
                for i, (current, previous) in enumerate(zip(landmarks, self.previous_landmarks)):
                    dx = current.get('x_norm', 0.0) - previous.get('x_norm', 0.0)
                    dy = current.get('y_norm', 0.0) - previous.get('y_norm', 0.0)
                    
                    vx = dx / dt
                    vy = dy / dt
                    velocity_magnitude = np.sqrt(vx**2 + vy**2)
                    
                    # Only include velocities for key landmarks to reduce dimensionality
                    if i in self.key_landmarks:
                        velocities.extend([vx, vy, velocity_magnitude])
        
        # Pad with zeros if no previous data
        if not velocities:
            velocities = [0.0] * (len(self.key_landmarks) * 3)
        
        # Update history
        self.previous_landmarks = landmarks.copy()
        self.previous_timestamp = timestamp
        
        return velocities
    
    def get_feature_size(self) -> int:
        """Get the size of the feature vector."""
        base_size = 21 * 3  # 21 landmarks × 3 coordinates
        relative_size = 20   # 20 distances from wrist
        finger_size = 5 * 2  # 5 fingers × (angle + length)
        velocity_size = len(self.key_landmarks) * 3 if self.include_velocity else 0
        
        return base_size + relative_size + finger_size + velocity_size


class GestureClassifier(nn.Module):
    """
    Neural network for gesture classification from hand landmarks.
    """
    
    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 128):
        """
        Initialize gesture classifier.
        
        Args:
            input_size: Size of input feature vector
            num_classes: Number of gesture classes
            hidden_size: Size of hidden layers
        """
        super(GestureClassifier, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Feature extraction layers
        self.feature_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        """Forward pass."""
        features = self.feature_layers(x)
        logits = self.classifier(features)
        return logits


class FaderRegressor(nn.Module):
    """
    Neural network for fader position regression from hand landmarks.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64):
        """
        Initialize fader regressor.
        
        Args:
            input_size: Size of input feature vector
            hidden_size: Size of hidden layers
        """
        super(FaderRegressor, self).__init__()
        
        self.input_size = input_size
        
        # Regression layers
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
    def forward(self, x):
        """Forward pass."""
        return self.layers(x)


class ModelManager:
    """
    Manages loading and inference with trained PyTorch models.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model manager.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = models_dir
        self.loaded_models: Dict[str, Dict] = {}
        self.feature_extractor = HandLandmarkFeatureExtractor()
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"ModelManager using device: {self.device}")
        
        # Performance tracking
        self.inference_times = []
        self.total_inferences = 0
    
    def load_model(self, model_name: str, config: ModelConfig) -> bool:
        """
        Load a trained model.
        
        Args:
            model_name: Name identifier for the model
            config: Model configuration
            
        Returns:
            True if model loaded successfully
        """
        try:
            if not os.path.exists(config.model_path):
                logging.error(f"Model file not found: {config.model_path}")
                return False
            
            # Load model state dict
            checkpoint = torch.load(config.model_path, map_location=self.device)
            
            # Create model instance based on type
            if config.model_type == ModelType.GESTURE_CLASSIFIER:
                model = GestureClassifier(
                    input_size=config.input_size,
                    num_classes=config.output_size
                )
            elif config.model_type == ModelType.FADER_REGRESSOR:
                model = FaderRegressor(input_size=config.input_size)
            else:
                logging.error(f"Unsupported model type: {config.model_type}")
                return False
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            # Store loaded model
            self.loaded_models[model_name] = {
                'model': model,
                'config': config,
                'loaded_at': time.time()
            }
            
            logging.info(f"Model '{model_name}' loaded successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load model '{model_name}': {e}")
            return False
    
    def predict_gesture(self, model_name: str, landmarks: List[Dict], 
                       timestamp: Optional[float] = None) -> Optional[Dict]:
        """
        Predict gesture using a loaded classifier model.
        
        Args:
            model_name: Name of the loaded model
            landmarks: Hand landmarks
            timestamp: Optional timestamp for velocity features
            
        Returns:
            Prediction dictionary with class, confidence, and probabilities
        """
        if model_name not in self.loaded_models:
            return None
        
        model_info = self.loaded_models[model_name]
        model = model_info['model']
        config = model_info['config']
        
        if config.model_type != ModelType.GESTURE_CLASSIFIER:
            return None
        
        try:
            start_time = time.time()
            
            # Extract features
            features = self.feature_extractor.extract_features(landmarks, timestamp)
            
            # Prepare input tensor
            input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                logits = model(input_tensor)
                probabilities = F.softmax(logits, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
            
            # Convert to numpy
            probabilities = probabilities.cpu().numpy()[0]
            predicted_class = predicted_class.cpu().item()
            confidence = confidence.cpu().item()
            
            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.total_inferences += 1
            
            # Prepare result
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities.tolist(),
                'inference_time_ms': inference_time * 1000
            }
            
            # Add class name if available
            if config.class_names and predicted_class < len(config.class_names):
                result['class_name'] = config.class_names[predicted_class]
            
            # Check confidence threshold
            result['above_threshold'] = confidence >= config.confidence_threshold
            
            return result
            
        except Exception as e:
            logging.error(f"Error during gesture prediction: {e}")
            return None
    
    def predict_fader_position(self, model_name: str, landmarks: List[Dict],
                              timestamp: Optional[float] = None) -> Optional[float]:
        """
        Predict fader position using a loaded regressor model.
        
        Args:
            model_name: Name of the loaded model
            landmarks: Hand landmarks
            timestamp: Optional timestamp for velocity features
            
        Returns:
            Predicted fader position (0.0 to 1.0) or None if error
        """
        if model_name not in self.loaded_models:
            return None
        
        model_info = self.loaded_models[model_name]
        model = model_info['model']
        config = model_info['config']
        
        if config.model_type != ModelType.FADER_REGRESSOR:
            return None
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(landmarks, timestamp)
            
            # Prepare input tensor
            input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                prediction = model(input_tensor)
            
            # Convert to float
            position = prediction.cpu().item()
            
            return position
            
        except Exception as e:
            logging.error(f"Error during fader position prediction: {e}")
            return None
    
    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model names."""
        return list(self.loaded_models.keys())
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            logging.info(f"Model '{model_name}' unloaded")
            return True
        return False
    
    def get_performance_stats(self) -> Dict:
        """Get model inference performance statistics."""
        if not self.inference_times:
            return {'total_inferences': 0}
        
        recent_times = self.inference_times[-100:]  # Last 100 inferences
        
        return {
            'total_inferences': self.total_inferences,
            'average_inference_time_ms': np.mean(self.inference_times) * 1000,
            'recent_average_ms': np.mean(recent_times) * 1000,
            'min_inference_time_ms': np.min(self.inference_times) * 1000,
            'max_inference_time_ms': np.max(self.inference_times) * 1000,
            'loaded_models': len(self.loaded_models),
            'device': str(self.device)
        }


def create_sample_models():
    """Create sample model files for testing."""
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Create sample gesture classifier
    input_size = HandLandmarkFeatureExtractor().get_feature_size()
    gesture_classes = ['tap', 'drag', 'rotate', 'hold', 'none']
    
    gesture_model = GestureClassifier(input_size, len(gesture_classes))
    
    # Save model
    torch.save({
        'model_state_dict': gesture_model.state_dict(),
        'config': {
            'input_size': input_size,
            'num_classes': len(gesture_classes),
            'class_names': gesture_classes
        }
    }, "models/gesture_classifier.pth")
    
    # Create sample fader regressor
    fader_model = FaderRegressor(input_size)
    
    torch.save({
        'model_state_dict': fader_model.state_dict(),
        'config': {
            'input_size': input_size
        }
    }, "models/fader_regressor.pth")
    
    # Create model configuration
    model_configs = {
        'gesture_classifier': {
            'model_type': 'gesture_classifier',
            'model_path': 'models/gesture_classifier.pth',
            'input_size': input_size,
            'output_size': len(gesture_classes),
            'class_names': gesture_classes,
            'confidence_threshold': 0.7,
            'enabled': True
        },
        'fader_regressor': {
            'model_type': 'fader_regressor',
            'model_path': 'models/fader_regressor.pth',
            'input_size': input_size,
            'output_size': 1,
            'confidence_threshold': 0.0,
            'enabled': True
        }
    }
    
    with open("models/model_configs.json", 'w') as f:
        json.dump(model_configs, f, indent=2)
    
    print("Sample models created in models/ directory")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing PyTorch Model Integration...")
    
    # Create sample models
    print("Creating sample models...")
    create_sample_models()
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # Load gesture classifier
    gesture_config = ModelConfig(
        model_type=ModelType.GESTURE_CLASSIFIER,
        model_path="models/gesture_classifier.pth",
        input_size=HandLandmarkFeatureExtractor().get_feature_size(),
        output_size=5,
        class_names=['tap', 'drag', 'rotate', 'hold', 'none'],
        confidence_threshold=0.5
    )
    
    success = model_manager.load_model("gesture_classifier", gesture_config)
    print(f"Gesture classifier loaded: {success}")
    
    # Test prediction with dummy data
    if success:
        # Create dummy hand landmarks
        dummy_landmarks = [
            {'x_norm': 0.5 + 0.01 * i, 'y_norm': 0.5, 'z': 0.0} 
            for i in range(21)
        ]
        
        # Predict gesture
        result = model_manager.predict_gesture("gesture_classifier", dummy_landmarks)
        print(f"Gesture prediction: {result}")
    
    # Performance stats
    stats = model_manager.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    print("PyTorch model integration test completed!")