#!/usr/bin/env python3
"""
DJ Board Detection Demo
Demonstrates detection of physical DJ controllers and paper templates.
"""

import sys
import os
import time
import logging
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dj_board_detector import DJBoardDetector, DJBoardType
from capture import CameraCapture


def demo_dj_board_detection():
    """Demonstrate DJ board detection capabilities."""
    print("=== DJ Board Detection Demo ===")
    print()
    
    # Initialize components
    camera = CameraCapture()
    detector = DJBoardDetector()
    
    if not camera.is_opened:
        print("❌ Failed to initialize camera")
        return False
    
    print("📷 Camera initialized successfully")
    print("🎛️  DJ Board Detector ready")
    print()
    print("Available board layouts:")
    
    for name, layout in detector.board_layouts.items():
        print(f"  • {layout.name} ({layout.board_type.value})")
        print(f"    - Dimensions: {layout.expected_dimensions[0]}x{layout.expected_dimensions[1]} cm")
        print(f"    - Control zones: {len(layout.control_zones)}")
    
    print()
    print("Controls:")
    print("  1 - Detect physical controllers")
    print("  2 - Detect paper templates") 
    print("  3 - Auto-detect all types")
    print("  z - Toggle control zones overlay")
    print("  s - Save detection config")
    print("  l - Load detection config")
    print("  i - Show board info")
    print("  q - Quit")
    print()
    
    show_zones = False
    detection_count = 0
    
    try:
        while True:
            success, frame = camera.read_frame()
            
            if not success:
                print("❌ Failed to capture frame")
                break
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            detection_performed = False
            
            if key == ord('1'):
                print("🔍 Detecting physical controllers...")
                success, detection_info = detector.detect_board(frame, DJBoardType.PHYSICAL_CONTROLLER)
                detection_performed = True
                if success:
                    detection_count += 1
                    print(f"✅ Detected: {detection_info['layout_name']} (confidence: {detection_info['confidence']:.2f})")
                else:
                    print("❌ No physical controllers detected")
            
            elif key == ord('2'):
                print("🔍 Detecting paper templates...")
                success, detection_info = detector.detect_board(frame, DJBoardType.PAPER_TEMPLATE)
                detection_performed = True
                if success:
                    detection_count += 1
                    print(f"✅ Detected: {detection_info['layout_name']} (confidence: {detection_info['confidence']:.2f})")
                else:
                    print("❌ No paper templates detected")
            
            elif key == ord('3'):
                print("🔍 Auto-detecting...")
                success, detection_info = detector.detect_board(frame, DJBoardType.AUTO_DETECT)
                detection_performed = True
                if success:
                    detection_count += 1
                    print(f"✅ Auto-detected: {detection_info['layout_name']} (confidence: {detection_info['confidence']:.2f})")
                else:
                    print("❌ No boards detected")
            
            elif key == ord('z'):
                show_zones = not show_zones
                print(f"🎛️  Control zones overlay: {'ON' if show_zones else 'OFF'}")
            
            elif key == ord('s'):
                if detector.save_detection_config("config/dj_board_detection.json"):
                    print("💾 Detection config saved!")
                else:
                    print("❌ Failed to save detection config")
            
            elif key == ord('l'):
                if detector.load_detection_config("config/dj_board_detection.json"):
                    print("📂 Detection config loaded!")
                else:
                    print("❌ Failed to load detection config")
            
            elif key == ord('i'):
                if detector.is_detected:
                    print(f"\n📊 Current Detection Info:")
                    print(f"   Board: {detector.detected_board}")
                    print(f"   Confidence: {detector.detection_confidence:.2f}")
                    layout = detector.board_layouts[detector.detected_board]
                    print(f"   Type: {layout.board_type.value}")
                    print(f"   Dimensions: {layout.expected_dimensions[0]}x{layout.expected_dimensions[1]} cm")
                    print(f"   Control zones: {len(layout.control_zones)}")
                    
                    # Show some control zones
                    print("   Zones:")
                    for i, zone in enumerate(layout.control_zones[:5]):  # Show first 5
                        print(f"     • {zone['name']} ({zone['type']})")
                    if len(layout.control_zones) > 5:
                        print(f"     ... and {len(layout.control_zones) - 5} more")
                    print()
                else:
                    print("ℹ️  No board currently detected")
            
            elif key == ord('q'):
                break
            
            # Continuous auto-detection (every 15 frames for performance)
            if not detection_performed and cv2.getTickCount() % 15 == 0:
                success, detection_info = detector.detect_board(frame, DJBoardType.AUTO_DETECT)
            
            # Draw detection overlay
            if detector.is_detected:
                # Get the current detection info for overlay
                current_info = {
                    'detected': True,
                    'layout_name': detector.detected_board,
                    'confidence': detector.detection_confidence,
                    'corners': detector.board_corners,
                    'method': 'continuous'
                }
                
                frame = detector.draw_detection_overlay(frame, current_info)
                
                if show_zones:
                    frame = detector.draw_control_zones(frame)
            
            # Add status information
            status_color = (0, 255, 0) if detector.is_detected else (128, 128, 128)
            status_text = f"Detected: {detector.detected_board}" if detector.is_detected else "No board detected"
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Add detection count
            cv2.putText(frame, f"Detections: {detection_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Add instructions
            cv2.putText(frame, "Press 1-3 to detect, Z for zones, I for info, Q to quit", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('DJ Board Detection Demo', frame)
    
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    
    finally:
        camera.close()
        cv2.destroyAllWindows()
        
        print(f"\n📈 Demo Summary:")
        print(f"   Total detections: {detection_count}")
        print(f"   Final state: {'Board detected' if detector.is_detected else 'No board'}")
        if detector.is_detected:
            print(f"   Final board: {detector.detected_board}")
        print("   Demo completed successfully! ✅")
        
        return True


def main():
    """Main demo function."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("DJ Controller - Board Detection Demo")
    print("=" * 50)
    print()
    
    # Test DJ board detection
    if not demo_dj_board_detection():
        print("❌ DJ board detection demo failed")
        return 1
    
    print("\n🎉 All demos completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())