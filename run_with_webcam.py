import cv2
import argparse
from object_detection_tracking import ObjectDetectionTracking

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run object detection and tracking with webcam")
    parser.add_argument("--max-frames", type=int, default=300, 
                        help="Maximum number of frames to process (default: 300)")
    args = parser.parse_args()
    
    # Check if webcam is available
    print("Checking webcam availability...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Please check your camera connection.")
        return False
    
    # Webcam is available, release it and use it with our detector
    cap.release()
    print("Webcam is available. Starting object detection and tracking...")
    print(f"Processing up to {args.max_frames} frames")
    
    # Initialize detector with webcam
    detector = ObjectDetectionTracking(source=0)
    
    # Run detection and tracking (no output file to display in window)
    detector.run(max_frames=args.max_frames)
    
    return True

if __name__ == "__main__":
    main()