import sys
import os
import argparse
from object_detection_tracking import ObjectDetectionTracking

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process video with object detection and tracking")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--max-frames", type=int, default=300, 
                        help="Maximum number of frames to process (default: 300)")
    
    # Handle case when no arguments are provided
    if len(sys.argv) < 2:
        parser.print_help()
        return False
    
    args = parser.parse_args()
    
    # Check if the video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return False
    
    # Define output file name based on input file
    base_name = os.path.splitext(os.path.basename(args.video_path))[0]
    output_file = f"{base_name}_detection.mp4"
    
    print(f"Processing video: {args.video_path}")
    print(f"Output will be saved to: {output_file}")
    print(f"Processing up to {args.max_frames} frames")
    
    # Initialize detector with video file
    detector = ObjectDetectionTracking(source=args.video_path)
    
    # Run detection and tracking with max_frames parameter
    detector.run(output_file=output_file, max_frames=args.max_frames)
    
    print(f"\nProcessing complete! Output saved to {output_file}")
    return True

if __name__ == "__main__":
    main()