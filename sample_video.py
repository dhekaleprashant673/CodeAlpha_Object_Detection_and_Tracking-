import cv2
import numpy as np
import math

def create_sample_video(output_file='sample_video.mp4', duration=5, fps=30):
    """Create a sample video with moving rectangles for testing object detection.
    
    Args:
        output_file (str): Path to save the output video file
        duration (int): Duration of the video in seconds
        fps (int): Frames per second
        
    Returns:
        bool: True if video was created successfully, False otherwise
    """
    # Video settings
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    
    try:
        # Create video writer
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"Error: Could not create video writer for {output_file}")
            return False
        
        # Generate video frames
        total_frames = duration * fps
        for i in range(total_frames):
            # Create a blank frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw some rectangles that move
            t = i / fps  # time in seconds
            
            # Red rectangle
            x1 = int(100 + 100 * np.sin(t))
            y1 = int(100 + 50 * np.cos(t))
            cv2.rectangle(frame, (x1, y1), (x1 + 50, y1 + 50), (0, 0, 255), -1)
            
            # Green rectangle
            x2 = int(300 + 80 * np.cos(t * 0.7))
            y2 = int(200 + 70 * np.sin(t * 0.7))
            cv2.rectangle(frame, (x2, y2), (x2 + 70, y2 + 70), (0, 255, 0), -1)
            
            # Blue rectangle
            x3 = int(400 + 60 * np.sin(t * 1.2))
            y3 = int(300 + 40 * np.cos(t * 1.2))
            cv2.rectangle(frame, (x3, y3), (x3 + 60, y3 + 60), (255, 0, 0), -1)
            
            # Add frame number
            cv2.putText(frame, f"Frame: {i}/{total_frames}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Write frame to video
            out.write(frame)
            
            # Display progress
            if i % fps == 0:
                print(f"Creating video: {i/total_frames*100:.1f}% complete")
        
        # Release resources
        out.release()
        print(f"Sample video created successfully: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error creating sample video: {e}")
        return False

if __name__ == "__main__":
    create_sample_video()