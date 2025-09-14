import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO
from collections import defaultdict

# Simple implementation of SORT (Simple Online and Realtime Tracking)
class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = defaultdict(dict)
        self.frame_count = 0
        self.id_count = 0

    def _iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        # Convert to [x1, y1, x2, y2] format
        bbox1_x1, bbox1_y1, bbox1_w, bbox1_h = bbox1
        bbox1_x2, bbox1_y2 = bbox1_x1 + bbox1_w, bbox1_y1 + bbox1_h
        
        bbox2_x1, bbox2_y1, bbox2_w, bbox2_h = bbox2
        bbox2_x2, bbox2_y2 = bbox2_x1 + bbox2_w, bbox2_y1 + bbox2_h
        
        # Calculate intersection area
        x_left = max(bbox1_x1, bbox2_x1)
        y_top = max(bbox1_y1, bbox2_y1)
        x_right = min(bbox1_x2, bbox2_x2)
        y_bottom = min(bbox1_y2, bbox2_y2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = (bbox1_x2 - bbox1_x1) * (bbox1_y2 - bbox1_y1)
        bbox2_area = (bbox2_x2 - bbox2_x1) * (bbox2_y2 - bbox2_y1)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou

    def update(self, detections):
        """Update trackers with new detections"""
        self.frame_count += 1
        
        # No detections
        if len(detections) == 0:
            # Age all trackers and remove dead ones
            trackers_to_del = []
            for track_id in self.trackers:
                self.trackers[track_id]['age'] += 1
                if self.trackers[track_id]['age'] > self.max_age:
                    trackers_to_del.append(track_id)
            
            for track_id in trackers_to_del:
                del self.trackers[track_id]
                
            return []
        
        # We have detections
        if len(self.trackers) == 0:
            # First detections, initialize trackers
            for i, det in enumerate(detections):
                self.trackers[self.id_count] = {
                    'bbox': det,
                    'age': 0,
                    'hits': 1,
                    'class_id': det[4],
                    'conf': det[5]
                }
                self.id_count += 1
        else:
            # Match detections to existing trackers
            matched_indices = []
            unmatched_detections = []
            
            # For each detection, find the best matching tracker
            for i, det in enumerate(detections):
                best_iou = self.iou_threshold
                best_tracker_id = None
                
                for tracker_id, tracker in self.trackers.items():
                    # Only match with same class
                    if tracker['class_id'] != det[4]:
                        continue
                        
                    iou = self._iou(tracker['bbox'][:4], det[:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_tracker_id = tracker_id
                
                if best_tracker_id is not None:
                    # Update the matched tracker
                    self.trackers[best_tracker_id]['bbox'] = det
                    self.trackers[best_tracker_id]['age'] = 0
                    self.trackers[best_tracker_id]['hits'] += 1
                    self.trackers[best_tracker_id]['conf'] = det[5]
                    matched_indices.append(best_tracker_id)
                else:
                    # No match found, add as new detection
                    unmatched_detections.append(det)
            
            # Age trackers that weren't matched
            trackers_to_del = []
            for tracker_id in self.trackers:
                if tracker_id not in matched_indices:
                    self.trackers[tracker_id]['age'] += 1
                    
                if self.trackers[tracker_id]['age'] > self.max_age:
                    trackers_to_del.append(tracker_id)
            
            # Remove dead trackers
            for tracker_id in trackers_to_del:
                del self.trackers[tracker_id]
            
            # Add new trackers for unmatched detections
            for det in unmatched_detections:
                self.trackers[self.id_count] = {
                    'bbox': det,
                    'age': 0,
                    'hits': 1,
                    'class_id': det[4],
                    'conf': det[5]
                }
                self.id_count += 1
        
        # Return active trackers
        result = []
        for tracker_id, tracker in self.trackers.items():
            if tracker['hits'] >= self.min_hits:
                x, y, w, h, class_id, conf = tracker['bbox']
                result.append([x, y, w, h, class_id, conf, tracker_id])
                
        return result


class ObjectDetectionTracking:
    def __init__(self, source=0, model_path="yolov8n.pt"):
        """
        Initialize the object detection and tracking system
        
        Args:
            source: Video source (0 for webcam, or path to video file)
            model_path: Path to YOLO model weights
        """
        self.source = source
        self.model = YOLO(model_path)
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
        
    def _process_detections(self, frame, results):
        """Process YOLO detections and convert to format for tracker"""
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates (convert to [x, y, w, h])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w, h = x2 - x1, y2 - y1
                
                # Get class and confidence
                cls = int(box.cls.cpu().numpy()[0])
                conf = float(box.conf.cpu().numpy()[0])
                
                # Only keep detections with confidence > 0.5
                if conf > 0.5:
                    detections.append([x1, y1, w, h, cls, conf])
        
        return detections
    
    def run(self, output_file=None, max_frames=1500):
        """Run object detection and tracking on video stream
        
        Args:
            output_file: If provided, save output to this file instead of displaying
            max_frames: Maximum number of frames to process (default: 150)
        """
        print(f"Attempting to open video source: {self.source}")
        
        # Open video capture
        try:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                print(f"Error: Could not open video source {self.source}")
                print("Please check if your webcam is connected or if the video file exists.")
                return
            
            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Video source opened successfully: {frame_width}x{frame_height} at {fps} FPS")
            
            # Create video writer if output file is specified
            if output_file:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
                print(f"Saving output to: {output_file}")
        except Exception as e:
            print(f"Exception when opening video source: {e}")
            return
        
        # Main loop
        frame_count = 0
        # Use the max_frames parameter instead of hardcoded value
        
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
            
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            
            # Process detections
            detections = self._process_detections(frame, results)
            
            # Update tracker
            tracks = self.tracker.update(detections)
            
            # Draw tracks on frame
            for track in tracks:
                x, y, w, h, class_id, conf, track_id = track
                
                # Get color for this track ID
                color = self.colors[int(track_id) % len(self.colors)].tolist()
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                
                # Get class name
                class_name = self.model.names[int(class_id)]
                
                # Draw label
                label = f"{class_name} #{int(track_id)} {conf:.2f}"
                cv2.putText(frame, label, (int(x), int(y - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Calculate FPS
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save or display frame
            if output_file:
                out.write(frame)
                print(f"Processed frame {frame_count}/{max_frames}", end="\r")
            else:
                # In a headless environment, this might not work
                try:
                    cv2.imshow("Object Detection and Tracking", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    print(f"Error displaying frame: {e}")
                    if not output_file:  # If we're not saving to a file, exit
                        break
            
            frame_count += 1
            if frame_count >= max_frames:
                print(f"\nReached maximum frame count ({max_frames})")
                break
        
        # Release resources
        cap.release()
        if output_file:
            out.release()
            print(f"\nOutput saved to {output_file}")
        
        try:
            cv2.destroyAllWindows()
        except:
            pass


if __name__ == "__main__":
    import os
    import sys
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Object Detection and Tracking")
    parser.add_argument("--max-frames", type=int, default=300, 
                        help="Maximum number of frames to process (default: 300)")
    args = parser.parse_args()
    
    # Define output file
    output_file = "output_detection.mp4"
    
    # Check if sample video exists
    sample_video = "sample_video.mp4"
    if os.path.exists(sample_video):
        print(f"Using sample video: {sample_video}")
        print(f"Processing up to {args.max_frames} frames")
        detector = ObjectDetectionTracking(source=sample_video)
        detector.run(output_file=output_file, max_frames=args.max_frames)
        print(f"\nProcessing complete! Output saved to {output_file}")
    else:
        print("Sample video not found. Creating one first...")
        from sample_video import create_sample_video
        if create_sample_video():
            print(f"\nNow processing the sample video...")
            print(f"Processing up to {args.max_frames} frames")
            detector = ObjectDetectionTracking(source=sample_video)
            detector.run(output_file=output_file, max_frames=args.max_frames)
            print(f"\nProcessing complete! Output saved to {output_file}")
        else:
            print("Failed to create sample video.")
            sys.exit(1)