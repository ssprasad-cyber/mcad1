import torch
import numpy as np
from ultralytics import YOLO
import cv2

class FeatureExtractor:
    def __init__(self, model_path='models/yolov8n.pt', device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.model = YOLO(model_path).to(self.device)
        self.person_class_id = 0  # YOLOv8 person class index

    def detect_persons(self, frame):
        """
        Detect persons in a frame.
        Returns: list of [x, y, w, h] normalized by frame size
        """
        results = self.model(frame, verbose=False, classes=[self.person_class_id])[0]
        boxes = results.boxes.xywhn.cpu().numpy()  # [x, y, w, h] normalized
        return boxes

    def extract_window_features(self, frames):
        """
        Extract features for a sequence of frames.
        frames: numpy array of shape (T, H, W, C)
        Returns: list of frame-level person features
        """
        T = len(frames)
        all_frame_detections = []
        
        for i in range(T):
            detections = self.detect_persons(frames[i])
            all_frame_detections.append(detections)
            
        # Compute velocity by matching detections between frames
        # Simple approach: for each detection in frame t, find nearest in t-1
        all_features = []
        
        for t in range(T):
            frame_detections = all_frame_detections[t]
            frame_features = []
            
            for i, det in enumerate(frame_detections):
                x, y, w, h = det
                vx, vy = 0.0, 0.0
                
                if t > 0:
                    prev_detections = all_frame_detections[t-1]
                    if len(prev_detections) > 0:
                        # Find nearest neighbor in previous frame
                        dists = np.linalg.norm(prev_detections[:, :2] - det[:2], axis=1)
                        nearest_idx = np.argmin(dists)
                        if dists[nearest_idx] < 0.1:  # Threshold for matching
                            vx = x - prev_detections[nearest_idx, 0]
                            vy = y - prev_detections[nearest_idx, 1]
                
                motion_mag = np.sqrt(vx**2 + vy**2)
                
                # Feature vector: [x, y, w, h, vx, vy, mag]
                feat = [x, y, w, h, vx, vy, motion_mag]
                frame_features.append(feat)
                
            all_features.append(np.array(frame_features) if frame_features else np.empty((0, 7)))
            
        return all_features

if __name__ == "__main__":
    extractor = FeatureExtractor()
    # Dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    feats = extractor.detect_persons(dummy_frame)
    print(f"Detected {len(feats)} persons in dummy frame.")
