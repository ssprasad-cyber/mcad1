import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class UCFDataset(Dataset):
    def __init__(self, frames_dir, annotations=None, window_size=10, transform=None):
        """
        UCF-Crime Dataset for frame-level or window-level anomaly detection.
        frames_dir: Path to extracted frames organized by category/video_name/frame.png
        annotations: Optional dictionary with video_name -> [(start, end), ...] temporal segments
        """
        self.frames_dir = Path(frames_dir)
        self.window_size = window_size
        self.transform = transform
        self.samples = []
        
        self._prepare_samples(annotations)

    def _prepare_samples(self, annotations):
        # Walk through the category directories
        if not self.frames_dir.exists():
            print(f"Warning: {self.frames_dir} does not exist.")
            return

        for category_dir in self.frames_dir.iterdir():
            if not category_dir.is_dir(): continue
            
            label_base = 0 if category_dir.name.lower() == "normalvideos" else 1
            
            # Group frames by video name
            video_frames_map = {}
            for f_path in category_dir.glob("*.png"):
                # Filename format: VideoName_FrameIdx.png
                # e.g. Abuse001_x264_0.png -> Abuse001_x264
                parts = f_path.stem.split('_')
                video_name = "_".join(parts[:-1])
                try:
                    frame_idx = int(parts[-1])
                except ValueError:
                    continue # Skip if last part is not an integer
                
                if video_name not in video_frames_map:
                    video_frames_map[video_name] = []
                video_frames_map[video_name].append((frame_idx, f_path))
            
            for video_name, frames in video_frames_map.items():
                # Sort frames by index
                frames.sort()
                frame_files = [f for idx, f in frames]
                
                if len(frame_files) < self.window_size:
                    continue
                
                # Create sliding windows
                for i in range(0, len(frame_files) - self.window_size + 1):
                    window_frames = frame_files[i : i + self.window_size]
                    
                    # Determine label for the window
                    if annotations and video_name in annotations:
                        is_anomaly = False
                        for f in window_frames:
                            idx = int(f.stem.split('_')[-1])
                            for start, end in annotations[video_name]:
                                if start <= idx <= end:
                                    is_anomaly = True
                                    break
                            if is_anomaly: break
                        window_label = 1 if is_anomaly else 0
                    else:
                        window_label = label_base
                    
                    self.samples.append({
                        "frames": window_frames,
                        "label": window_label
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = []
        for f_path in sample["frames"]:
            img = cv2.imread(str(f_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            frames.append(img)
            
        return {
            "frames": torch.stack(frames) if isinstance(frames[0], torch.Tensor) else np.stack(frames),
            "label": torch.tensor(sample["label"], dtype=torch.float32)
        }

def parse_annotations(file_path):
    annotations = {}
    if not os.path.exists(file_path):
        return annotations
        
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4: continue
            
            video_file = parts[0]
            video_name = video_file.split('.')[0]
            
            segments = []
            for i in range(2, len(parts), 2):
                if i + 1 >= len(parts): break
                start = int(parts[i])
                end = int(parts[i+1])
                if start == -1 or end == -1: continue
                segments.append((start, end))
            
            if segments:
                annotations[video_name] = segments
    return annotations

if __name__ == "__main__":
    # Test loading
    ann_path = "/home/ssaiprasad/projects/mcad1/MCAD_V2/data/ucf_crime/Temporal_Anomaly_Annotation.txt"
    anns = parse_annotations(ann_path)
    
    dataset = UCFDataset(
        frames_dir="/home/ssaiprasad/projects/MCAD/data/raw/Train", 
        annotations=anns,
        window_size=10
    )
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        print(f"First sample label: {dataset[0]['label']}")
        # Count labels
        labels = [s["label"] for s in dataset.samples]
        print(f"Normal samples: {labels.count(0)}, Anomaly samples: {labels.count(1)}")
