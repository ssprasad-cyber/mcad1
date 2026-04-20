import cv2
import os
import argparse
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_path, output_dir, fps=10):
    """
    Extract frames from a video at a specific FPS.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        print(f"Error: Could not get FPS for {video_path}")
        return

    hop = round(video_fps / fps)
    if hop < 1: hop = 1

    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % hop == 0:
            frame_name = f"{video_path.stem}_{frame_count}.png"
            cv2.imwrite(str(output_dir / frame_name), frame)
            saved_count += 1
            
        frame_count += 1
        
    cap.release()
    return saved_count

def process_all_videos(raw_dir, output_root, fps=10):
    raw_dir = Path(raw_dir)
    output_root = Path(output_root)
    
    video_files = list(raw_dir.rglob("*.mp4")) + list(raw_dir.rglob("*.avi"))
    
    print(f"Found {len(video_files)} videos in {raw_dir}")
    
    for v_path in tqdm(video_files, desc="Extracting frames"):
        # Maintain directory structure
        rel_path = v_path.relative_to(raw_dir).parent
        target_dir = output_root / rel_path / v_path.stem
        extract_frames(v_path, target_dir, fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from UCF-Crime videos")
    parser.add_argument("--raw_dir", type=str, required=True, help="Path to raw videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save frames")
    parser.add_argument("--fps", type=int, default=10, help="Target FPS")
    
    args = parser.parse_args()
    process_all_videos(args.raw_dir, args.output_dir, args.fps)
