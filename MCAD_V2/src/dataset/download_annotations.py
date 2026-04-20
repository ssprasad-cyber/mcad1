import requests
import os
from pathlib import Path

def download_annotations(output_path):
    url = "https://raw.githubusercontent.com/WaqasSultani/AnomalyDetectionCVPR2018/master/Temporal_Anomaly_Annotation.txt"
    print(f"Downloading annotations from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, "w") as f:
            f.write(response.text)
        print(f"Saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading annotations: {e}")
        return False

def parse_annotations(file_path):
    annotations = {}
    if not os.path.exists(file_path):
        return annotations
        
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3: continue
            
            video_name = parts[0].replace(".mp4", "")
            # Parts 1 is category, parts 2 and 3 are start/end frames
            # Some lines have multiple segments
            segments = []
            i = 2
            while i + 1 < len(parts):
                start = int(parts[i])
                end = int(parts[i+1])
                if start != -1:
                    segments.append((start, end))
                i += 2
            
            if segments:
                annotations[video_name] = segments
                
    return annotations

if __name__ == "__main__":
    target = "/home/ssaiprasad/projects/mcad1/MCAD_V2/data/ucf_crime/Temporal_Anomaly_Annotation.txt"
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if download_annotations(target):
        annos = parse_annotations(target)
        print(f"Parsed {len(annos)} annotated videos.")
