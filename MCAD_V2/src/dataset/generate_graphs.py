import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
import argparse

from src.dataset.ucf_dataset import UCFDataset, parse_annotations
from src.features.feature_extractor import FeatureExtractor
from src.graph.graph_constructor import GraphConstructor

def generate_graphs(frames_dir, output_dir, annotation_path=None, window_size=10):
    anns = None
    if annotation_path:
        anns = parse_annotations(annotation_path)
        print(f"Loaded annotations for {len(anns)} videos")

    extractor = FeatureExtractor()
    constructor = GraphConstructor()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # We'll use the UCFDataset class to get the list of window frames
    dataset = UCFDataset(frames_dir=frames_dir, annotations=anns, window_size=window_size)
    print(f"Total sequences to process: {len(dataset)}")

    for idx in tqdm(range(len(dataset)), desc="Generating graphs"):
        sample = dataset.samples[idx]
        
        # Load frames
        frames = []
        for f_path in sample["frames"]:
            img = cv2.imread(str(f_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        
        # Extract features
        all_window_features = extractor.extract_window_features(frames)
        
        # Build graphs
        graphs = constructor.build_sequence_graphs(all_window_features)
        
        # Save as a dictionary
        data = {
            "graphs": graphs,
            "label": sample["label"]
        }
        
        torch.save(data, output_dir / f"sample_{idx}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--annotation_path", type=str, default=None)
    args = parser.parse_args()
    
    generate_graphs(args.frames_dir, args.output_dir, args.annotation_path)
