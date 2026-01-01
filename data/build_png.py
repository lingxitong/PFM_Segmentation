#!/usr/bin/env python3
"""
Convert single-class GeoJSON annotations (with hole support) to grayscale mask PNGs
that match the dimensions of SVS Level-0. Skip existing PNGs to avoid redundant computation.

This script processes medical image annotations and converts them to binary masks
for semantic segmentation training and evaluation.

Function: Convert GeoJSON annotations to PNG mask files
"""

import json
import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from openslide import OpenSlide
import logging
import argparse
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logs import setup_logging


class GeoJSONToMaskConverter:
    """
    Convert GeoJSON annotations to PNG mask files.
    
    This class handles the conversion of polygon annotations (including holes)
    to binary masks that match the dimensions of SVS files.
    
    Args:
        json_dir (str): Directory containing GeoJSON annotation files
        svs_dir (str): Directory containing SVS image files
        save_dir (str): Directory to save generated PNG masks
        mask_value (int): Grayscale value for foreground pixels (default: 255)
        min_contour_area (int): Minimum area threshold for valid contours (default: 100)
    """
    
    def __init__(self, json_dir: str, svs_dir: str, save_dir: str,
                 mask_value: int = 255, min_contour_area: int = 100):
        self.json_dir = json_dir
        self.svs_dir = svs_dir
        self.save_dir = save_dir
        self.mask_value = mask_value
        self.min_contour_area = min_contour_area
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
    
    def calculate_contour_area(self, points: np.ndarray) -> float:
        """
        Calculate the area of a contour.
        
        Args:
            points (np.ndarray): Contour points as numpy array
            
        Returns:
            float: Contour area
        """
        return cv2.contourArea(points.astype(np.float32))
    
    def fill_polygon_with_holes(self, mask: np.ndarray, rings: List[List[List[float]]]) -> None:
        """
        Fill a polygon with holes in the mask.
        
        Args:
            mask (np.ndarray): Binary mask array to fill
            rings (List[List[List[float]]]): List of coordinate rings where rings[0] is the outer contour
                                           and remaining rings are holes
        """
        outer = np.asarray(rings[0], dtype=np.int32)
        if self.calculate_contour_area(outer) >= self.min_contour_area:
            cv2.fillPoly(mask, [outer], self.mask_value)
        
        # Fill holes with background (0)
        for hole in rings[1:]:
            hole_pts = np.asarray(hole, dtype=np.int32)
            if self.calculate_contour_area(hole_pts) >= self.min_contour_area:
                cv2.fillPoly(mask, [hole_pts], 0)
    
    def generate_mask_from_geojson(self, json_path: str, svs_path: str, save_path: str) -> str:
        """
        Generate PNG mask from GeoJSON annotation and SVS file.
        
        Args:
            json_path (str): Path to GeoJSON annotation file
            svs_path (str): Path to SVS image file
            save_path (str): Path to save the generated PNG mask
            
        Returns:
            str: Status of the operation ('ok', 'skip-existing', 'error')
        """
        # Skip if PNG already exists
        if os.path.exists(save_path):
            return "skip-existing"
        
        try:
            # Load GeoJSON data
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            
            # Get SVS dimensions
            slide = OpenSlide(svs_path)
            width, height = slide.dimensions
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Process all features
            for feat in data.get("features", []):
                geom = feat.get("geometry", {})
                gtype = geom.get("type", "")
                coords = geom.get("coordinates", [])
                
                if gtype == "Polygon" and coords:
                    self.fill_polygon_with_holes(mask, coords)
                elif gtype == "MultiPolygon":
                    for poly in coords:
                        if poly:
                            self.fill_polygon_with_holes(mask, poly)
            
            # Save mask
            cv2.imwrite(save_path, mask)
            return "ok"
            
        except Exception as err:
            self.logger.error(f"Error processing {json_path}: {err}")
            return "error"
    
    def process_all_files(self) -> Dict[str, int]:
        """
        Process all JSON files in the directory.
        
        Returns:
            Dict[str, int]: Statistics of processing results
        """
        json_files = [
            os.path.join(self.json_dir, f) 
            for f in os.listdir(self.json_dir) 
            if f.endswith(".json")
        ]
        
        self.logger.info(f"Found {len(json_files)} JSON files, starting conversion...")
        
        stats = {
            "ok": 0, 
            "skip-existing": 0, 
            "skip-nosvs": 0, 
            "error": 0
        }
        
        for json_path in tqdm(json_files, desc="Generating masks"):
            base = os.path.splitext(os.path.basename(json_path))[0]
            svs_path = os.path.join(self.svs_dir, f"{base}.svs")
            save_path = os.path.join(self.save_dir, f"{base}.png")
            
            # Skip if SVS file doesn't exist
            if not os.path.exists(svs_path):
                stats["skip-nosvs"] += 1
                continue
            
            status = self.generate_mask_from_geojson(json_path, svs_path, save_path)
            stats[status] += 1
        
        return stats
    
    def print_statistics(self, stats: Dict[str, int]) -> None:
        """
        Print processing statistics.
        
        Args:
            stats (Dict[str, int]): Processing statistics
        """
        self.logger.info("\n=== Processing Complete ===")
        self.logger.info(f"Successfully generated: {stats['ok']}")
        self.logger.info(f"Skipped (already exists): {stats['skip-existing']}")
        self.logger.info(f"Skipped (missing SVS): {stats['skip-nosvs']}")
        self.logger.info(f"Errors: {stats['error']}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert GeoJSON annotations to PNG mask files'
    )
    
    parser.add_argument('--json-dir', type=str, default=None,
                        help='Directory containing GeoJSON annotation files')
    
    parser.add_argument('--svs-dir', type=str, default=None,
                        help='Directory containing SVS image files')
    
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save generated PNG masks')
    
    parser.add_argument('--mask-value', type=int, default=255,
                        help='Grayscale value for foreground pixels (default: 255)')
    
    parser.add_argument('--min-contour-area', type=int, default=100,
                        help='Minimum area threshold for valid contours (default: 100)')
    
    return parser.parse_args()


def main():
    """Main function to run the GeoJSON to mask conversion."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create converter instance
    converter = GeoJSONToMaskConverter(
        json_dir=args.json_dir,
        svs_dir=args.svs_dir,
        save_dir=args.save_dir,
        mask_value=args.mask_value,
        min_contour_area=args.min_contour_area
    )
    
    # Process all files
    stats = converter.process_all_files()
    
    # Print results
    converter.print_statistics(stats)


if __name__ == "__main__":
    main()
