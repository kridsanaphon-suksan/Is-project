import cv2
import configparser
import os
from geolocation_module import GeoLocator
from yolo_wrapper import YOLODetector
from visualization import create_map, add_marker, save_map
import numpy as np
from preprocessing import resize_image
import logging

logging.basicConfig(level=logging.INFO)


class SARSystem:
    def __init__(self, config):
        self.geolocator = GeoLocator(config)
        self.detector = YOLODetector(config['Model']['model_path'])

    def resize_image(image, target_size=(640, 640)):
        return cv2.resize(image, target_size)
        
    def process_mission(self, img_paths):
        logging.info(f"Processing {len(img_paths)} images...")
        results = []
        for img_path in img_paths:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"Failed to read image: {img_path}")
                img = cv2.imread(img_path)
                img = resize_image(img)
                detections = self.detector.detect(img)
                geo_results = self.geolocator.process_image(img_path, detections)
                results.extend(geo_results)
            except Exception as e:
                logging.error(r"# filepath: e:\IS project\main.py")
                print(f"Error processing {img_path}: {e}")
        return results