import os
import numpy as np
from pyproj import CRS, Transformer
import subprocess
import re

class GeoLocator:
    def __init__(self, config):
        self.camera_params = config['Camera']
        self.confidence_threshold = float(config['GeoLocator']['confidence_threshold'])

    def process_image(self, img_path, detection_results):
        """Main processing pipeline"""
        metadata = self._extract_metadata(img_path)
        drone_position = self._calculate_drone_position(metadata)
        return self._process_detections(drone_position, detection_results, metadata)

    def _extract_metadata(self, img_path):
        process = subprocess.Popen(['exiftool', img_path], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.STDOUT,
                                   universal_newlines=True)
        return {k.strip(): v.strip() for k, _, v in (line.partition(':') for line in process.stdout)}

    def _calculate_drone_position(self, metadata):
        lat = self.dms_to_dd(metadata['GPS Latitude'])
        lon = self.dms_to_dd(metadata['GPS Longitude'])
        relative_alt = float(metadata['Relative Altitude'].replace('+', '').strip())
        return [lat, lon, relative_alt]  # Use relative altitude directly

    def _process_detections(self, drone_position, detections, metadata):
        results = []
        for det in detections:
            if det['confidence'] > self.confidence_threshold:
                geo_coords = self._calculate_geolocation(det['bbox'], drone_position, metadata)
                results.append({**det, **geo_coords})
        return results

    def _calculate_geolocation(self, bbox, drone_position, metadata):
        roll = float(metadata.get('Camera Roll', '0'))
        pitch = float(metadata.get('Camera Pitch', '0'))
        yaw = float(metadata.get('Camera Yaw', '0'))
        focal_length_mm = float(metadata['Focal Length'].split()[0])
        sensor_width_mm = float(self.camera_params['sensor_width_mm'])
        sensor_height_mm = float(self.camera_params['sensor_height_mm'])
        image_width_px = float(metadata['Exif Image Width'])
        image_height_px = float(metadata['Exif Image Height'])

        object_pixel_x, object_pixel_y = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
        altitude = float(drone_position[2])  # Use relative altitude

        camera_matrix = self.get_camera_matrix(focal_length_mm, sensor_width_mm, sensor_height_mm, image_width_px, image_height_px)
        rotation_matrix = self.get_combined_rotation_matrix(roll, pitch, yaw)
        camera_coordinates = self.get_rotated_point(object_pixel_x, object_pixel_y, camera_matrix, rotation_matrix, altitude)

        utm_num = self.utmzone(drone_position[1], drone_position[0])
        n_or_s = metadata['GPS Latitude'][-1]
        drone_crs_epsg = self.epsgcode(utm_num, n_or_s)
        wgs84_epsg = 4326  # EPSG code for WGS84

        object_latitude, object_longitude = self.translate_coordinates(camera_coordinates, drone_position, drone_crs_epsg, wgs84_epsg)
        return {'latitude': object_latitude, 'longitude': object_longitude, 'elevation': altitude}

    def dms_to_dd(self, dms_string):
        degrees, minutes, seconds, direction = re.findall(r"[\d.]+|[NSEW]", dms_string)
        degrees = float(degrees)
        minutes = float(minutes)
        seconds = float(seconds)
        dd = degrees + minutes / 60 + seconds / 3600
        if direction in ['S', 'W']:
            dd *= -1
        return dd

    def epsgcode(self, utm_num, n_or_s):
        return int(f"326{utm_num}" if n_or_s == "N" else f"327{utm_num}")

    def utmzone(self, lon, lat):
        if -180 <= lon <= 180:
            return int((lon + 180) // 6) % 60 + 1
        raise ValueError(f"Invalid longitude: {lon}")

    def get_camera_matrix(self, focal_length_mm, sensor_width_mm, sensor_height_mm, image_width_px, image_height_px):
        fx = (focal_length_mm / sensor_width_mm) * image_width_px
        fy = (focal_length_mm / sensor_height_mm) * image_height_px
        cx = image_width_px / 2
        cy = image_height_px / 2
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def create_rotation_matrix_x(self, roll):
        roll_rad = np.radians(roll)
        return np.array([[1, 0, 0], [0, np.cos(roll_rad), -np.sin(roll_rad)], [0, np.sin(roll_rad), np.cos(roll_rad)]])

    def create_rotation_matrix_y(self, pitch):
        pitch_rad = np.radians(pitch)
        return np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)], [0, 1, 0], [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])

    def create_rotation_matrix_z(self, yaw):
        yaw_rad = np.radians(yaw)
        return np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0], [np.sin(yaw_rad), np.cos(yaw_rad), 0], [0, 0, 1]])

    def get_combined_rotation_matrix(self, roll, pitch, yaw):
        return self.create_rotation_matrix_z(yaw) @ self.create_rotation_matrix_y(pitch) @ self.create_rotation_matrix_x(roll)

    def get_rotated_point(self, object_pixel_x, object_pixel_y, camera_matrix, rotation_matrix, altitude):
        camera_point = np.linalg.inv(camera_matrix) @ np.array([object_pixel_x, object_pixel_y, 1])
        camera_point *= altitude
        return rotation_matrix @ camera_point

    def translate_coordinates(self, camera_coordinates, drone_position, drone_crs_epsg, wgs84_epsg):
        wgs84_crs = CRS.from_epsg(wgs84_epsg)
        drone_crs = CRS.from_epsg(drone_crs_epsg)
        transformer_to_local = Transformer.from_crs(wgs84_crs, drone_crs, always_xy=True)
        transformer_to_wgs84 = Transformer.from_crs(drone_crs, wgs84_crs, always_xy=True)
        x0, y0 = transformer_to_local.transform(drone_position[1], drone_position[0])
        x = x0 + camera_coordinates[0]
        y = y0 + camera_coordinates[1]
        return transformer_to_wgs84.transform(x, y)