import argparse
import os
import configparser
import logging
import cv2
from main import SARSystem
from visualization import create_map, add_marker, save_map

logging.basicConfig(level=logging.INFO)

def get_image_paths(folder_path):
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(supported_formats)]

def visualize_results(image_path, results):
    # Load the original image
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]

    # YOLO typically resizes images to 640x640 during inference
    yolo_input_size = 640

    # Annotate the image with bounding boxes and geolocations
    for result in results:
        bbox = result['bbox']   

        # Scale bounding box coordinates back to the original image dimensions
        center_x = bbox[0] * original_width / yolo_input_size
        center_y = bbox[1] * original_height / yolo_input_size
        width = bbox[2] * original_width / yolo_input_size
        height = bbox[3] * original_height / yolo_input_size

        # Calculate top-left corner of the bounding box
        x = int(center_x - (width / 2))
        y = int(center_y - (height / 2))
        w = int(width)
        h = int(height)
        
        CLASS_LABELS = ["clothing"]  
        class_index = int(result['class_name'])  
        class_name = CLASS_LABELS[class_index]  

        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add class name, confidence, and geolocation
        label = f"{class_name} ({float(result['confidence']):.2f})"
        geo_text = f"Lat: {result['latitude']:.6f}, Lon: {result['longitude']:.6f}"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, geo_text, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Resize image for display
    screen_res = 1920, 1080
    scale_width = screen_res[0] / image.shape[1]
    scale_height = screen_res[1] / image.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(image.shape[1] * scale)
    window_height = int(image.shape[0] * scale)
    resized_image = cv2.resize(image, (window_width, window_height))

    # Save the annotated image
    output_image_path = "output_image.jpg"
    cv2.imwrite(output_image_path, image)
    print(f"Annotated image saved as: {output_image_path}")

    # Display the image
    cv2.imshow('Annotated Image', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    logging.info("Starting SAR Search Application...")
    parser = argparse.ArgumentParser(description="SAR Search")
    parser.add_argument('--image', type=str, help="Path to an image file or folder containing images")
    args = parser.parse_args()

    if not args.image:
        print("Please provide an image file or folder path using --image")
        return

    # Read configuration from config.ini
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Initialize SARSystem
    sar_system = SARSystem(config)

    # Get list of image paths
    if os.path.isdir(args.image):
        img_paths = get_image_paths(args.image)
    else:
        img_paths = [args.image]

    # Process mission
    results = sar_system.process_mission(img_paths)

    # Check if results are empty
    if not results:
        print("No objects detected in the images.")
        return

    corrected_results = results
    if results:
    # if corrected_results:
        center = (corrected_results[0]['latitude'], corrected_results[0]['longitude'])
        map_obj = create_map(center)
        for result in corrected_results:
            location = (result['longitude'],result['latitude'])
            popup_text = f"Class: {result['class_name']}, Confidence: {result['confidence']}"
            add_marker(map_obj, location, popup_text)

        # Save map to file
        save_map(map_obj, "output_map.html")
        print("Map saved as 'output_map.html'.")

    # Visualize results for each image
    for img_path in img_paths:
        visualize_results(img_path, corrected_results)

    # Print results
    for result in corrected_results:
        print(f"Latitude: {result['latitude']}, Longitude: {result['longitude']}, "
              f"Class: {result['class_name']}, Confidence: {result['confidence']}")

if __name__ == "__main__":
    main()