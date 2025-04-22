import cv2

def resize_image(image, target_size=(640, 640)):
    return cv2.resize(image, target_size)