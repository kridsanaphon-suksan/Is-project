import torch
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path, confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, img):
        results = self.model(img)
        return self._format_results(results)

    def _format_results(self, results):
        formatted = []
        for result in results:
            for box in result.boxes:
                x, y, w, h = box.xywh[0].cpu().numpy()  # Extract bounding box as [x, y, width, height]
                confidence = box.conf.cpu().item()
                class_name = box.cls.cpu().item()
                formatted.append({
                    'bbox': [x, y, w, h],
                    'confidence': confidence,
                    'class_name': class_name
                })
        return formatted
