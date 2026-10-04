import cv2
import numpy as np
import os

class FaceDetector:
    def __init__(self):
        self.confidence_threshold = 0.5
        self.detector_type = 'haar'
        self.detector = None
        
        self._load_detector()
    
    def _load_detector(self):
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')
            proto_path = os.path.join(os.path.dirname(__file__), 'models', 'deploy.prototxt.txt')
            
            if os.path.exists(model_path) and os.path.exists(proto_path):
                self.detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
                self.detector_type = 'dnn'
                print("OK: DNN face detector loaded")
            else:
                raise FileNotFoundError("Model files not found")
        except Exception as e:
            print(f"Warning: DNN detector failed to load: {str(e)}")
            print("Falling back to Haar cascade")
            self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.detector_type = 'haar'
    
    def detect_faces(self, frame):
        if self.detector_type == 'dnn':
            return self._detect_dnn(frame)
        else:
            return self._detect_haar(frame)
    
    def _detect_dnn(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                      (300, 300), (104.0, 177.0, 123.0))
        self.detector.setInput(blob)
        detections = self.detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append({
                    'box': (startX, startY, endX, endY),
                    'confidence': confidence
                })
        return faces
    
    def _detect_haar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rect = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        
        faces = []
        for (x, y, w, h) in faces_rect:
            faces.append({
                'box': (x, y, x + w, y + h),
                'confidence': 0.9
            })
        return faces
    
    def apply_clahe(self, frame):
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR) if len(frame.shape) == 3 else equalized
    
    def get_face_roi(self, frame, box):
        (startX, startY, endX, endY) = box
        padding = 20
        startX = max(0, startX - padding)
        startY = max(0, startY - padding)
        endX = min(frame.shape[1], endX + padding)
        endY = min(frame.shape[0], endY + padding)
        return frame[startY:endY, startX:endX]