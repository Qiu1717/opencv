import cv2
import numpy as np

class TailgatingDetector:
    def __init__(self):
        self.history = []
        self.max_history = 10
        self.detection_window = 3
        self.max_persons = 2
        self.distance_threshold = 150
    
    def detect_persons(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = cv2.createBackgroundSubtractorMOG2().apply(gray)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        persons = []
        for contour in contours:
            if cv2.contourArea(contour) > 5000:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                persons.append((center_x, center_y, w, h))
        
        return persons
    
    def check_tailgating(self, frame):
        persons = self.detect_persons(frame)
        self.history.append(len(persons))
        
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        recent_counts = self.history[-self.detection_window:]
        
        if len(recent_counts) >= self.detection_window:
            avg_count = sum(recent_counts) / len(recent_counts)
            max_count = max(recent_counts)
            
            if max_count >= self.max_persons and avg_count > 1:
                return True, persons
        
        return False, persons
    
    def draw_detection(self, frame, persons, is_tailgating):
        result_frame = frame.copy()
        
        for (x, y, w, h) in persons:
            color = (0, 0, 255) if is_tailgating else (0, 255, 0)
            cv2.rectangle(result_frame, (x - w//2, y - h//2), 
                         (x + w//2, y + h//2), color, 2)
        
        if is_tailgating:
            cv2.putText(result_frame, "TAILGATING DETECTED!", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return result_frame
