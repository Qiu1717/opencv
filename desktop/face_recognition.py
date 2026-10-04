import cv2
import numpy as np

class FaceRecognizer:
    def __init__(self, model_path=None):
        self.transform_size = (150, 150)
    
    def preprocess_face(self, face_image):
        if len(face_image.shape) == 2:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
        face_image = cv2.resize(face_image, self.transform_size)
        return face_image
    
    def extract_feature(self, face_image):
        face_image = self.preprocess_face(face_image)
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        hog = cv2.HOGDescriptor((150, 150), (30, 30), (15, 15), (15, 15), 9)
        feature = hog.compute(gray)
        feature = feature.flatten()
        
        if np.linalg.norm(feature) > 0:
            feature = feature / np.linalg.norm(feature)
        
        return feature
    
    def calculate_similarity(self, feature1, feature2):
        min_len = min(len(feature1), len(feature2))
        if min_len == 0:
            return 0.0
        
        f1 = feature1[:min_len]
        f2 = feature2[:min_len]
        
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        f1_norm = f1 / norm1
        f2_norm = f2 / norm2
        
        cosine_sim = np.dot(f1_norm, f2_norm)
        
        return cosine_sim
    
    def match_feature(self, query_feature, database_features, threshold=0.6):
        best_match = None
        best_score = -1
        
        for student_id, feature in database_features.items():
            score = self.calculate_similarity(query_feature, feature)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = student_id
        
        return best_match, best_score