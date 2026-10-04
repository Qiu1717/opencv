import cv2
import numpy as np
from scipy import fftpack

class LivenessDetector:
    def __init__(self):
        self.min_face_size = 64
        self.blur_threshold = 100
        self.texture_threshold = 0.02
        self.motion_threshold = 0.5
        self.prev_frame = None
    
    def detect_blur(self, face):
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = np.var(laplacian)
        return blur_score, blur_score < self.blur_threshold
    
    def analyze_texture(self, face):
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        dft = fftpack.fft2(gray)
        dft_shift = fftpack.fftshift(dft)
        magnitude = 20 * np.log(np.abs(dft_shift) + 1)
        
        h, w = magnitude.shape
        center = (h // 2, w // 2)
        radius = min(h, w) // 8
        
        mask = np.ones_like(magnitude)
        cv2.circle(mask, center, radius, 0, -1)
        
        high_freq_energy = np.sum(magnitude * mask)
        total_energy = np.sum(magnitude)
        
        if total_energy == 0:
            texture_score = 0
        else:
            texture_score = high_freq_energy / total_energy
        
        return texture_score, texture_score < self.texture_threshold
    
    def detect_motion(self, prev_frame, curr_frame):
        if prev_frame is None:
            return 0, False
        
        # 检查尺寸是否一致
        if prev_frame.shape != curr_frame.shape:
            return 0, False
        
        try:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            avg_motion = np.mean(magnitude)
            return avg_motion, avg_motion < self.motion_threshold
        except:
            return 0, False
    
    def check_liveness(self, face, prev_face=None):
        results = []
        
        blur_score, is_blurry = self.detect_blur(face)
        results.append({
            'type': 'blur',
            'score': blur_score,
            'is_suspicious': is_blurry
        })
        
        texture_score, low_texture = self.analyze_texture(face)
        results.append({
            'type': 'texture',
            'score': texture_score,
            'is_suspicious': low_texture
        })
        
        if prev_face is not None:
            motion_score, no_motion = self.detect_motion(prev_face, face)
            results.append({
                'type': 'motion',
                'score': motion_score,
                'is_suspicious': no_motion
            })
        
        suspicious_count = sum(1 for r in results if r['is_suspicious'])
        is_live = suspicious_count < 2
        
        return is_live, results
