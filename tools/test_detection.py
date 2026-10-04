import cv2
import numpy as np
import os
from PIL import Image
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(PROJECT_ROOT, 'backend', 'models', 'insightface')

def test_insightface_detection(image_path):
    print(f"\nTesting image: {image_path}")
    
    try:
        img_pil = Image.open(image_path).convert('RGB')
        frame = np.array(img_pil)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        print(f"  Image shape: {frame_bgr.shape}")
        
        import insightface
        from insightface.app import FaceAnalysis
        
        app = FaceAnalysis(name='buffalo_l', root=MODEL_DIR)
        app.prepare(ctx_id=-1, det_size=(640, 640))
        
        print("  Running InsightFace detection...")
        faces = app.get(frame_bgr)
        print(f"  Detected {len(faces)} faces")
        
        for i, face in enumerate(faces):
            print(f"  Face {i}: confidence={face.det_score:.4f}, bbox={face.bbox}")
        
        if len(faces) == 0:
            print("  Trying different det_size...")
            app.prepare(ctx_id=-1, det_size=(1280, 1280))
            faces = app.get(frame_bgr)
            print(f"  Detected {len(faces)} faces with det_size=1280")
            for i, face in enumerate(faces):
                print(f"  Face {i}: confidence={face.det_score:.4f}, bbox={face.bbox}")
        
        return len(faces) > 0
        
    except Exception as e:
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_images = [
        os.path.join(PROJECT_ROOT, 'test1.jpg'),
        os.path.join(PROJECT_ROOT, 'test2.jpg'),
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            test_insightface_detection(img_path)
        else:
            print(f"Image not found: {img_path}")