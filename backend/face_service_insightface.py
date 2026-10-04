import cv2
import numpy as np
import os
import pickle

class InsightFaceService:
    def __init__(self, model_name='buffalo_l', model_dir='models/insightface'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.app = None
        self.detector = None
        self._load_models(model_name)
        
        self.feature_db_path = os.path.join(model_dir, 'features.pkl')
        self.features_db = self._load_features()
        self._migrate_features()
        
        self.confidence_threshold = 0.10
        self.similarity_threshold = 0.55
        self._features_migrated = False
    
    def _load_models(self, model_name):
        self._try_insightface(model_name)
        self._ensure_fallback_dnn()
    
    def _try_insightface(self, model_name):
        try:
            import insightface
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(name=model_name, root=self.model_dir,
                                     allowed_modules=['detection', 'recognition'])
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            print("OK: InsightFace " + model_name + " loaded (det 640x640)")
        except Exception as e:
            print("InsightFace init fail: " + str(e) + ", using DNN/Haar")
            self.app = None
    
    def _ensure_fallback_dnn(self):
        prototxt = os.path.join('models', 'deploy.prototxt.txt')
        caffemodel = os.path.join('models', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')
        if os.path.exists(prototxt) and os.path.exists(caffemodel):
            self.detector = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
            print("DNN detector ready (fallback)")
    
    def _load_features(self):
        if os.path.exists(self.feature_db_path):
            with open(self.feature_db_path, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def _save_features(self):
        with open(self.feature_db_path, 'wb') as f:
            pickle.dump(self.features_db, f)

    def _insightface_detect(self, frame):
        faces = self.app.get(frame)
        results = []
        for f in faces:
            if f.det_score < self.confidence_threshold:
                continue
            bbox = f.bbox.astype(int)
            results.append({
                'box': (bbox[0], bbox[1], bbox[2], bbox[3]),
                'confidence': float(f.det_score),
                'feature': f.normed_embedding,
            })
            if hasattr(f, 'landmark') and f.landmark is not None:
                results[-1]['landmark'] = f.landmark.tolist()
        return sorted(results, key=lambda x: -x['confidence'])
    
    def _dnn_detect(self, frame):
        if self.detector is None:
            return []
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                      (300, 300), (104.0, 177.0, 123.0))
        self.detector.setInput(blob)
        detections = self.detector.forward()
        results = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > 0.20:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (sx, sy, ex, ey) = box.astype("int")
                results.append({'box': (sx, sy, ex, ey), 'confidence': float(conf)})
        return sorted(results, key=lambda x: -x['confidence'])
    
    def _haar_detect(self, frame):
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        h, w = gray.shape[:2]
        min_sz = max(int(min(w, h) * 0.04), 20)
        max_sz = int(min(w, h) * 0.9)
        
        for cp in ['haarcascade_frontalface_default.xml',
                   'haarcascade_frontalface_alt.xml',
                   'haarcascade_frontalface_alt2.xml']:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cp)
            if cascade.empty():
                continue
            faces = cascade.detectMultiScale(gray, 1.05, 2, minSize=(min_sz, min_sz),
                                              maxSize=(max_sz, max_sz))
            if len(faces) > 0:
                results = []
                for (x, y, fw, fh) in faces:
                    results.append({'box': (x, y, x+fw, y+fh), 'confidence': 0.80})
                return sorted(results, key=lambda x:
                    -(x['box'][2]-x['box'][0])*(x['box'][3]-x['box'][1]))
        return []
    
    def detect_faces(self, frame):
        if self.app is not None:
            results = self._insightface_detect(frame)
            if results:
                return results
        results = self._dnn_detect(frame)
        if results:
            return results
        return self._haar_detect(frame)
    
    def detect_faces_robust(self, frame, debug=False):
        if frame is None or len(frame.shape) < 2:
            return ([], {}) if debug else []
        h, w = frame.shape[:2]
        if w < 30 or h < 30:
            if debug:
                return ([], {'error': 'Image too small: ' + str(w) + 'x' + str(h)})
            return []
        
        log = [] if debug else None
        
        results = self.detect_faces(frame)
        if results:
            if debug:
                log.append('Standard detection: ' + str(len(results)) + ' faces')
            return results if not debug else (results, {'methods_succeeded': log})
        if debug:
            log.append('Standard detection: 0 faces')
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        enhanced = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
        results = self.detect_faces(enhanced)
        if results:
            if debug:
                log.append('CLAHE enhanced: ' + str(len(results)) + ' faces')
            return results if not debug else (results, {'methods_succeeded': log})
        if debug:
            log.append('CLAHE enhanced: 0 faces')
        
        for scale in [1.25, 0.75, 1.5, 0.5]:
            nw, nh = int(w * scale), int(h * scale)
            if nw < 40 or nh < 40 or nw > 2000 or nh > 2000:
                continue
            resized = cv2.resize(frame, (nw, nh))
            if self.app:
                results = self._insightface_detect(resized)
                if results:
                    inv = 1.0 / scale
                    for r in results:
                        r['box'] = tuple(int(v * inv) for v in r['box'])
                    if debug:
                        log.append('Scale ' + str(scale) + 'x: ' + str(len(results)) + ' faces')
                    return results if not debug else (results, {'methods_succeeded': log})
            results = self._haar_detect(resized)
            if results:
                inv = 1.0 / scale
                for r in results:
                    r['box'] = tuple(int(v * inv) for v in r['box'])
                if debug:
                    log.append('Haar scale ' + str(scale) + 'x: ' + str(len(results)) + ' faces')
                return results if not debug else (results, {'methods_succeeded': log})
        if debug:
            log.append('Multi-scale: 0 faces')
        
        clahe2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        eq2 = clahe2.apply(gray)
        enhanced2 = cv2.cvtColor(eq2, cv2.COLOR_GRAY2BGR)
        results = self._haar_detect(enhanced2)
        if results:
            if debug:
                log.append('CLAHE v2: ' + str(len(results)) + ' faces')
            return results if not debug else (results, {'methods_succeeded': log})
        if debug:
            log.append('CLAHE v2: 0 faces')
        
        if self.app:
            mirrored = cv2.flip(frame, 1)
            results = self._insightface_detect(mirrored)
            if results:
                for r in results:
                    x1, y1, x2, y2 = r['box']
                    r['box'] = (w - x2, y1, w - x1, y2)
                if debug:
                    log.append('Mirrored: ' + str(len(results)) + ' faces')
                return results if not debug else (results, {'methods_succeeded': log})
            if debug:
                log.append('Mirrored: 0 faces')
        
        if debug:
            return ([], {'methods_attempted': log, 'error': 'All detection methods failed'})
        return []
    
    def extract_feature(self, face_image):
        if self.app is not None:
            try:
                faces = self.app.get(face_image)
                if faces:
                    return faces[0].normed_embedding
            except:
                pass
        return self._extract_fallback_feature(face_image)
    
    def _extract_fallback_feature(self, face_image):
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        gray = cv2.resize(gray, (150, 150))
        hog = cv2.HOGDescriptor((150, 150), (30, 30), (15, 15), (15, 15), 9)
        feature = hog.compute(gray).flatten()
        norm = np.linalg.norm(feature)
        return feature / norm if norm > 0 else feature
    
    def calculate_similarity(self, feature1, feature2):
        min_len = min(len(feature1), len(feature2))
        if min_len == 0:
            return 0.0
        return float(np.dot(
            feature1[:min_len] / np.linalg.norm(feature1[:min_len]),
            feature2[:min_len] / np.linalg.norm(feature2[:min_len])
        ))
    
    def _migrate_features(self):
        for sid, data in self.features_db.items():
            if 'feature' in data and 'features' not in data:
                data['features'] = [data['feature']]
                del data['feature']
        self._features_migrated = True

    def _get_features(self, student_id):
        data = self.features_db.get(student_id)
        if not data:
            return []
        if 'features' in data:
            return data['features']
        if 'feature' in data:
            return [data['feature']]
        return []

    def match_feature(self, query_feature):
        best_match = None
        best_score = -1
        for student_id, data in self.features_db.items():
            features = self._get_features(student_id)
            for feat in features:
                score = self.calculate_similarity(query_feature, feat)
                if score > best_score:
                    best_score = score
                    best_match = student_id
        if best_score >= self.similarity_threshold:
            return best_match, best_score
        return None, best_score

    def register_face(self, student_id, feature, name=''):
        if student_id in self.features_db:
            data = self.features_db[student_id]
            if 'features' not in data:
                if 'feature' in data:
                    data['features'] = [data['feature']]
                    del data['feature']
                else:
                    data['features'] = []
            data['features'].append(feature)
            if name:
                data['name'] = name
        else:
            self.features_db[student_id] = {'features': [feature], 'name': name}
        self._save_features()

    def remove_face(self, student_id):
        if student_id in self.features_db:
            del self.features_db[student_id]
            self._save_features()
            return True
        return False
    
    def get_face_roi(self, frame, box):
        startX, startY, endX, endY = box
        startX = max(0, startX - 15)
        startY = max(0, startY - 15)
        endX = min(frame.shape[1], endX + 15)
        endY = min(frame.shape[0], endY + 15)
        return frame[startY:endY, startX:endY]

if __name__ == '__main__':
    print("Testing InsightFaceService...")
    service = InsightFaceService()
    if service.app:
        print("OK: InsightFace loaded and ready")
    else:
        print("WARN: InsightFace not loaded, using fallbacks")