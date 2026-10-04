import os
import cv2
import numpy as np
import pickle
from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(PROJECT_ROOT, 'backend', 'models', 'insightface', 'models', 'buffalo_l')

class FeatureExtractor:
    def __init__(self):
        self.net = cv2.dnn.readNetFromONNX(os.path.join(MODEL_DIR, 'w600k_r50.onnx'))
        print("Model loaded")

    def extract_feature(self, face_img):
        if len(face_img.shape) == 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
        
        face = cv2.resize(face_img, (112, 112)).astype(np.float32)
        face = (face - 127.5) / 127.5
        face = np.expand_dims(np.transpose(face, (2, 0, 1)), 0)
        self.net.setInput(face)
        f = self.net.forward()[0]
        return f / np.linalg.norm(f)

def load_image(filepath):
    try:
        img_pil = Image.open(filepath)
        img = np.array(img_pil)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    except:
        return None

def prepare_dataset(data_dir, extractor):
    faces = []
    labels = []

    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        print("Processing " + person_name + "...")
        count = 0
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = load_image(img_path)
            if img is None:
                continue
            
            if img.shape[0] > 20 and img.shape[1] > 20:
                feature = extractor.extract_feature(img)
                faces.append(feature)
                labels.append(person_name)
                count += 1
        
        print("  Extracted " + str(count) + " features")

    return np.array(faces), np.array(labels)

def train_classifier(faces, labels):
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    if len(faces) < 5:
        X_train, X_test, y_train, y_test = faces, faces, labels_encoded, labels_encoded
    else:
        X_train, X_test, y_train, y_test = train_test_split(faces, labels_encoded, test_size=0.2, random_state=42)

    clf = SVC(kernel='rbf', C=10, gamma=0.01, probability=True)
    clf.fit(X_train, y_train)

    if len(X_test) > 0:
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("\nModel accuracy: " + str(accuracy * 100) + "%")

    return clf, le

def save_model(clf, le, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, 'backend', 'models')

    model_data = {
        'classifier': clf,
        'label_encoder': le,
        'model_type': 'svm_classifier'
    }

    path = os.path.join(output_dir, 'classifier.pkl')
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)

    print("\nClassifier saved to: " + path)
    return path

if __name__ == '__main__':
    print("=" * 50)
    print("  Face Recognition Model Training")
    print("=" * 50)

    data_dir = os.path.join(PROJECT_ROOT, 'dataset', 'augmented')

    if not os.path.exists(data_dir):
        print("Error: " + data_dir + " does not exist")
        exit()

    print("\nLoading model...")
    extractor = FeatureExtractor()

    print("\nExtracting face features...")
    faces, labels = prepare_dataset(data_dir, extractor)

    if len(faces) == 0:
        print("Error: No face features extracted")
        exit()

    unique_persons = np.unique(labels)
    print("\nTotal: " + str(len(faces)) + " faces | " + str(len(unique_persons)) + " persons")

    print("\nTraining classifier...")
    clf, le = train_classifier(faces, labels)

    save_model(clf, le)
    print("\nTraining completed! Restart backend service to use new model")