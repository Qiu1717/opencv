import os
import cv2
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(PROJECT_ROOT, 'backend', 'models', 'insightface', 'models', 'buffalo_l')
FACE_DETECTOR = os.path.join(PROJECT_ROOT, 'backend', 'models', 'deploy.prototxt.txt')
FACE_DETECTOR_WEIGHTS = os.path.join(PROJECT_ROOT, 'backend', 'models', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')

class FeatureExtractor:
    def __init__(self):
        self.net = cv2.dnn.readNetFromONNX(os.path.join(MODEL_DIR, 'w600k_r50.onnx'))
        self.detector = cv2.dnn.readNetFromCaffe(FACE_DETECTOR, FACE_DETECTOR_WEIGHTS)

    def detect_face(self, img):
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
        self.detector.setInput(blob)
        detections = self.detector.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                return img[y1:y2, x1:x2]
        return None

    def extract_feature(self, face_img):
        face = cv2.resize(face_img, (112, 112))
        face = face.astype(np.float32)
        face = (face - 127.5) / 127.5
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)
        self.net.setInput(face)
        feature = self.net.forward()[0]
        return feature / np.linalg.norm(feature)

def prepare_dataset(data_dir, extractor):
    faces = []
    labels = []

    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        print(f"\n处理 {person_name}...")
        count = 0
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                face = extractor.detect_face(img)
                if face is not None:
                    feature = extractor.extract_feature(face)
                    faces.append(feature)
                    labels.append(person_name)
                    count += 1
            except Exception as e:
                print(f"  跳过 {img_name}: {e}")
        print(f"  成功提取 {count} 张人脸")

    return np.array(faces), np.array(labels)

def train_classifier(faces, labels):
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    if len(faces) < 5:
        print("数据太少，无法划分训练集/测试集，使用全部数据训练")
        X_train, y_train = faces, labels_encoded
        X_test, y_test = faces, labels_encoded
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            faces, labels_encoded, test_size=0.2, random_state=42
        )

    clf = SVC(kernel='rbf', C=10, gamma=0.1, probability=True)
    clf.fit(X_train, y_train)

    if len(X_test) > 0:
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n模型准确率: {accuracy * 100:.2f}%")

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

    print(f"\n分类器已保存到: {path}")
    return path

if __name__ == '__main__':
    print("=" * 50)
    print("  人脸识别模型训练")
    print("=" * 50)

    data_dir = os.path.join(PROJECT_ROOT, 'dataset')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"\n请先将人脸照片放入: {data_dir}")
        print("目录结构：")
        print("  dataset/")
        print("    ├── 张三/")
        print("    │   ├── 000.jpg")
        print("    │   └── 001.jpg")
        print("    └── 李四/")
        print("        ├── 000.jpg")
        print("        └── 001.jpg")
        print("\n提示: 运行 collect_faces.py 可以用摄像头采集人脸")
        exit()

    print("\n加载模型中...")
    extractor = FeatureExtractor()
    print("模型加载成功")

    print("\n提取人脸特征...")
    faces, labels = prepare_dataset(data_dir, extractor)

    if len(faces) == 0:
        print("错误: 未提取到任何人脸特征，请检查数据和光照条件")
        exit()

    unique_persons = np.unique(labels)
    print(f"\n总计: {len(faces)} 张人脸 | {len(unique_persons)} 个人")

    if len(unique_persons) < 2:
        print("警告: 只有1个人，训练效果有限。建议至少有2人以上")

    print("\n训练分类器...")
    clf, le = train_classifier(faces, labels)

    save_model(clf, le)
    print("\n训练完成！重启后端服务即可使用新模型")