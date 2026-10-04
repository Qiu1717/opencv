import os
import cv2
import numpy as np
import pickle
import warnings
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

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
        best = None
        best_area = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best = (x1, y1, x2, y2)
        if best:
            x1, y1, x2, y2 = best
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
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
    faces, labels = [], []
    for person_name in sorted(os.listdir(data_dir)):
        person_dir = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        print(f"  处理 {person_name}...", end=" ")
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
            except:
                pass
        print(f"{count} 张")
    return np.array(faces), np.array(labels)

def round1_grid_search(X, y):
    print("参数搜索空间:")
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1e-4, 1e-3, 0.01, 0.1, 1, 'scale', 'auto'],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    print(f"  C:     {param_grid['C']}")
    print(f"  gamma: {param_grid['gamma']}")
    print(f"  kernel:{param_grid['kernel']}")
    print(f"  共 {5 * 7 * 3} 种组合")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        SVC(probability=True, class_weight='balanced'),
        param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1, return_train_score=True
    )
    grid.fit(X, y)

    print(f"\n最佳参数: {grid.best_params_}")
    print(f"最佳交叉验证准确率: {grid.best_score_ * 100:.2f}%")
    print(f"训练准确率: {grid.cv_results_['mean_train_score'][grid.best_index_] * 100:.2f}%")

    return grid.best_estimator_, grid.best_score_

def round2_ensemble(X, y, best_svm):
    print("训练 KNN + SVM 集成模型...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance')

    ensemble = VotingClassifier(
        estimators=[('svm', best_svm), ('knn', knn)],
        voting='soft', weights=[2, 1]
    )

    scores = cross_val_score(ensemble, X, y, cv=cv, scoring='accuracy')
    ensemble.fit(X, y)

    print(f"KNN单独准确率: {cross_val_score(knn, X, y, cv=cv).mean() * 100:.2f}%")
    print(f"集成模型5折交叉验证: {[f'{s*100:.1f}%' for s in scores]}")
    print(f"集成模型平均准确率: {scores.mean() * 100:.2f}%")

    return ensemble, scores.mean()

def round3_final_eval(model, X, y, label_encoder):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds, all_trues = [], []

    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        all_preds.extend(preds)
        all_trues.extend(y[test_idx])

    n_classes = len(label_encoder.classes_)
    if n_classes <= 10:
        print("\n分类报告:")
        print(classification_report(all_trues, all_preds, target_names=label_encoder.classes_))
    else:
        print(f"\n分类报告(共{n_classes}类，仅显示前10类):")
        print(classification_report(all_trues, all_preds, labels=list(range(10)), target_names=list(label_encoder.classes_[:10])))
        print(f"  ... 共 {n_classes} 个人")

    acc = accuracy_score(all_trues, all_preds)
    print(f"最终准确率: {acc * 100:.2f}%")
    return acc

def save_best_model(model, le, scaler, output_dir):
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, 'backend', 'models')

    model_data = {
        'classifier': model,
        'label_encoder': le,
        'scaler': scaler,
        'model_type': 'ensemble_classifier'
    }

    path = os.path.join(output_dir, 'classifier.pkl')
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n模型已保存: {path}")

if __name__ == '__main__':
    print("=" * 60)
    print("  多轮训练 - 人脸识别模型")
    print("  策略: 网格搜索 → 集成学习 → 交叉验证")
    print("=" * 60)

    data_dir = os.path.join(PROJECT_ROOT, 'dataset')
    if not os.path.exists(data_dir):
        print(f"错误: 数据集目录不存在: {data_dir}")
        print("请先运行: python train_model.py")
        exit()

    print("\n[1/4] 加载模型和提取特征...")
    extractor = FeatureExtractor()
    X, labels = prepare_dataset(data_dir, extractor)

    if len(X) == 0:
        print("错误: 未提取到人脸特征")
        exit()

    le = LabelEncoder()
    y = le.fit_transform(labels)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"\n数据统计: {len(X)} 张人脸, {len(le.classes_)} 个人")
    print(f"特征维度: {X.shape[1]}")

    print("\n" + "=" * 60)
    print("[2/4] 第1轮: 网格搜索超参数优化")
    print("=" * 60)
    best_svm, svm_score = round1_grid_search(X, y)

    print("\n" + "=" * 60)
    print("[3/4] 第2轮: SVM+KNN集成学习")
    print("=" * 60)
    ensemble, ensemble_score = round2_ensemble(X, y, best_svm)

    print("\n" + "=" * 60)
    print("[4/4] 第3轮: 最终评估")
    print("=" * 60)
    final_acc = round3_final_eval(ensemble, X, y, le)

    improved = ensemble_score > svm_score
    print(f"\n{'=' * 60}")
    print(f"  SVM最佳准确率:     {svm_score * 100:.2f}%")
    print(f"  集成模型准确率:     {ensemble_score * 100:.2f}%")
    if improved:
        print(f"  提升: +{(ensemble_score - svm_score) * 100:.2f}% {'🎉' if ensemble_score > 0.98 else ''}")
    print(f"{'=' * 60}")

    save_best_model(ensemble, le, scaler, None)
    print("\n训练完成！重启后端服务即可使用新模型")