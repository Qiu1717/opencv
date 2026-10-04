import os, cv2, numpy as np, pickle, warnings
warnings.filterwarnings('ignore')
from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(PROJECT_ROOT, 'backend', 'models', 'insightface', 'models', 'buffalo_l')

class FeatureExtractor:
    def __init__(self):
        self.net = cv2.dnn.readNetFromONNX(os.path.join(MODEL_DIR, 'w600k_r50.onnx'))

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

def eval_model(model, X, y, name):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=1)
    print("  " + name + ": " + str(round(scores.mean()*100, 2)) + "%")
    print("    folds: " + str([round(s*100,1) for s in scores]))
    return scores.mean()

if __name__ == '__main__':
    print("=" * 50)
    print("  Multi-Round Training (5 img/person)")
    print("=" * 50)

    data_dir = os.path.join(PROJECT_ROOT, 'dataset', 'augmented')

    print("\nExtracting features...")
    extractor = FeatureExtractor()
    X, y_labels = [], []
    for _, person_name in enumerate(sorted(os.listdir(data_dir))):
        person_dir = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        for img_name in sorted(os.listdir(person_dir))[:5]:
            img = load_image(os.path.join(person_dir, img_name))
            if img is None:
                continue
            if img.shape[0] > 20 and img.shape[1] > 20:
                X.append(extractor.extract_feature(img))
                y_labels.append(person_name)
    print("  Extracted " + str(len(X)) + " features")

    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    
    print("  Data: " + str(len(Xs)) + " faces, " + str(len(le.classes_)) + " persons")

    results = []
    best_m, best_s, best_n = None, 0, ""

    print("\n--- Training ---")

    rounds = [
        ("SVM RBF", SVC(probability=True), {'C': [0.1,1,10,100], 'gamma': ['scale',0.1,0.01,0.001]}),
        ("SVM Poly", SVC(probability=True), {'C': [1,10,100], 'gamma': ['scale',0.01], 'kernel': ['rbf','poly'], 'degree': [2,3]}),
        ("SVM Sigmoid", SVC(probability=True), {'C': [1,10,100], 'gamma': ['scale',0.01,0.001], 'kernel': ['rbf','sigmoid']}),
        ("KNN", KNeighborsClassifier(), {'n_neighbors': [1,3,5,7], 'weights': ['uniform','distance'], 'metric': ['cosine','euclidean']}),
        ("RandomForest", RandomForestClassifier(random_state=42), {'n_estimators': [50,100,200], 'max_depth': [5,10,None]}),
    ]

    svm_best = None
    knn_best = None
    rf_best = None

    for i, (name, est, params) in enumerate(rounds):
        print("\nRound " + str(i+1) + ": " + name)
        gs = GridSearchCV(est, params, cv=5, scoring='accuracy', n_jobs=1)
        gs.fit(Xs, y)
        m = gs.best_estimator_
        s = gs.best_score_
        print("  Best: " + str(gs.best_params_))
        print("  CV score: " + str(round(s*100, 2)) + "%")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fs = cross_val_score(m, Xs, y, cv=cv, scoring='accuracy', n_jobs=1)
        print("  Folds: " + str([round(v*100,1) for v in fs]))
        
        results.append((name, s))
        if s > best_s:
            best_s, best_m, best_n = s, m, name
        
        if name.startswith("SVM") and (svm_best is None or s > 0):
            svm_best = m
        if name == "KNN":
            knn_best = m
        if name == "RandomForest":
            rf_best = m

    if svm_best is None:
        svm_best = SVC(probability=True).fit(Xs, y)
    if knn_best is None:
        knn_best = KNeighborsClassifier(n_neighbors=3).fit(Xs, y)
    if rf_best is None:
        rf_best = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xs, y)

    print("\nRound 6: Ensemble (SVM+KNN+RF)")
    ensemble = VotingClassifier([('svm',svm_best),('knn',knn_best),('rf',rf_best)], voting='soft', weights=[2,1,1])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    es = cross_val_score(ensemble, Xs, y, cv=cv, scoring='accuracy', n_jobs=1)
    print("  Folds: " + str([round(v*100,1) for v in es]))
    em = es.mean()
    print("  Mean: " + str(round(em*100, 2)) + "%")
    ensemble.fit(Xs, y)
    results.append(("Ensemble", em))
    if em > best_s:
        best_s, best_m, best_n = em, ensemble, "Ensemble"

    print("\n" + "=" * 50)
    print("  Results")
    print("=" * 50)
    for name, s in results:
        b = "#" * int(s * 50)
        print("  " + name.ljust(16) + " " + str(round(s*100, 2)).rjust(7) + "% " + b)
    
    print("\n  >>> Best: " + best_n + " (" + str(round(best_s*100, 2)) + "%)")

    model_data = {'classifier': best_m, 'label_encoder': le, 'scaler': scaler, 'model_type': 'multi_round_v3'}
    path = os.path.join(PROJECT_ROOT, 'backend', 'models', 'classifier.pkl')
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)
    print("\n  Saved: " + path)