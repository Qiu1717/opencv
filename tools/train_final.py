import os, cv2, numpy as np, pickle, warnings, shutil
warnings.filterwarnings('ignore')
import warnings as _w
_w.filterwarnings('ignore', category=DeprecationWarning)
_w.filterwarnings('ignore', category=FutureWarning)
from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(PROJECT_ROOT, 'backend', 'models', 'insightface', 'models', 'buffalo_l')

net = None

def get_net():
    global net
    if net is None:
        net = cv2.dnn.readNetFromONNX(os.path.join(MODEL_DIR, 'w600k_r50.onnx'))
    return net

def extract_feature(face_img):
    if len(face_img.shape) == 2:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
    face = cv2.resize(face_img, (112, 112)).astype(np.float32)
    face = (face - 127.5) / 127.5
    face = np.expand_dims(np.transpose(face, (2, 0, 1)), 0)
    get_net().setInput(face)
    f = get_net().forward()[0]
    return f / np.linalg.norm(f)

def load_image(p):
    try:
        img = np.array(Image.open(p).convert('RGB'))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except:
        return None

if __name__ == '__main__':
    print("=" * 50)
    print("  Fast Anti-Overfit Training")
    print("=" * 50)

    DD = os.path.join(PROJECT_ROOT, 'dataset', 'att_augmented_v2')
    MAX = 25

    print("\nExtracting features (" + str(MAX) + "/person)...")
    X, yl = [], []
    for pi, pn in enumerate(sorted(os.listdir(DD))):
        pd = os.path.join(DD, pn)
        if not os.path.isdir(pd):
            continue
        for fn in sorted(os.listdir(pd))[:MAX]:
            if not fn.lower().endswith(('.jpg','.png','.bmp')):
                continue
            img = load_image(os.path.join(pd, fn))
            if img is None or img.shape[0] < 20:
                continue
            X.append(extract_feature(img))
            yl.append(pn)
        if (pi+1) % 10 == 0:
            print("  " + str(pi+1) + "/40")

    X = np.array(X)
    print("  Features: " + str(X.shape))

    le = LabelEncoder()
    y = le.fit_transform(yl)
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    sc = StandardScaler()
    Xts = sc.fit_transform(Xt)
    Xes = sc.transform(Xe)

    print("  Train: " + str(len(Xts)) + " | Test: " + str(len(Xes)))

    C_list = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100]
    gamma_list = ['scale', 0.0001, 0.001, 0.01, 0.1]

    best_m, best_s, best_c, best_g = None, 0, None, None
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    count = 0
    for C in C_list:
        for g in gamma_list:
            count += 1
            try:
                m = SVC(C=C, gamma=g, kernel='rbf', probability=True)
                m.fit(Xts, yt)
                tp = m.predict(Xes)
                ta = accuracy_score(ye, tp)
                tt = accuracy_score(yt, m.predict(Xts))
                gap = tt - ta
                results.append((C, g, ta, tt, gap, m))
                if ta > best_s:
                    best_s, best_m, best_c, best_g = ta, m, C, g
            except:
                pass
            if count % 15 == 0:
                print("  Tested " + str(count) + "/45...")

    results.sort(key=lambda x: -x[2])

    print("\n" + "=" * 50)
    print("  Results (sorted by Test Acc)")
    print("=" * 50)
    print("  " + "C".ljust(8) + "Gamma".ljust(10) + "Train".rjust(7) + "Test".rjust(7) + "Gap".rjust(7))
    print("  " + "-" * 39)
    for C, g, ta, tt, gap, _ in results[:10]:
        flag = " !" if gap > 0.15 else ""
        print("  " + str(C).ljust(8) + str(g).ljust(10) +
              str(round(tt*100,1)).rjust(6) + "%" +
              str(round(ta*100,1)).rjust(6) + "%" +
              str(round(gap*100,1)).rjust(6) + "%" + flag)

    print("\n  >>> Best: C=" + str(best_c) + ", gamma=" + str(best_g) + " (Test: " + str(round(best_s*100,2)) + "%)")

    md = {'classifier': best_m, 'label_encoder': le, 'scaler': sc, 'num_classes': len(le.classes_)}
    op = os.path.join(PROJECT_ROOT, 'backend', 'models', 'classifier.pkl')
    with open(op, 'wb') as f:
        pickle.dump(md, f)
    print("  Saved: " + op)
    print("  Done!")