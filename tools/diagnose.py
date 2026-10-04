import base64, os, cv2, numpy as np, traceback
from PIL import Image

MODEL_DIR = 'backend/models/insightface'

def test(image_path):
    print("=" * 60)
    print("TEST: " + image_path)
    print("=" * 60)

    with open(image_path, 'rb') as f:
        raw = f.read()

    b64 = base64.b64encode(raw).decode('ascii')
    data_url = "data:image/jpeg;base64," + b64

    frame_data = data_url.split(',')[1]
    image_data = base64.b64decode(frame_data)
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        print("ERROR: cv2.imdecode FAILED!")
        return
    print("Decoded: " + str(frame.shape))

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', root=MODEL_DIR,
                       allowed_modules=['detection'])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    faces = app.get(frame)
    print("InsightFace (640): " + str(len(faces)) + " faces")
    for f in faces:
        print("  score=" + str(round(f.det_score, 4)) + " box=" + str(f.bbox.astype(int)))

    if len(faces) == 0:
        app.prepare(ctx_id=-1, det_size=(1280, 1280))
        faces = app.get(frame)
        print("InsightFace (1280): " + str(len(faces)) + " faces")
        for f in faces:
            print("  score=" + str(round(f.det_score, 4)) + " box=" + str(f.bbox.astype(int)))
        app.prepare(ctx_id=-1, det_size=(640, 640))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    haar = cascade.detectMultiScale(gray, 1.05, 2, minSize=(20, 20))
    print("Haar: " + str(len(haar)) + " faces")
    for (x, y, fw, fh) in haar:
        print("  box=(" + str(x) + "," + str(y) + "," + str(x+fw) + "," + str(y+fh) + ")")

    print()

if __name__ == '__main__':
    test_dir = 'testface'
    for fn in sorted(os.listdir(test_dir)):
        if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
            test(os.path.join(test_dir, fn))