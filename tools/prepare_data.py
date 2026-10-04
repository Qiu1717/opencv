import os, cv2, numpy as np, shutil, warnings
warnings.filterwarnings('ignore')
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')

def read_pgm(filepath):
    with open(filepath, 'rb') as f:
        header = f.readline()
        if header.startswith(b'P5'):
            while True:
                line = f.readline()
                if not line.startswith(b'#'):
                    break
            dims = line.decode().strip().split()
            w, h = int(dims[0]), int(dims[1])
            f.readline()
            data = np.fromfile(f, dtype=np.uint8, count=w*h)
            return data.reshape((h, w))
    return None

def augment_image(img, num_aug=10):
    results = []
    if img is None:
        return results
    
    h, w = img.shape[:2]
    if h < 30 or w < 30:
        return [img]

    results.append(img)

    for i in range(num_aug):
        aug = img.copy()
        h_a, w_a = aug.shape[:2]
        r = np.random.random()

        if r < 0.15:
            aug = cv2.flip(aug, 1)
        elif r < 0.25:
            angle = np.random.uniform(-8, 8)
            M = cv2.getRotationMatrix2D((w_a//2, h_a//2), angle, 1.0)
            aug = cv2.warpAffine(aug, M, (w_a, h_a))
        elif r < 0.35:
            beta = np.random.randint(-25, 25)
            aug = cv2.convertScaleAbs(aug, alpha=1, beta=beta)
        elif r < 0.45:
            alpha = np.random.uniform(0.8, 1.2)
            aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=0)
        elif r < 0.55:
            aug = cv2.GaussianBlur(aug, (3, 3), 0)
        elif r < 0.65:
            noise = np.random.randint(-12, 12, aug.shape, dtype=np.int16)
            aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif r < 0.75:
            shift_x = np.random.randint(-5, 5)
            shift_y = np.random.randint(-5, 5)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            aug = cv2.warpAffine(aug, M, (w_a, h_a))
        elif r < 0.85:
            crop_m = np.random.randint(2, 6)
            if h_a > 2*crop_m and w_a > 2*crop_m:
                cropped = aug[crop_m:-crop_m, crop_m:-crop_m]
                aug = cv2.resize(cropped, (w_a, h_a))
        elif r < 0.92:
            if len(aug.shape) == 3:
                for c in range(3):
                    aug[:,:,c] = cv2.equalizeHist(aug[:,:,c])

        if aug.shape[0] >= 20 and aug.shape[1] >= 20:
            results.append(aug)

    return results

def save_image(img, filepath):
    if len(img.shape) == 3:
        if img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
    else:
        img_rgb = img
    try:
        Image.fromarray(img_rgb).save(filepath, 'JPEG', quality=95)
    except:
        pass

def augment_dataset(src_dir, dst_dir, num_aug=10):
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)
    total = 0

    for person_dir in sorted(os.listdir(src_dir)):
        full_src = os.path.join(src_dir, person_dir)
        if not os.path.isdir(full_src):
            continue

        person_name = 'att_' + person_dir.replace('s', '')
        full_dst = os.path.join(dst_dir, person_name)
        os.makedirs(full_dst, exist_ok=True)

        img_files = [f for f in sorted(os.listdir(full_src)) 
                     if f.lower().endswith(('.pgm', '.jpg', '.png', '.bmp'))]

        count = 0
        for img_name in img_files:
            img_path = os.path.join(full_src, img_name)
            img = None
            
            if img_name.lower().endswith('.pgm'):
                img_gray = read_pgm(img_path)
                if img_gray is not None:
                    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            else:
                img = cv2.imread(img_path)
                if img is None:
                    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img_gray is not None:
                        img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

            if img is None:
                continue

            augs = augment_image(img, num_aug=num_aug)
            for j, aug in enumerate(augs):
                save_image(aug, os.path.join(full_dst, f'img_{count:04d}.jpg'))
                count += 1

        print(f"  {person_name}: {count} images")
        total += count

    print(f"\n  Total: {total} images across {len(os.listdir(dst_dir))} persons")
    return total

if __name__ == '__main__':
    print("=" * 50)
    print("  Data Preparation (PIL save)")
    print("=" * 50)

    src = os.path.join(DATASET_DIR, 'att_faces')
    dst = os.path.join(DATASET_DIR, 'att_augmented_v2')

    print(f"\nAugmenting AT&T: {src} -> {dst}")
    n = augment_dataset(src, dst, num_aug=10)

    print(f"\nChecking result...")
    for person_dir in sorted(os.listdir(dst))[:3]:
        pd = os.path.join(dst, person_dir)
        imgs = os.listdir(pd)
        print(f"  {person_dir}: {len(imgs)} images, first: {imgs[:3]}")

    free_gb = shutil.disk_usage('d:').free // 1024**3
    print(f"\nFree space: {free_gb} GB")
    print("Done!")