import os
import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def augment_and_save(img, base_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    if len(img.shape) == 3 and img.shape[2] == 3:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    elif len(img.shape) == 2:
        img_pil = Image.fromarray(img)
    else:
        img_pil = Image.fromarray(img)

    save_path = os.path.join(output_dir, f'{base_name}_original.jpg')
    try:
        img_pil.save(save_path)
    except Exception as e:
        print(f"  Failed to save: {save_path} - {e}")
        return 0

    transformations = [
        ('flip', cv2.flip(img, 1)),
        ('rot5', cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 5, 1.0), (img.shape[1], img.shape[0]))),
        ('rot_m5', cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), -5, 1.0), (img.shape[1], img.shape[0]))),
        ('bright_up', cv2.convertScaleAbs(img, alpha=1, beta=30)),
        ('bright_down', cv2.convertScaleAbs(img, alpha=1, beta=-30)),
        ('blur', cv2.GaussianBlur(img, (3, 3), 0)),
        ('noise', np.clip(img.astype(np.int16) + np.random.randint(-15, 15, img.shape, dtype=np.int16), 0, 255).astype(np.uint8)),
        ('contrast', cv2.convertScaleAbs(img, alpha=1.2, beta=0)),
        ('zoom_crop', img[5:-5, 5:-5] if img.shape[0] > 20 and img.shape[1] > 20 else img),
    ]

    saved = 1
    for suffix, t_img in transformations:
        if t_img.shape[0] > 10 and t_img.shape[1] > 10:
            t_img = cv2.resize(t_img, (img.shape[1], img.shape[0]))
            
            if len(t_img.shape) == 3 and t_img.shape[2] == 3:
                t_pil = Image.fromarray(cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB))
            elif len(t_img.shape) == 2:
                t_pil = Image.fromarray(t_img)
            else:
                t_pil = Image.fromarray(t_img)
                
            save_path = os.path.join(output_dir, f'{base_name}_{suffix}.jpg')
            try:
                t_pil.save(save_path)
                saved += 1
            except Exception as e:
                print(f"  Failed to save: {save_path} - {e}")

    return saved

def read_pgm_file(filepath):
    with open(filepath, 'rb') as f:
        header = f.readline()
        if header.startswith(b'P5'):
            while True:
                line = f.readline()
                if not line.startswith(b'#'):
                    break
            dims = line.decode().strip().split()
            width, height = int(dims[0]), int(dims[1])
            max_val = int(f.readline().decode().strip())
            data = np.fromfile(f, dtype=np.uint8, count=width * height)
            img = data.reshape((height, width))
            return img
        else:
            return None

def main():
    src_dir = os.path.join(PROJECT_ROOT, 'dataset', 'att_faces')
    dst_dir = os.path.join(PROJECT_ROOT, 'dataset', 'augmented')

    if not os.path.exists(src_dir):
        print(f"Error: {src_dir} not found!")
        return

    total_orig = 0
    total_aug = 0

    for person_dir in sorted(os.listdir(src_dir)):
        full_src = os.path.join(src_dir, person_dir)
        if not os.path.isdir(full_src):
            continue

        base_name = person_dir.replace('s', 'person_')
        full_dst = os.path.join(dst_dir, base_name)
        os.makedirs(full_dst, exist_ok=True)

        count = 0
        for img_name in os.listdir(full_src):
            img_path = os.path.join(full_src, img_name)
            
            img = None
            if img_name.lower().endswith('.pgm'):
                img = read_pgm_file(img_path)
            else:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                img = cv2.imread(img_path)
            
            if img is None:
                print(f"  Warning: Cannot read {img_path}")
                continue
            
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            n = augment_and_save(img, img_name.replace('.pgm', '').replace('.jpg', '').replace('.png', ''), full_dst)
            count += n

        print(f"  {person_dir} -> {base_name}: {count} images")
        total_orig += len([f for f in os.listdir(full_src) if f.lower().endswith(('.pgm', '.jpg', '.png'))])
        total_aug += count

    print(f"\nDone! {total_orig} original -> {total_aug} augmented images")
    print(f"Saved to: {dst_dir}")
    
    test_dir = os.path.join(dst_dir, 'person_1')
    if os.path.exists(test_dir):
        files = os.listdir(test_dir)
        print(f"\nTest - person_1 has {len(files)} files")
        if files:
            print(f"First 3 files: {files[:3]}")

if __name__ == '__main__':
    main()