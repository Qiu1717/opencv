import os, cv2, numpy as np, sys, urllib.request, zipfile, tarfile, shutil, ssl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ssl._create_default_https_context = ssl._create_unverified_context

DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')

def download_file(url, save_path, desc):
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')
        urllib.request.urlretrieve(url, save_path)
        size_kb = os.path.getsize(save_path) // 1024
        print("  Done: " + str(size_kb) + " KB")
        return True
    except Exception as e:
        print("  Failed: " + str(e)[:80])
        return False

def extract_zip(zip_path, extract_dir):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        print("  Extracted OK")
        return True
    except Exception as e:
        print("  Extract err: " + str(e)[:80])
        return False

def extract_tar(tar_path, extract_dir):
    try:
        with tarfile.open(tar_path, 'r:*') as tf:
            tf.extractall(extract_dir)
        print("  Extracted OK")
        return True
    except Exception as e:
        print("  Extract err: " + str(e)[:80])
        return False

def organize_images(src_dir, dst_dir, prefix):
    person_map = {}
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pgm', '.bmp')):
                parts = os.path.splitext(f)[0].split('_')
                subj = None
                for p in parts:
                    if p.isdigit():
                        subj = p
                        break
                if subj is None:
                    subj = str(abs(hash(f)) % 1000)
                person_map.setdefault(subj, []).append(os.path.join(root, f))

    os.makedirs(dst_dir, exist_ok=True)
    total = 0
    for pid, paths in person_map.items():
        person_dir = os.path.join(dst_dir, prefix + '_' + pid)
        os.makedirs(person_dir, exist_ok=True)
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                dst = os.path.join(person_dir, str(total) + '.jpg')
                cv2.imwrite(dst, img)
                total += 1
    return len(person_map), total

def download_yale(dst):
    print("\n" + "=" * 50)
    print("  Yale Face Database")
    print("=" * 50)
    
    dest = os.path.join(dst, 'yale_faces')
    if os.path.exists(dest) and len(os.listdir(dest)) > 5:
        n = len(os.listdir(dest))
        t = sum(len(os.listdir(os.path.join(dest, d))) for d in os.listdir(dest))
        print("  Already exists: " + str(n) + " persons, " + str(t) + " imgs")
        return True

    yale_urls = [
        "http://cvc.cs.yale.edu/cvc/projects/yalefaces/yalefaces.zip",
        "https://github.com/welluz/face-recognition/raw/master/dataset/yalefaces.zip",
        "https://vasc.ri.cmu.edu/idb/images/face/frontal_images/images.tar",
    ]

    for url in yale_urls:
        fpath = os.path.join(dst, 'yale_temp.zip')
        if download_file(url, fpath, "Yale Faces"):
            extract_zip(fpath, os.path.join(dst, 'yale_temp'))
            os.remove(fpath)
            break
    else:
        print("  All Yale URLs failed")
        return False

    temp = os.path.join(dst, 'yale_temp')
    n, t = 0, 0
    if os.path.exists(temp):
        n, t = organize_images(temp, dest, 'yale')
        shutil.rmtree(temp, ignore_errors=True)
    
    print("  Result: " + str(n) + " persons, " + str(t) + " images")
    return n > 0

def download_umist(dst):
    print("\n" + "=" * 50)
    print("  UMIST (Sheffield) Face Database")
    print("=" * 50)

    dest = os.path.join(dst, 'umist_faces')
    if os.path.exists(dest) and len(os.listdir(dest)) > 5:
        n = len(os.listdir(dest))
        t = sum(len(os.listdir(os.path.join(dest, d))) for d in os.listdir(dest))
        print("  Already exists: " + str(n) + " persons, " + str(t) + " imgs")
        return True

    umist_urls = [
        "https://www.sheffield.ac.uk/polopoly_fs/1.88665!/file/umist_cropped.tar.gz",
        "https://github.com/rowdypatel/Face-Recognition-Using-PCA/raw/master/umist_cropped.zip",
    ]

    for url in umist_urls:
        fpath = os.path.join(dst, 'umist_temp.tar.gz')
        if download_file(url, fpath, "UMIST Faces"):
            try:
                extract_tar(fpath, os.path.join(dst, 'umist_temp'))
            except:
                extract_zip(fpath, os.path.join(dst, 'umist_temp'))
            os.remove(fpath)
            break
    else:
        print("  All UMIST URLs failed")
        return False

    temp = os.path.join(dst, 'umist_temp')
    n, t = 0, 0
    if os.path.exists(temp):
        n, t = organize_images(temp, dest, 'umist')
        shutil.rmtree(temp, ignore_errors=True)
    
    print("  Result: " + str(n) + " persons, " + str(t) + " images")
    return n > 0

def download_georgia_tech(dst):
    print("\n" + "=" * 50)
    print("  Georgia Tech Face Database")
    print("=" * 50)

    dest = os.path.join(dst, 'gt_faces')
    if os.path.exists(dest) and len(os.listdir(dst)) > 5:
        n = len(os.listdir(dst))
        t = sum(len(os.listdir(os.path.join(dest, d))) for d in os.listdir(dst))
        print("  Already exists: " + str(n) + " persons, " + str(t) + " imgs")
        return True

    gt_urls = [
        "https://s3.amazonaws.com/com.neo.face.recognition/gt_db.zip",
        "http://www.anefian.com/research/face_reco.zip",
    ]

    for url in gt_urls:
        fpath = os.path.join(dst, 'gt_temp.zip')
        if download_file(url, fpath, "Georgia Tech Faces"):
            extract_zip(fpath, os.path.join(dst, 'gt_temp'))
            os.remove(fpath)
            break
    else:
        print("  All GT URLs failed")
        return False

    temp = os.path.join(dst, 'gt_temp')
    n, t = 0, 0
    if os.path.exists(temp):
        n, t = organize_images(temp, dest, 'gt')
        shutil.rmtree(temp, ignore_errors=True)

    print("  Result: " + str(n) + " persons, " + str(t) + " images")
    return n > 0

def download_jaffe(dst):
    print("\n" + "=" * 50)
    print("  JAFFE Face Database")
    print("=" * 50)

    dest = os.path.join(dst, 'jaffe_faces')
    if os.path.exists(dest) and len(os.listdir(dst)) > 5:
        n = len(os.listdir(dst))
        t = sum(len(os.listdir(os.path.join(dest, d))) for d in os.listdir(dst))
        print("  Already exists: " + str(n) + " persons, " + str(t) + " imgs")
        return True

    jaffe_urls = [
        "https://zenodo.org/record/3451524/files/jaffedbase.zip",
        "http://www.kasrl.org/jaffe_download.html",
    ]

    for url in jaffe_urls:
        fpath = os.path.join(dst, 'jaffe_temp.zip')
        if download_file(url, fpath, "JAFFE Faces"):
            extract_zip(fpath, os.path.join(dst, 'jaffe_temp'))
            os.remove(fpath)
            break
    else:
        print("  All JAFFE URLs failed")
        return False

    temp = os.path.join(dst, 'jaffe_temp')
    n, t = 0, 0
    if os.path.exists(temp):
        n, t = organize_images(temp, dest, 'jaffe')
        shutil.rmtree(temp, ignore_errors=True)

    print("  Result: " + str(n) + " persons, " + str(t) + " images")
    return n > 0

if __name__ == '__main__':
    print("=" * 50)
    print("  Downloading Face Datasets v2")
    print("=" * 50)
    free_gb = shutil.disk_usage('d:').free // 1024**3
    print("  Free space: " + str(free_gb) + " GB")

    results = []
    results.append(("Yale Faces", download_yale(DATASET_DIR)))
    results.append(("UMIST Faces", download_umist(DATASET_DIR)))
    results.append(("Georgia Tech", download_georgia_tech(DATASET_DIR)))
    results.append(("JAFFE", download_jaffe(DATASET_DIR)))

    print("\n" + "=" * 50)
    print("  Summary")
    print("=" * 50)
    for name, ok in results:
        status = "OK" if ok else "FAILED"
        print("  " + name + ": " + status)

    total_persons = 0
    total_images = 0
    for d in os.listdir(DATASET_DIR):
        dp = os.path.join(DATASET_DIR, d)
        if os.path.isdir(dp) and d not in ['augmented', 'att_faces']:
            for person in os.listdir(dp):
                person_dir = os.path.join(dp, person)
                if os.path.isdir(person_dir):
                    n_imgs = len(os.listdir(person_dir))
                    total_images += n_imgs
                    total_persons += 1

    print("\n  Total new: " + str(total_persons) + " persons, " + str(total_images) + " images")
    print("  Free space: " + str(shutil.disk_usage('d:').free // 1024**3) + " GB")