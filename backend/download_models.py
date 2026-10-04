"""下载 InsightFace buffalo_l 模型权重。

模型文件总计约 350MB，不纳入版本控制。首次运行前执行：

    python download_models.py

若已安装 insightface，也可以直接调用其内置下载器：

    python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', root='models/insightface').prepare(ctx_id=-1, det_size=(640,640))"
"""

import os
import sys
import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'insightface')
BUFFALO_DIR = os.path.join(MODEL_DIR, 'models', 'buffalo_l')

# 只下载运行时实际使用的权重。
# face_service_insightface.py 中 allowed_modules=['detection', 'recognition']，
# 因此 1k3d68 / 2d106det / genderage 三个模型不会被加载，无需下载。
REQUIRED = {
    'det_10g.onnx': 'SCRFD 人脸检测',
    'w600k_r50.onnx': 'ArcFace R50 人脸识别',
}

OPTIONAL = {
    '1k3d68.onnx': '68点关键点（未启用）',
    '2d106det.onnx': '106点关键点（未启用）',
    'genderage.onnx': '性别年龄（未启用）',
}


def download(url, save_path):
    if os.path.exists(save_path):
        size_mb = os.path.getsize(save_path) / 1024 / 1024
        print('  [skip] %s (已存在, %.1f MB)' % (os.path.basename(save_path), size_mb))
        return True
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as resp, open(save_path, 'wb') as out:
            total = int(resp.headers.get('Content-Length', 0))
            done = 0
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    sys.stdout.write('\r  下载中 %5.1f%%' % (done * 100.0 / total))
                    sys.stdout.flush()
        if total:
            sys.stdout.write('\r')
        print('  [ok]   %s (%.1f MB)' % (os.path.basename(save_path),
                                        os.path.getsize(save_path) / 1024 / 1024))
        return True
    except Exception as exc:
        print('  [fail] %s :: %s' % (os.path.basename(save_path), str(exc)[:100]))
        return False


def main():
    os.makedirs(BUFFALO_DIR, exist_ok=True)
    print('模型目录: %s' % BUFFALO_DIR)

    base = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'
    print('\n[方案一] 从buffalo_l 官方发布包下载并解压（推荐，包含全部模型）')
    print('  下载地址: %s' % base)

    zip_path = os.path.join(MODEL_DIR, 'buffalo_l.zip')
    ok = download(base, zip_path)
    if ok and zip_path.lower().endswith('.zip'):
        import zipfile
        print('  解压中...')
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(MODEL_DIR)
            print('  解压完成')
        except Exception as exc:
            print('  解压失败: %s' % str(exc)[:100])

    print('\n[校验] 检查运行时必需的权重:')
    missing = []
    for name, desc in REQUIRED.items():
        path = os.path.join(BUFFALO_DIR, name)
        if os.path.exists(path):
            print('  [ok]   %-18s %s (%.1f MB)' % (name, desc, os.path.getsize(path) / 1024 / 1024))
        else:
            print('  [缺失] %-18s %s' % (name, desc))
            missing.append(name)

    print('\n当前目录内容:')
    if os.path.isdir(BUFFALO_DIR):
        for f in sorted(os.listdir(BUFFALO_DIR)):
            print('  - %s' % f)

    if missing:
        print('\n仍缺少: %s' % ', '.join(missing))
        print('请手动下载 buffalo_l 模型包并解压到: %s' % BUFFALO_DIR)
        return 1

    print('\n全部就绪，可以启动: python app.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())