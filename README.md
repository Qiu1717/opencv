# 智能校园人脸识别门禁系统

基于 Flask + Socket.IO 的校园人脸识别门禁系统，支持人脸注册、实时识别、活体检测、权限管理、异常告警与通行统计。

本仓库仅包含源代码与必要配置，不含模型权重、数据集及课程设计文档。

---

## 项目简介与背景

传统校园门禁依赖校园卡刷卡，存在忘带、丢失、冒用等问题。人脸识别门禁利用摄像头捕捉人脸图像，与预先注册的特征向量比对完成身份核验，通行全程无需物理凭证。

本项目实现的核心逻辑是**特征比对 + 权限裁决**：识别到人脸后，先判断是否为已注册人员，再根据该人员的区域权限、时段权限与有效期决定是否放行。任一环节不通过则记录异常并实时推送告警到管理端。

识别采用 InsightFace 预训练模型 `buffalo_l`，输出 512 维人脸特征向量，通过余弦相似度完成匹配。检测环节设计了三级降级链路（InsightFace → OpenCV DNN → Haar 级联），并在失败时尝试 CLAHE 对比度增强、多尺度缩放与镜像翻转，提升复杂光照与姿态下的检出率。

---

## 核心功能

| 模块 | 说明 |
|------|------|
| 人脸注册 | 上传照片或调用摄像头采集，自动检测人脸、提取特征向量并落库；同一人可多次注册以提升准确率 |
| 实时识别 | 浏览器采集视频帧并通过 WebSocket 推送至后端，识别结果实时回传并在页面标注人脸框与姓名 |
| 活体检测 | 结合拉普拉斯方差模糊检测、频域纹理分析与光流运动检测，识别照片翻拍与视频攻击 |
| 权限管理 | 支持按通行区域、时段区间、有效日期范围组合授权；无权限人员识别后记录拒绝并告警 |
| 异常告警 | 未注册人员、权限不足等事件写入异常记录表，通过 Socket.IO 实时推送至管理端 |
| 通行统计 | 按日期范围统计通行总数、成功率、异常数，可导出 CSV |
| 人员管理 | 人员增删改查、批量导入 CSV 名单 |
| 尾随检测 | 基于目标跟踪与轨迹判断的尾随行为检测模块 |

---

## 技术栈

### 后端

| 技术 | 版本 | 用途 |
|------|------|------|
| Flask | 2.3.3 | Web 框架 |
| Flask-SQLAlchemy | 3.0.5 | ORM |
| Flask-SocketIO | 5.3.6 | WebSocket 实时通信 |
| Flask-CORS | 4.0.0 | 跨域支持 |
| OpenCV-Python | 4.5.5.64 | 图像处理 |
| InsightFace | — | 人脸检测与识别（`buffalo_l`） |
| SQLite | — | 数据存储 |
| scikit-learn | — | 特征分类器训练（SVM） |

### 前端

原生 HTML5 + CSS3 + JavaScript（ES6+），无需构建步骤。Socket.IO 客户端与图标库通过 CDN 引入。

### 模型

| 模型 | 用途 |
|------|------|
| `det_10g.onnx` | SCRFD 人脸检测 |
| `w600k_r50.onnx` | ArcFace R50 人脸识别，输出 512 维特征 |
| `res10_300x300_ssd_iter_140000_fp16.caffemodel` | OpenCV DNN 检测（降级用） |

### 运行环境

- Python 3.8+（开发环境为 3.7/ 3.12）
- Windows / Linux / macOS
- 内存 ≥ 4GB
- 浏览器需支持 WebRTC（Chrome、Edge、Firefox）

---

## 目录结构

```
qzz/
├── backend/                    # 后端服务
│   ├── app.py                  # Flask 主应用与全部 REST/WebSocket 接口
│   ├── models.py               # SQLAlchemy 数据模型（6 张表）
│   ├── config.py               # 配置常量与日志初始化
│   ├── utils.py                # 图片解码工具（multipart / base64 双通道）
│   ├── face_service_insightface.py  # 人脸检测与识别服务
│   ├── liveness_service.py     # 活体检测服务
│   ├── download_models.py      # 模型权重下载脚本
│   ├── requirements.txt        # 后端依赖
│   └── models/                 # 模型权重目录（不入库，需自行下载）
│       ├── deploy.prototxt.txt
│       ├── res10_300x300_ssd_iter_140000_fp16.caffemodel
│       └── insightface/        # buffalo_l 权重，需运行下载脚本获取
│
├── frontend/                   # 前端静态资源
│   ├── index.html              # 主页面
│   ├── login.html              # 登录页
│   ├── styles.css              # 样式
│   └── app.js                  # 前端逻辑
│
├── tools/                      # 数据准备与训练脚本
│   ├── augment_data.py         # 人脸数据增强（9 种变换）
│   ├── prepare_data.py         # 训练数据预处理
│   ├── train_final.py          # 特征提取 + SVM 训练（最终方案）
│   └── download_datasets.py    # 数据集下载工具
│
├── desktop/                    # PyQt5 桌面版（早期方案，保留供参考）
│   ├── run.py                  # 桌面版入口：python run.py
│   ├── gui_main.py
│   ├── config.py
│   ├── database.py
│   ├── face_detection.py
│   ├── face_recognition.py
│   ├── liveness_detection.py
│   └── tailgating_detection.py
│
├── requirements.txt            # 桌面版依赖
├── start.bat                   # Windows 一键启动
├── .env.example                # 环境变量示例
└── .gitignore
```

---

## 安装与启动

### 方式一：一键启动（Windows）

```bat
双击 start.bat
```

脚本会自动创建虚拟环境、安装依赖、下载模型并启动服务，随后打开浏览器。

### 方式二：手动启动

**1. 安装依赖**

```bash
cd backend
pip install -r requirements.txt
```

**2. 下载模型权重**

```bash
python download_models.py
```

约下载 350MB，解压至 `backend/models/insightface/models/buffalo_l/`。

**3. 启动服务**

```bash
cd backend
python app.py
```

**4. 访问**

浏览器打开 <http://localhost:5000>

> 首次启动会自动建表并创建管理员账号 `admin`。密码取自环境变量 `ADMIN_PASSWORD`；
> 若未设置则自动生成随机密码，请查看启动日志。

---

## 配置说明

### 环境变量

复制 `.env.example` 为 `.env` 后按需配置：

| 变量 | 说明 | 默认行为 |
|------|------|----------|
| `SECRET_KEY` | Flask 会话签名密钥 | 未设置时每次启动自动生成随机值（重启后需重新登录） |
| `ADMIN_PASSWORD` | 初始管理员密码 | 未设置时自动生成随机密码，见启动日志 |

### 关键配置项

位于 `backend/config.py`：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `SIMILARITY_THRESHOLD` | 0.55 | 人脸匹配相似度阈值，调高更严格 |
| `CONFIDENCE_THRESHOLD` | 0.10 | 人脸检测置信度下限 |
| `MIN_RECORD_INTERVAL` | 5 | 同一人重复记录通行日志的最小间隔（秒），用于抑制实时视频流刷屏 |
| `MAX_UPLOAD_SIZE` | 50MB | 单次上传体积上限 |

### 数据库

使用 SQLite，表结构由 SQLAlchemy 模型定义，首次启动自动创建：

`persons`（人员）、`access_logs`（通行记录）、`abnormal_records`（异常记录）、`permissions`（权限）、`cameras`（摄像头）、`users`（账号）

数据库文件为运行时生成，不纳入版本管理。

### 摄像头配置

预置 9 个虚拟摄像头点位（东门、南门、北门、图书馆、宿舍楼等），在 `backend/models.py` 的 `_seed_default_cameras()` 中调整。

---

## 常见问题

**Q：启动报`InsightFace init fail`，退回到 DNN/Haar 检测？**

模型未下载或路径不对。执行 `python backend/download_models.py` 校验，确认 `backend/models/insightface/models/buffalo_l/` 下存在 `det_10g.onnx` 与 `w600k_r50.onnx`。注意需在 `backend` 目录下执行，因模型路径为相对路径。

**Q：上传照片提示「未检测到人脸」？**

系统已内置 CLAHE 增强、多尺度与镜像重试。仍失败通常是因为人脸占比过小、光线过暗或严重侧脸。建议正对摄像头、光线均匀、人脸占画面 1/4 以上。

**Q：识别准确率不高，识别成其他人？**

先调高 `SIMILARITY_THRESHOLD`（如 0.65）观察效果；再为同一人多次注册（不同角度、表情），系统会保留该人的全部特征并取最高相似度。这是最有效的提升手段。

**Q：默认管理员密码是多少？**

未设置 `ADMIN_PASSWORD` 时启动会生成随机密码并打印在日志中。请在**首次启动时**查看，或删除 `backend/face_system.db` 重新初始化。

**Q：摄像头无法打开？**

浏览器要求安全上下文。`localhost` 可直接访问；若通过局域网IP 访问，需配置 HTTPS，否则浏览器会禁用摄像头权限。

**Q：WebSocket 连接失败？**

确认 `flask-socketio`、`python-socketio`、`python-engineio` 版本匹配（见 `backend/requirements.txt`），并检查 5000 端口未被占用。

**Q：如何重新训练分类器？**

数据集准备好后执行 `python tools/train_final.py`。该脚本提取特征后进行 SVM 参数网格搜索，将最优模型保存至 `backend/models/classifier.pkl`。注意在线匹配使用的是余弦相似度，该分类器为离线实验产物。

---

## 桌面版（早期方案）

`desktop/` 是早期的 PyQt5 桌面实现，与 Web 版共用部分算法模块但已不作为主线维护，保留供对照参考。

```bash
pip install -r requirements.txt
python desktop/run.py
```

---

## 说明

- 本仓库不含模型权重、数据集与测试照片，需按上述步骤自行获取
- 数据库文件在运行时生成，包含人脸图像与特征向量，不应提交到版本库
- 识别准确率受光照、角度与设备影响，生产部署建议配合活体检测与人工复核通道