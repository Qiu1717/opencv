const API_BASE = window.location.origin + '/api';
let socket;
let selectedPersonId = null;
let capturedImage = null;
let recognitionStream = null;
let recognitionEnabled = false;
let frameCount = 0;
let lastFpsUpdate = Date.now();
let lastFrameSent = 0;
let recognitionCameraId = '';
let pendingRegisterPerson = null;

const state = {
    currentTab: 'register',
    persons: [],
    accessLogs: [],
    abnormalRecords: []
};

(async function checkAuth() {
    try {
        const res = await fetch(`${API_BASE}/auth/session`);
        const data = await res.json();
        if (!data.authenticated) {
            window.location.href = '/login.html';
            return;
        }
        var uname = data.data.username || 'admin';
        var urole = data.data.role === 'admin' ? '超级管理员' : '用户';
        var uinit = (uname || 'A')[0].toUpperCase();
        document.getElementById('sidebarUsername').textContent = uname;
        document.getElementById('sidebarRole').textContent = urole;
        document.getElementById('userAvatar').textContent = uinit;
        document.getElementById('logoutBtn').addEventListener('click', async () => {
            await fetch(`${API_BASE}/auth/logout`, { method: 'POST' });
            window.location.href = '/login.html';
        });
        initApp();
    } catch {
        window.location.href = '/login.html';
    }
})();

async function apiRequest(url, options = {}) {
    const response = await fetch(`${API_BASE}${url}`, {
        headers: { 'Content-Type': 'application/json', ...options.headers },
        ...options
    });
    if (response.status === 401) {
        window.location.href = '/login.html';
        throw new Error('未登录');
    }
    return response.json();
}

function initSocket() {
    socket = io(window.location.origin);
    
    socket.on('connect', () => {
        document.getElementById('connectionStatus').className = 'conn-status online';
        document.getElementById('connectionStatus').textContent = '已连接';
        showToast('已连接到服务器', 'success');
    });
    
    socket.on('disconnect', () => {
        document.getElementById('connectionStatus').className = 'conn-status offline';
        document.getElementById('connectionStatus').textContent = '未连接';
        showToast('连接已断开', 'error');
    });
    
    socket.on('recognition_result', (result) => {
        handleRecognitionResult(result);
    });
    
    socket.on('access_log', (log) => {
        addAccessLog(log);
        updateStats();
    });
    
    socket.on('abnormal_event', (event) => {
        addAbnormalRecord(event);
        updateStats();
    });
}

const PAGE_TITLES = {
    register: '人脸注册',
    recognition: '实时识别',
    alarm: '异常告警',
    report: '统计报表',
    permission: '权限管理'
};

function initTabs() {
    const navItems = document.querySelectorAll('.sidebar .nav-item[data-tab]');
    const pageSections = document.querySelectorAll('.page-section');
    const pageTitle = document.getElementById('pageTitle');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const tabId = item.dataset.tab;

            navItems.forEach(n => n.classList.remove('active'));
            pageSections.forEach(s => s.classList.remove('active'));

            item.classList.add('active');
            const target = document.getElementById('page-' + tabId);
            if (target) target.classList.add('active');

            pageTitle.textContent = PAGE_TITLES[tabId] || tabId;
            state.currentTab = tabId;

            if (tabId === 'report') {
                setDefaultDates();
            }
            if (tabId === 'alarm') {
                loadAbnormalRecords();
            }
        });
    });
}

function setDefaultDates() {
    const today = new Date();
    const todayStr = today.toISOString().split('T')[0];
    const timeStr = String(today.getHours()).padStart(2, '0') + ':' + String(today.getMinutes()).padStart(2, '0');
    
    document.getElementById('reportStartDate').valueAsDate = today;
    document.getElementById('reportEndDate').valueAsDate = today;
    document.getElementById('validFrom').value = todayStr;
    document.getElementById('validTo').value = todayStr;
}

function updateTime() {
    const now = new Date();
    document.getElementById('currentTime').textContent = now.toLocaleString('zh-CN');
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast show ${type}`;
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

function compressImage(img, maxWidth, maxHeight, quality) {
    let width = img.width;
    let height = img.height;
    
    if (width > maxWidth || height > maxHeight) {
        if (width > height) {
            height = (height * maxWidth) / width;
            width = maxWidth;
        } else {
            width = (width * maxHeight) / height;
            height = maxHeight;
        }
    }
    
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, width, height);
    
    return canvas.toDataURL('image/jpeg', quality);
}

function initRegister() {
    const video = document.getElementById('registerVideo');
    const canvas = document.getElementById('registerCanvas');
    const captureBtn = document.getElementById('captureBtn');
    const uploadBtn = document.getElementById('uploadBtn');
    const fileInput = document.getElementById('fileInput');
    const registerFaceBtn = document.getElementById('registerFaceBtn');
    const registerForm = document.getElementById('registerForm');
    
    let cameraStream = null;
    
    registerFaceBtn.textContent = '请先拍照或上传照片';
    
    captureBtn.addEventListener('click', () => {
        if (cameraStream) {
            const ctx = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);
            const dataUrl = canvas.toDataURL('image/jpeg');
            document.getElementById('registerImage').src = dataUrl;
            document.getElementById('registerImage').style.display = 'block';
            video.style.display = 'none';
            cameraStream.getTracks().forEach(t => t.stop());
            cameraStream = null;
            capturedImage = dataUrl;
            registerFaceBtn.disabled = false;
            registerFaceBtn.textContent = '注册人脸';
            captureBtn.textContent = '重新拍照';
            showToast('已捕获图像，请点击注册人脸', 'success');
        } else {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    cameraStream = stream;
                    video.srcObject = stream;
                    video.style.display = 'block';
                    document.getElementById('registerPlaceholder').style.display = 'none';
                    document.getElementById('registerImage').style.display = 'none';
                    video.play();
                    captureBtn.textContent = '拍照';
                    showToast('摄像头已启动，调整姿势后点击拍照按钮', 'info');
                })
                .catch(err => {
                    showToast('摄像头启动失败，请使用上传照片功能', 'error');
                });
        }
    });
    
    uploadBtn.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                const compressedDataUrl = compressImage(img, 1200, 1200, 0.92);
                document.getElementById('registerImage').src = compressedDataUrl;
                document.getElementById('registerImage').style.display = 'block';
                video.style.display = 'none';
                document.getElementById('registerPlaceholder').style.display = 'none';
                capturedImage = compressedDataUrl;
                registerFaceBtn.disabled = false;
                registerFaceBtn.textContent = '注册人脸';
                
                const headerEnd = compressedDataUrl.indexOf(',') + 1;
                const compressedBytes = Math.round((compressedDataUrl.length - headerEnd) * 0.75);
                const originalSizeKB = (file.size / 1024).toFixed(1);
                const compressedSizeKB = (compressedBytes / 1024).toFixed(1);
                if (compressedBytes < file.size) {
                    showToast(`图片已优化: ${originalSizeKB}KB → ${compressedSizeKB}KB`, 'success');
                } else {
                    showToast(`图片已处理 (${compressedSizeKB}KB)`, 'success');
                }
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
        fileInput.value = '';
    });
    
    registerFaceBtn.addEventListener('click', async () => {
        const name = document.getElementById('name').value.trim();
        const studentId = document.getElementById('studentId').value.trim();
        const department = document.getElementById('department').value.trim();
        const role = document.getElementById('role').value;
        
        if (!name) { showToast('请先填写姓名', 'error'); document.getElementById('name').focus(); return; }
        if (!studentId) { showToast('请先填写学号', 'error'); document.getElementById('studentId').focus(); return; }
        if (!capturedImage) { showToast('请先拍照或上传人脸照片', 'error'); return; }
        
        registerFaceBtn.disabled = true;
        registerFaceBtn.textContent = '正在提取人脸特征...';
        
        const statusEl = document.getElementById('registerStatus');
        statusEl.textContent = '正在提取人脸特征...';
        statusEl.className = 'status-msg';
        
        const formData = new FormData();
        formData.append('student_id', studentId);
        formData.append('name', name);
        formData.append('department', department);
        formData.append('role', role);
        formData.append('image_base64', capturedImage);
        
        try {
            const response = await fetch(`${API_BASE}/face/register`, {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            
            if (result.success) {
                registerForm.reset();
                document.getElementById('registerImage').style.display = 'none';
                document.getElementById('registerPlaceholder').style.display = 'block';
                video.style.display = 'none';
                capturedImage = null;
                registerFaceBtn.disabled = true;
                registerFaceBtn.textContent = '请先拍照或上传照片';
                captureBtn.textContent = '拍照';
                document.getElementById('studentId').readOnly = false;
                statusEl.textContent = `注册成功！置信度: ${(result.confidence * 100).toFixed(1)}%`;
                statusEl.className = 'status-msg success';
                
                loadPersons();
                
                setTimeout(() => {
                    statusEl.textContent = '';
                    statusEl.className = 'status-msg';
                }, 4000);
            } else {
                statusEl.textContent = result.error || '注册失败，请重试';
                statusEl.className = 'status-msg error';
                registerFaceBtn.disabled = false;
                registerFaceBtn.textContent = '重试注册';
            }
        } catch (error) {
            statusEl.textContent = '网络错误，请检查连接';
            statusEl.className = 'status-msg error';
            registerFaceBtn.disabled = false;
            registerFaceBtn.textContent = '重试注册';
        }
    });
    
    initFaceMatchTest();
}

function initFaceMatchTest() {
    const uploadBtn = document.getElementById('matchTestUploadBtn');
    const fileInput = document.getElementById('matchTestFileInput');
    const captureBtn = document.getElementById('matchTestCaptureBtn');
    const matchVideo = document.getElementById('matchTestVideo');
    const matchImage = document.getElementById('matchTestImage');
    const placeholder = document.getElementById('matchTestPlaceholder');
    const matchBtn = document.getElementById('matchTestBtn');
    const clearBtn = document.getElementById('matchTestClearBtn');
    const matchStatus = document.getElementById('matchTestStatus');
    const matchResult = document.getElementById('matchTestResult');
    
    let matchTestImageData = null;
    let matchTestStream = null;
    
    function clearMatchImage() {
        matchTestImageData = null;
        matchImage.src = '';
        matchImage.style.display = 'none';
        matchVideo.style.display = 'none';
        placeholder.style.display = 'flex';
        matchBtn.disabled = true;
        matchStatus.textContent = '';
        matchResult.style.display = 'none';
        clearBtn.style.display = 'none';
        captureBtn.textContent = '拍照';
        if (matchTestStream) {
            matchTestStream.getTracks().forEach(t => t.stop());
            matchTestStream = null;
        }
    }
    
    clearBtn.addEventListener('click', clearMatchImage);
    
    captureBtn.addEventListener('click', () => {
        if (matchTestStream) {
            const canvas = document.createElement('canvas');
            canvas.width = matchVideo.videoWidth;
            canvas.height = matchVideo.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(matchVideo, 0, 0);
            const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
            matchTestStream.getTracks().forEach(t => t.stop());
            matchTestStream = null;
            matchVideo.style.display = 'none';
            matchImage.src = dataUrl;
            matchImage.style.display = 'block';
            matchTestImageData = dataUrl;
            matchBtn.disabled = false;
            clearBtn.style.display = 'inline-flex';
            matchResult.style.display = 'none';
            matchStatus.textContent = '';
            captureBtn.textContent = '重新拍照';
            showToast('已捕获图像，点击开始匹配', 'success');
        } else {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    matchTestStream = stream;
                    matchVideo.srcObject = stream;
                    matchVideo.style.display = 'block';
                    matchImage.style.display = 'none';
                    placeholder.style.display = 'none';
                    matchVideo.play();
                    captureBtn.textContent = '拍照';
                    showToast('摄像头已启动，调整姿势后点击拍照按钮', 'info');
                })
                .catch(err => {
                    showToast('摄像头启动失败，请使用上传照片功能', 'error');
                });
        }
    });
    
    uploadBtn.addEventListener('click', () => {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleMatchFile(file);
        }
        fileInput.value = '';
    });
    
    function handleMatchFile(file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                const compressedDataUrl = compressImage(img, 1200, 1200, 0.92);
                
                matchImage.src = compressedDataUrl;
                matchImage.style.display = 'block';
                matchVideo.style.display = 'none';
                placeholder.style.display = 'none';
                matchTestImageData = compressedDataUrl;
                matchBtn.disabled = false;
                clearBtn.style.display = 'inline-flex';
                matchResult.style.display = 'none';
                matchStatus.textContent = '';
                
                if (matchTestStream) {
                    matchTestStream.getTracks().forEach(t => t.stop());
                    matchTestStream = null;
                }
                captureBtn.textContent = '拍照';
                
                const headerEnd = compressedDataUrl.indexOf(',') + 1;
                const compressedBytes = Math.round((compressedDataUrl.length - headerEnd) * 0.75);
                const compressedSizeKB = (compressedBytes / 1024).toFixed(1);
                showToast(`图片已处理 (${compressedSizeKB}KB)`, 'success');
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }
    
    matchBtn.addEventListener('click', async () => {
        if (!matchTestImageData) return;
        
        matchBtn.disabled = true;
        matchBtn.textContent = '匹配中...';
        matchStatus.textContent = '正在提取人脸特征...';
        matchStatus.className = 'status-text';
        matchResult.style.display = 'none';
        
        const formData = new FormData();
        formData.append('image_base64', matchTestImageData);
        
        try {
            matchStatus.textContent = '正在比对特征...';
            
            const response = await fetch(`${API_BASE}/face/match`, {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            
            if (result.success) {
                matchStatus.textContent = `检测到 ${result.detected_faces} 个人脸`;
                matchStatus.className = 'status-text success';
                
                if (result.matched && result.candidates && result.candidates.length > 0) {
                    displayMatchCandidates(result.candidates);
                } else {
                    matchResult.innerHTML = `
                        <div class="match-placeholder">
                            <div class="placeholder-icon">❌</div>
                            <p>未找到匹配的人员</p>
                            <p class="small-text">该人脸未在系统中注册</p>
                        </div>
                    `;
                    matchResult.style.display = 'block';
                }
            } else {
                showToast('❌ ' + (result.error || '匹配失败'), 'error');
                matchStatus.textContent = result.error || '匹配失败';
                matchStatus.className = 'status-text error';
            }
        } catch (error) {
            showToast('❌ 匹配失败: ' + error.message, 'error');
            matchStatus.textContent = '网络错误';
            matchStatus.className = 'status-text error';
        }
        
        matchBtn.disabled = false;
        matchBtn.textContent = '开始匹配';
    });
    
    function displayMatchCandidates(candidates) {
        const bestCandidate = candidates[0];
        let scoreClass = 'low';
        if (bestCandidate.match_score >= 0.8) {
            scoreClass = 'high';
        } else if (bestCandidate.match_score >= 0.6) {
            scoreClass = 'medium';
        }
        
        matchResult.innerHTML = `
            <div class="match-candidate best-match">
                <div class="match-candidate-avatar">👤</div>
                <div class="match-candidate-info">
                    <div class="match-candidate-name">${bestCandidate.name}</div>
                    <div class="match-candidate-id">${bestCandidate.student_id} | ${getRoleName(bestCandidate.role)}</div>
                </div>
                <span class="match-candidate-score ${scoreClass}">
                    ${(bestCandidate.match_score * 100).toFixed(1)}%
                </span>
            </div>
        `;
        matchResult.style.display = 'block';
    }
}

async function loadPersons() {
    const result = await apiRequest('/persons');
    
    if (result.success) {
        state.persons = result.data;
        renderPersonList();
        renderPersonsTable();
    }
}

function renderPersonList() {
    const container = document.getElementById('personList');
    const searchTerm = document.getElementById('searchPerson')?.value?.toLowerCase() || '';
    
    const filtered = state.persons.filter(p => 
        p.name.toLowerCase().includes(searchTerm) || 
        p.student_id.toLowerCase().includes(searchTerm)
    );
    
    if (filtered.length === 0) {
        container.innerHTML = '<p class="empty-text">暂无数据</p>';
        return;
    }
    
    container.innerHTML = filtered.map(person => `
        <div class="person-item" data-id="${person.id}" onclick="showPersonDetail(${person.id})" title="点击查看详情">
            <div class="person-avatar">
                ${person.face_photo 
                    ? `<img src="${person.face_photo}" alt="${person.name}">`
                    : '<iconify-icon icon="mdi:account-circle" style="font-size: 2rem; color: var(--gray-400);"></iconify-icon>'
                }
            </div>
            <div class="person-info">
                <span class="person-name">${person.name}</span>
                <span class="person-id">${person.student_id}</span>
                <span class="person-dept">${person.department || ''}</span>
            </div>
            <div class="person-meta">
                <span class="person-role">${getRoleName(person.role)}</span>
                <span class="person-arrow">›</span>
            </div>
        </div>
    `).join('');
}

function renderPersonsTable() {
    const tbody = document.getElementById('personsTableBody');
    
    if (state.persons.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="empty-text">暂无人员数据</td></tr>';
        return;
    }
    
    tbody.innerHTML = state.persons.map(person => {
        const permCount = (person.permissions || []).length;
        return `
        <tr>
            <td>${person.name}</td>
            <td>${person.student_id}</td>
            <td>${person.department || '-'}</td>
            <td>${getRoleName(person.role)}</td>
            <td><span class="status ${person.status === 'active' ? 'online' : 'offline'}">${person.status === 'active' ? '正常' : '禁用'}</span></td>
            <td><span class="tag tag-teal">${permCount} 个区域</span></td>
            <td>
                <button class="btn btn-sm btn-primary" onclick="selectPerson(${person.id})">选择</button>
                <button class="btn btn-sm btn-danger" onclick="deletePerson(${person.id})">删除</button>
            </td>
        </tr>
    `}).join('');
}

function getRoleName(role) {
    const roles = {
        'student': '学生',
        'teacher': '教师',
        'staff': '职工',
        'visitor': '访客'
    };
    return roles[role] || role;
}

function selectPerson(personId) {
    selectedPersonId = personId;
    const person = state.persons.find(p => p.id === personId);
    
    if (person) {
        document.getElementById('selectedPerson').textContent = `已选择：${person.name}（${person.student_id}）`;
        loadPersonPermissions(personId);
    }
}

async function deletePerson(personId) {
    if (!confirm('确定要删除该人员吗？')) return;
    
    const result = await apiRequest(`/persons/${personId}`, { method: 'DELETE' });
    
    if (result.success) {
        showToast('人员已删除', 'success');
        loadPersons();
    } else {
        showToast(result.error || '删除失败', 'error');
    }
}

function initRecognition() {
    const video = document.getElementById('recognitionVideo');
    const canvas = document.getElementById('recognitionCanvas');
    const uploadedImage = document.getElementById('uploadedRecognitionImage');
    const startBtn = document.getElementById('startRecognition');
    const stopBtn = document.getElementById('stopRecognition');
    const uploadBtn = document.getElementById('uploadRecognitionBtn');
    const resetBtn = document.getElementById('resetRecognition');
    const fileInput = document.getElementById('recognitionFileInput');
    const cameraSelect = document.getElementById('recognitionCamera');
    
    loadRecognitionCameras();
    
    cameraSelect.addEventListener('change', () => {
        recognitionCameraId = cameraSelect.value;
    });
    
    startBtn.addEventListener('click', async () => {
        try {
            recognitionStream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = recognitionStream;
            video.style.display = 'block';
            uploadedImage.style.display = 'none';
            canvas.style.display = 'none';
            recognitionEnabled = true;
            
            startBtn.disabled = true;
            stopBtn.disabled = false;
            resetBtn.disabled = true;
            uploadBtn.disabled = true;
            
            processVideoFrame();
        } catch (err) {
            showToast('无法访问摄像头，可使用上传照片识别功能', 'error');
        }
    });
    
    stopBtn.addEventListener('click', () => {
        recognitionEnabled = false;
        
        const recognitionVideo = document.getElementById('recognitionVideo');
        const recognitionCanvas = document.getElementById('recognitionCanvas');
        
        if (recognitionVideo && recognitionVideo.style.display !== 'none') {
            const ctx = recognitionCanvas.getContext('2d');
            recognitionCanvas.width = recognitionVideo.videoWidth || 640;
            recognitionCanvas.height = recognitionVideo.videoHeight || 480;
            ctx.drawImage(recognitionVideo, 0, 0, recognitionCanvas.width, recognitionCanvas.height);
            recognitionCanvas.style.display = 'block';
            recognitionVideo.style.display = 'none';
        }
        
        if (recognitionStream) {
            recognitionStream.getTracks().forEach(track => track.stop());
            recognitionStream = null;
        }
        
        startBtn.disabled = false;
        stopBtn.disabled = true;
        resetBtn.disabled = false;
        uploadBtn.disabled = false;
    });
    
    resetBtn.addEventListener('click', () => {
        const recognitionCanvas = document.getElementById('recognitionCanvas');
        const ctx = recognitionCanvas.getContext('2d');
        ctx.clearRect(0, 0, recognitionCanvas.width, recognitionCanvas.height);
        recognitionCanvas.style.display = 'none';
        
        const recognitionVideo = document.getElementById('recognitionVideo');
        recognitionVideo.style.display = 'none';
        recognitionVideo.srcObject = null;
        
        const uploadedRecognitionImage = document.getElementById('uploadedRecognitionImage');
        uploadedRecognitionImage.style.display = 'none';
        
        document.getElementById('matchResult').innerHTML = `
            <div class="match-placeholder">
                <div class="placeholder-icon">🔍</div>
                <p>等待识别...</p>
                <p class="small-text">点击"开始识别"按钮开始人脸匹配</p>
            </div>
        `;
        document.getElementById('faceMatches').innerHTML = '';
        document.getElementById('recognitionStatus').textContent = '🟢 就绪';
        document.getElementById('recognitionStatus').className = 'status-badge';
        document.getElementById('fpsCounter').textContent = 'FPS: --';
        
        startBtn.disabled = false;
        stopBtn.disabled = true;
        resetBtn.disabled = true;
        uploadBtn.disabled = false;
        recognitionEnabled = false;
        frameCount = 0;
    });
    
    uploadBtn.addEventListener('click', () => {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        fileInput.value = '';
        
        const reader = new FileReader();
        reader.onload = async (event) => {
            const originalDataUrl = event.target.result;
            
            const img = new Image();
            img.onload = async () => {
                const compressedDataUrl = compressImage(img, 1200, 1200, 0.92);
                
                uploadedImage.src = compressedDataUrl;
                uploadedImage.style.display = 'block';
                video.style.display = 'none';
                
                if (recognitionEnabled) {
                    stopBtn.click();
                }
                
                const ctx = canvas.getContext('2d');
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                
                document.getElementById('recognitionStatus').textContent = '🔄 识别中...';
                document.getElementById('recognitionStatus').className = 'status-badge processing';
                
                showToast('正在识别...', 'success');
                
                await recognizePhoto(compressedDataUrl);
            };
            img.src = originalDataUrl;
        };
        reader.readAsDataURL(file);
    });
    
    async function recognizePhoto(imageData) {
        try {
            const formData = new FormData();
            formData.append('image_base64', imageData);
            formData.append('skip_liveness', 'true');
            if (recognitionCameraId) {
                formData.append('camera_id', recognitionCameraId);
            }
            
            const response = await fetch(`${API_BASE}/face/recognize`, {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            
            if (result.success) {
                if (result.recognized) {
                    showToast(`🎉 识别成功！欢迎 ${result.person.name}`, 'success');
                    displayMatchResult(result.person, result.confidence, true, '');
                } else {
                    let message = result.message || '识别失败';
                    if (result.liveness_results && !result.liveness) {
                        message = '活体检测失败';
                    }
                    showToast(message, 'error');
                    displayMatchResult(null, 0, false, message);
                }
            } else {
                showToast(result.error || '识别失败', 'error');
                displayMatchResult(null, 0, false, '识别失败');
            }
            
            await updateStats();
            await loadAbnormalRecords();
        } catch (error) {
            showToast('识别失败: ' + error.message, 'error');
            displayMatchResult(null, 0, false, '网络错误');
        }
        
        document.getElementById('recognitionStatus').textContent = '🟢 就绪';
        document.getElementById('recognitionStatus').className = 'status-badge';
        document.getElementById('resetRecognition').disabled = false;
    }
    
    function displayMatchResult(person, confidence, success, message = '') {
        const matchResultDiv = document.getElementById('matchResult');
        
        if (success && person) {
            let confidenceClass = 'low';
            if (confidence >= 0.8) {
                confidenceClass = 'high';
            } else if (confidence >= 0.6) {
                confidenceClass = 'medium';
            }
            
            matchResultDiv.innerHTML = `
                <div class="match-card success">
                    <div class="match-avatar">👤</div>
                    <div class="match-info">
                        <div class="match-name">${person.name}</div>
                        <div class="match-id">学号: ${person.student_id}</div>
                        <div class="match-confidence ${confidenceClass}">
                            匹配度: ${(confidence * 100).toFixed(1)}%
                        </div>
                    </div>
                    <span class="match-status grant">允许通行</span>
                </div>
            `;
        } else {
            let cardClass = 'danger';
            let displayMessage = message;
            
            if (message === '活体检测失败') {
                cardClass = 'warning';
            }
            
            matchResultDiv.innerHTML = `
                <div class="match-card ${cardClass}">
                    <div class="match-avatar">❌</div>
                    <div class="match-info">
                        <div class="match-name">未知人员</div>
                        <div class="match-confidence low">
                            ${displayMessage}
                        </div>
                    </div>
                    <span class="match-status deny">拒绝通行</span>
                </div>
            `;
        }
    }
    
    function processVideoFrame() {
        if (!recognitionEnabled) return;
        
        const ctx = canvas.getContext('2d');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const pixels = imageData.data;
        
        let totalBrightness = 0;
        let nonBlackPixels = 0;
        for (let i = 0; i < pixels.length; i += 4) {
            const brightness = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
            if (brightness > 10) {
                nonBlackPixels++;
            }
            totalBrightness += brightness;
        }
        
        const avgBrightness = totalBrightness / (pixels.length / 4);
        const nonBlackRatio = nonBlackPixels / (pixels.length / 4);
        
        if (avgBrightness < 15 || nonBlackRatio < 0.02) {
            document.getElementById('recognitionStatus').textContent = '⚠️ 等待画面...';
            document.getElementById('recognitionStatus').className = 'status-badge warning';
            requestAnimationFrame(processVideoFrame);
            return;
        }
        
        const now = Date.now();
        
        if (now - lastFrameSent >= 200) {
            const dataUrl = canvas.toDataURL('image/jpeg', 0.75);
            
            if (socket && socket.connected) {
                socket.emit('video_frame', { frame: dataUrl, camera_id: recognitionCameraId || undefined });
            }
            lastFrameSent = now;
        }
        
        frameCount++;
        if (now - lastFpsUpdate >= 1000) {
            document.getElementById('fpsCounter').textContent = `FPS: ${frameCount}`;
            frameCount = 0;
            lastFpsUpdate = now;
        }
        
        requestAnimationFrame(processVideoFrame);
    }
}

function handleRecognitionResult(result) {
    const canvas = document.getElementById('recognitionCanvas');
    const ctx = canvas.getContext('2d');
    const video = document.getElementById('recognitionVideo');
    
    const matchResultDiv = document.getElementById('matchResult');
    const faceMatchesDiv = document.getElementById('faceMatches');
    const recognitionStatus = document.getElementById('recognitionStatus');
    
    recognitionStatus.textContent = '🔄 识别中...';
    recognitionStatus.className = 'status-badge processing';
    
    if (result.faces && result.faces.length > 0) {
        const matchedFace = result.faces[0];
        
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        let cardClass = 'danger';
        let statusClass = 'deny';
        let statusText = '拒绝通行';
        let confidenceClass = 'low';
        
        if (matchedFace.recognized) {
            cardClass = 'success';
            statusClass = 'grant';
            statusText = '允许通行';
            
            if (matchedFace.match_score >= 0.8) {
                confidenceClass = 'high';
            } else if (matchedFace.match_score >= 0.6) {
                confidenceClass = 'medium';
            }
        } else if (!matchedFace.liveness) {
            cardClass = 'warning';
        }
        
        const isPermissionDenied = matchedFace.reason === 'permission_denied' && matchedFace.person;
        
        matchResultDiv.innerHTML = `
            <div class="match-card ${cardClass}">
                <div class="match-avatar">${isPermissionDenied ? '🚫' : '👤'}</div>
                <div class="match-info">
                    <div class="match-name">${matchedFace.recognized ? matchedFace.person.name : (isPermissionDenied ? matchedFace.person.name : '未知人员')}</div>
                    ${(matchedFace.recognized || isPermissionDenied) ? `<div class="match-id">学号: ${matchedFace.person.student_id}</div>` : ''}
                    <div class="match-confidence ${confidenceClass}">
                        ${isPermissionDenied ? 
                            (matchedFace.deny_detail || '无此区域通行权限') : 
                            (matchedFace.recognized ? `匹配度: ${(matchedFace.match_score * 100).toFixed(1)}%` : 
                             (matchedFace.liveness ? '未注册用户' : '活体检测失败'))
                        }
                    </div>
                </div>
                <span class="match-status ${statusClass}">${isPermissionDenied ? '权限不足' : statusText}</span>
            </div>
        `;
        
        faceMatchesDiv.innerHTML = result.faces.map((face, index) => {
            const confPercent = face.recognized ? (face.match_score * 100).toFixed(0) : '0';
            return `
                <div class="face-match-tag">
                    <span>人脸${index + 1}</span>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confPercent}%"></div>
                    </div>
                </div>
            `;
        }).join('');
        
        result.faces.forEach(face => {
            const [x1, y1, x2, y2] = face.box;
            
            const permDenied = face.reason === 'permission_denied';
            ctx.strokeStyle = face.recognized ? '#2ecc71' : (permDenied ? '#f39c12' : (face.liveness ? '#f39c12' : '#e74c3c'));
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            
            ctx.fillStyle = face.recognized ? '#2ecc71' : (permDenied ? '#f39c12' : (face.liveness ? '#f39c12' : '#e74c3c'));
            ctx.font = 'bold 18px Arial';
            
            const label = face.recognized ? 
                `${face.person.name} (${(face.match_score * 100).toFixed(0)}%)` : 
                (face.reason === 'permission_denied' ? 
                    `${face.person.name} 无通行权限` : 
                    (face.liveness ? '未注册' : '活体检测失败'));
            
            ctx.fillText(label, x1, y1 - 10);
        });
        
        recognitionEnabled = false;
        canvas.style.display = 'block';
        video.style.display = 'none';
        
        if (recognitionStream) {
            recognitionStream.getTracks().forEach(track => track.stop());
            recognitionStream = null;
        }
        
        const startBtn = document.getElementById('startRecognition');
        const stopBtn = document.getElementById('stopRecognition');
        const resetBtn = document.getElementById('resetRecognition');
        startBtn.disabled = false;
        stopBtn.disabled = true;
        resetBtn.disabled = false;
        
        recognitionStatus.textContent = '✅ 识别完成';
        recognitionStatus.className = 'status-badge success';
    } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        matchResultDiv.innerHTML = `
            <div class="match-placeholder">
                <div class="placeholder-icon">🔍</div>
                <p>未检测到人脸</p>
                <p class="small-text">请将人脸对准摄像头</p>
            </div>
        `;
        faceMatchesDiv.innerHTML = '';
        
        recognitionStatus.textContent = '✅ 识别完成';
        recognitionStatus.className = 'status-badge success';
    }
}

function addAccessLog(log) {
    const list = document.getElementById('accessLogList');
    
    const logElement = document.createElement('div');
    logElement.className = `log-item ${log.result === 'success' ? '' : 'danger'}`;
    logElement.innerHTML = `
        <div class="log-time">${new Date(log.access_time).toLocaleTimeString()}</div>
        <div class="log-name">${log.name || '未知'}</div>
        <div class="log-status">${log.result === 'success' ? '✅ 允许通行' : '❌ 拒绝通行'}</div>
    `;
    
    list.insertBefore(logElement, list.firstChild);
    
    if (list.children.length > 10) {
        list.removeChild(list.lastChild);
    }
}

function addAbnormalRecord(record) {
    state.abnormalRecords.unshift(record);
    
    if (state.abnormalRecords.length > 100) {
        state.abnormalRecords.pop();
    }
    
    if (state.currentTab === 'alarm') {
        renderAlarmTable();
        updateAlarmStats();
    }
}

async function updateStats() {
    const result = await apiRequest('/access-logs/stats');
    
    if (result.success) {
        document.getElementById('totalCount').textContent = result.data.total_today;
        document.getElementById('successCount').textContent = result.data.success_today;
        document.getElementById('abnormalCount').textContent = result.data.abnormal_today;
        document.getElementById('denyCount').textContent = result.data.deny_today;
    }
}

function initAlarm() {
    const refreshBtn = document.getElementById('refreshAlarm');
    const clearBtn = document.getElementById('clearAlarm');
    const filterSelect = document.getElementById('alarmFilter');
    
    refreshBtn.addEventListener('click', loadAbnormalRecords);
    
    clearBtn.addEventListener('click', async () => {
        if (!confirm('确定要清空所有告警记录吗？')) return;
        
        const result = await apiRequest('/abnormal-records', { method: 'DELETE' });
        
        if (result.success) {
            showToast('记录已清空', 'success');
            loadAbnormalRecords();
        }
    });
    
    filterSelect.addEventListener('change', loadAbnormalRecords);
}

async function loadAbnormalRecords() {
    const type = document.getElementById('alarmFilter').value;
    const endpoint = type ? `/abnormal-records?type=${type}` : '/abnormal-records';
    
    const result = await apiRequest(endpoint);
    
    if (result.success) {
        state.abnormalRecords = result.data;
        renderAlarmTable();
        updateAlarmStats();
    }
}

function renderAlarmTable() {
    const tbody = document.getElementById('alarmTableBody');
    
    if (state.abnormalRecords.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="empty-text">暂无告警记录</td></tr>';
        return;
    }
    
    tbody.innerHTML = state.abnormalRecords.map((record, i) => {
        const status = record.status || 'pending';
        const statusText = { pending: '待处理', processed: '已处理', ignored: '已忽略', false_alarm: '误报' }[status] || status;
        const statusClass = { pending: 'warning', processed: 'success', ignored: 'default', false_alarm: 'default' }[status] || '';
        return `
        <tr>
            <td><span class="alarm-tag ${getAlarmTypeClass(record.type)}">${record.type}</span></td>
            <td>${record.description || '-'}</td>
            <td>${new Date(record.timestamp).toLocaleString()}</td>
            <td><span class="status-badge ${statusClass}">${statusText}</span></td>
            <td>
                <div class="btn-group" style="margin-top:0;">
                    <button class="btn btn-sm btn-success" onclick="updateAlarmStatus(${record.id}, 'processed', ${i})" style="${status==='processed'?'opacity:0.5':''}">✓</button>
                    <button class="btn btn-sm btn-outline" onclick="updateAlarmStatus(${record.id}, 'false_alarm', ${i})">✕</button>
                    <button class="btn btn-sm btn-outline" onclick="updateAlarmStatus(${record.id}, 'ignored', ${i})">−</button>
                </div>
            </td>
        </tr>
    `}).join('');
}

function getAlarmTypeClass(type) {
    if (type.includes('未注册')) return 'danger';
    if (type.includes('活体')) return 'warning';
    if (type.includes('权限')) return 'warning';
    if (type.includes('尾随')) return 'info';
    return '';
}

async function updateAlarmStatus(recordId, status, index) {
    const result = await apiRequest(`/abnormal-records/${recordId}`, {
        method: 'PUT',
        body: JSON.stringify({ status })
    });
    if (result.success) {
        state.abnormalRecords[index].status = status;
        renderAlarmTable();
        updateAlarmStats();
        showToast(`告警已标记为: ${status}`, 'success');
    } else {
        showToast('更新失败，请重试', 'error');
    }
}

function updateAlarmStats() {
    const unknown = state.abnormalRecords.filter(r => r.type.includes('未注册')).length;
    const liveness = state.abnormalRecords.filter(r => r.type.includes('活体')).length;
    const tailgating = state.abnormalRecords.filter(r => r.type.includes('尾随')).length;
    
    document.getElementById('unknownCount').textContent = unknown;
    document.getElementById('livenessCount').textContent = liveness;
    document.getElementById('tailgatingCount').textContent = tailgating;
}

function initReport() {
    const generateBtn = document.getElementById('generateReport');
    const exportCSVBtn = document.getElementById('exportCSV');
    const exportReportBtn = document.getElementById('exportReport');
    
    generateBtn.addEventListener('click', generateReport);
    
    exportCSVBtn.addEventListener('click', async () => {
        const startDate = document.getElementById('reportStartDate').value;
        const endDate = document.getElementById('reportEndDate').value;
        
        window.location.href = `${API_BASE}/export/csv?start_date=${startDate}&end_date=${endDate}`;
    });
    
    exportReportBtn.addEventListener('click', () => {
        const reportContent = document.querySelector('#page-report').innerText;
        const blob = new Blob([reportContent], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `access_report_${new Date().toISOString().split('T')[0]}.txt`;
        a.click();
    });
}

async function generateReport() {
    const startDate = document.getElementById('reportStartDate').value;
    const endDate = document.getElementById('reportEndDate').value;
    
    if (!startDate || !endDate) {
        showToast('请选择日期范围', 'error');
        return;
    }
    
    const logsResult = await apiRequest(`/access-logs?start_date=${startDate}&end_date=${endDate}`);
    
    if (logsResult.success) {
        const logs = logsResult.data;
        const total = logs.length;
        const success = logs.filter(l => l.result === 'success').length;
        const fail = total - success;
        
        document.getElementById('reportTotal').textContent = total;
        document.getElementById('reportSuccess').textContent = success;
        document.getElementById('reportFail').textContent = fail;
        
        const abnormalResult = await apiRequest(`/abnormal-records?date_from=${startDate}&date_to=${endDate}`);
        if (abnormalResult.success) {
            document.getElementById('reportAbnormal').textContent = abnormalResult.data.length;
        }
        
        renderReportTable(logs);
    }
}

function renderReportTable(logs) {
    const tbody = document.getElementById('reportTableBody');
    
    if (logs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="empty-text">暂无记录</td></tr>';
        return;
    }
    
    tbody.innerHTML = logs.map(log => `
        <tr>
            <td>${new Date(log.access_time).toLocaleString()}</td>
            <td>${log.name || '-'}</td>
            <td>${log.student_id || '-'}</td>
            <td>${log.result === 'success' ? '✅ 成功' : '❌ 失败'}</td>
            <td>${log.confidence ? log.confidence.toFixed(2) : '-'}</td>
        </tr>
    `).join('');
}

async function initPermission() {
    const refreshBtn = document.getElementById('refreshPersons');
    const importBtn = document.getElementById('importPersons');
    const importFile = document.getElementById('importFile');
    const permissionForm = document.getElementById('permissionForm');
    
    refreshBtn.addEventListener('click', () => { loadPersons(); });
    
    importBtn.addEventListener('click', () => { importFile.click(); });
    
    importFile.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const formData = new FormData();
        formData.append('file', file);
        try {
            const response = await fetch(`${API_BASE}/persons/import`, {
                method: 'POST', body: formData
            });
            const result = await response.json();
            if (result.success) {
                showToast(result.message, 'success');
                loadPersons();
            } else {
                showToast(result.error || '导入失败', 'error');
            }
        } catch (error) {
            showToast('导入失败: ' + error.message, 'error');
        }
        importFile.value = '';
    });
    
    permissionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!selectedPersonId) {
            showToast('请先在左侧人员表中点击「选择」', 'error');
            return;
        }
        
        const checked = document.querySelectorAll('#gateCheckboxes .checkbox-area-item.checked');
        const areas = [];
        checked.forEach(el => {
            const val = el.dataset.area;
            if (val) areas.push(val);
        });
        
        if (areas.length === 0) {
            showToast('请至少选择一个通行区域', 'error');
            return;
        }
        
        const formData = {
            person_id: selectedPersonId,
            areas: areas,
            start_time: document.getElementById('startTime').value,
            end_time: document.getElementById('endTime').value,
            valid_from: document.getElementById('validFrom').value,
            valid_to: document.getElementById('validTo').value
        };
        
        const result = await apiRequest('/permissions', {
            method: 'POST',
            body: JSON.stringify(formData)
        });
        
        if (result.success) {
            showToast(`已为 ${result.count} 个区域设置权限`, 'success');
            uncheckAllGates();
            document.getElementById('areaCount').textContent = '未选择区域';
            document.getElementById('areaCount').className = 'area-count';
            loadPersons();
            loadPersonPermissions(selectedPersonId);
        } else {
            showToast(result.error || '设置失败', 'error');
        }
    });
    
    document.querySelectorAll('.time-preset').forEach(btn => {
        btn.addEventListener('click', () => {
            document.getElementById('startTime').value = btn.dataset.start;
            document.getElementById('endTime').value = btn.dataset.end;
            document.querySelectorAll('.time-preset').forEach(b => b.classList.remove('active-preset'));
            btn.classList.add('active-preset');
        });
    });
    
    loadCameras();
}

function uncheckAllGates() {
    document.querySelectorAll('#gateCheckboxes .checkbox-area-item, #gateCheckboxes .checkbox-area-all').forEach(el => {
        el.classList.remove('checked');
        const cb = el.querySelector('input[type="checkbox"]');
        if (cb) cb.checked = false;
    });
}

async function loadCameras() {
    try {
        const result = await apiRequest('/cameras');
        state.cameras = result.success ? result.data : [];
        const container = document.getElementById('gateCheckboxes');
        if (result.success && result.data.length > 0) {
            container.innerHTML = `
                <div class="checkbox-area-item checkbox-area-all" data-area="__ALL__">
                    <input type="checkbox" id="gate_all">
                    <span>全部区域（不限门禁）</span>
                </div>
                ${result.data.map(cam => `
                <div class="checkbox-area-item" data-area="${cam.camera_id}">
                    <input type="checkbox" id="gate_${cam.camera_id}">
                    <span>${cam.name}</span>
                </div>`).join('')}
            `;
            
            const countEl = document.getElementById('areaCount');
            container.querySelectorAll('.checkbox-area-item, .checkbox-area-all').forEach(item => {
                item.addEventListener('click', (e) => {
                    e.preventDefault();
                    const isAll = item.dataset.area === '__ALL__';
                    
                    if (isAll) {
                        const isNowChecked = !item.classList.contains('checked');
                        container.querySelectorAll('.checkbox-area-item, .checkbox-area-all').forEach(el => {
                            if (isNowChecked) {
                                el.classList.add('checked');
                                const cb = el.querySelector('input');
                                if (cb) cb.checked = true;
                            } else {
                                el.classList.remove('checked');
                                const cb = el.querySelector('input');
                                if (cb) cb.checked = false;
                            }
                        });
                    } else {
                        item.classList.toggle('checked');
                        const cb = item.querySelector('input[type="checkbox"]');
                        if (cb) cb.checked = item.classList.contains('checked');
                        
                        const allBtn = container.querySelector('.checkbox-area-all');
                        const allChecked = container.querySelectorAll('.checkbox-area-item:not(.checkbox-area-all).checked').length === result.data.length;
                        if (allChecked) {
                            allBtn.classList.add('checked');
                            const allCb = allBtn.querySelector('input');
                            if (allCb) allCb.checked = true;
                        } else {
                            allBtn.classList.remove('checked');
                            const allCb = allBtn.querySelector('input');
                            if (allCb) allCb.checked = false;
                        }
                    }
                    
                    const selected = container.querySelectorAll('.checkbox-area-item:not(.checkbox-area-all).checked').length;
                    if (selected > 0) {
                        countEl.textContent = `已选 ${selected}/${result.data.length}`;
                        countEl.className = 'area-count has-selection';
                    } else {
                        countEl.textContent = '未选择区域';
                        countEl.className = 'area-count';
                    }
                });
            });
        } else {
            container.innerHTML = '<p class="empty-text" style="padding:12px;">未检测到门禁区域（请重启后端服务）</p>';
        }
    } catch {
        const container = document.getElementById('gateCheckboxes');
        container.innerHTML = '<p class="empty-text" style="padding:12px;">加载失败，请重启后端服务</p>';
    }
}

async function loadRecognitionCameras() {
    const result = await apiRequest('/cameras');
    if (result.success) {
        state.cameras = result.data;
        const select = document.getElementById('recognitionCamera');
        select.innerHTML = '<option value="">全部区域（不限门禁）</option>';
        result.data.forEach(cam => {
            select.innerHTML += `<option value="${cam.camera_id}">📍 ${cam.name}</option>`;
        });
    }
}

async function loadPersonPermissions(personId) {
    const result = await apiRequest(`/permissions?person_id=${personId}`);
    const section = document.getElementById('currentPerms');
    const list = document.getElementById('permList');
    
    if (result.success && result.data.length > 0) {
        const camResult = await apiRequest('/cameras');
        const camMap = {};
        if (camResult.success) {
            camResult.data.forEach(c => { camMap[c.camera_id] = c.name; });
        }
        
        section.style.display = 'block';
        list.innerHTML = result.data.map(p => `
            <div class="perm-item">
                <div class="perm-info">
                    <span class="perm-area">${camMap[p.area] || p.area}</span>
                    <span class="perm-time">${p.start_time || '--'} - ${p.end_time || '--'}</span>
                    ${p.valid_from || p.valid_to ? `<span class="perm-date">${p.valid_from || '--'} 至 ${p.valid_to || '--'}</span>` : ''}
                </div>
                <button class="btn btn-sm btn-danger" onclick="deletePermission(${p.id}, ${personId})">删除</button>
            </div>
        `).join('');
    } else {
        section.style.display = 'none';
    }
}

async function deletePermission(permId, personId) {
    if (!confirm('确定要删除该权限吗？')) return;
    const result = await apiRequest(`/permissions/${permId}`, { method: 'DELETE' });
    if (result.success) {
        showToast('权限已删除', 'success');
        loadPersons();
        loadPersonPermissions(personId);
    } else {
        showToast(result.error || '删除失败', 'error');
    }
}

function initApp() {
    initSocket();
    initTabs();
    
    loadCameras();

    const sidebar = document.getElementById('sidebar');
    const toggleBtn = document.getElementById('toggleSidebar');
    let hoverLocked = false;

    toggleBtn.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
        if (sidebar.classList.contains('collapsed')) {
            hoverLocked = false;
        }
    });

    sidebar.addEventListener('mouseenter', () => {
        if (sidebar.classList.contains('collapsed') && !hoverLocked) {
            sidebar.classList.remove('collapsed');
            sidebar.classList.add('hover-expanded');
        }
    });

    sidebar.addEventListener('mouseleave', () => {
        if (sidebar.classList.contains('hover-expanded')) {
            sidebar.classList.add('collapsed');
            sidebar.classList.remove('hover-expanded');
        }
    });
    initRegister();
    initRecognition();
    initAlarm();
    initReport();
    initPermission();
    
    updateTime();
    setInterval(updateTime, 1000);
    
    setDefaultDates();
    loadPersons();
    updateStats();
    loadAbnormalRecords();
    
    if (document.getElementById('searchPerson')) {
        document.getElementById('searchPerson').addEventListener('input', renderPersonList);
    }
}

function showPersonDetail(personId) {
    const person = state.persons.find(p => p.id === personId);
    if (!person) {
        showToast('未找到该人员', 'error');
        return;
    }

    document.getElementById('detailModalTitle').textContent = person.name + ' 的详情';
    
    const camMap = {};
    if (state.cameras) {
        state.cameras.forEach(c => { camMap[c.camera_id] = c.name; });
    }
    
    const perms = person.permissions || [];
    const hasFace = !!person.face_photo;
    
    const permsHtml = perms.length > 0 ? perms.map(p => `
        <tr>
            <td><span class="perm-badge">${camMap[p.area] || p.area}</span></td>
            <td>${p.start_time || '-'} ~ ${p.end_time || '-'}</td>
            <td>${p.valid_from || '不限'} ~ ${p.valid_to || '不限'}</td>
        </tr>
    `).join('') : '<tr><td colspan="3" class="empty-text" style="padding:20px;">暂无授权区域</td></tr>';
    
    document.getElementById('detailModalBody').innerHTML = `
        <div class="detail-face-section">
            <div class="detail-face-container">
                ${hasFace ? 
                    `<img src="${person.face_photo}" alt="${person.name}">` :
                    `<div class="detail-face-placeholder">
                        <iconify-icon icon="mdi:account-circle" style="font-size:4rem;color:var(--gray-300);"></iconify-icon>
                        <p>暂无人脸照片</p>
                        <p class="small-text">请到「人脸注册」重新拍照注册</p>
                    </div>`
                }
            </div>
            ${!hasFace ? `
            <div style="text-align:center;margin-bottom:16px;">
                <button class="btn btn-sm btn-primary" onclick="goRegisterWithPerson(${person.id})" style="padding:6px 16px;">去注册人脸</button>
            </div>` : ''}
        </div>
        <div class="detail-grid">
            <div class="detail-item">
                <label>姓名</label>
                <span>${person.name}</span>
            </div>
            <div class="detail-item">
                <label>学号/工号</label>
                <span>${person.student_id}</span>
            </div>
            <div class="detail-item">
                <label>院系/部门</label>
                <span>${person.department || '-'}</span>
            </div>
            <div class="detail-item">
                <label>角色</label>
                <span>${getRoleName(person.role)}</span>
            </div>
        </div>
        <h4 class="perm-section-title">已授权区域</h4>
        <div class="perm-table-wrap">
            <table class="perm-detail-table">
                <thead>
                    <tr><th>区域</th><th>通行时段</th><th>有效期</th></tr>
                </thead>
                <tbody>${permsHtml}</tbody>
            </table>
        </div>
    `;
    
    document.getElementById('personDetailModal').classList.add('show');
}

function switchToRegister() {
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.querySelector('.nav-item[data-tab="register"]')?.classList.add('active');
    document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active'));
    document.getElementById('page-register')?.classList.add('active');
    
    if (pendingRegisterPerson) {
        document.getElementById('name').value = pendingRegisterPerson.name || '';
        document.getElementById('studentId').value = pendingRegisterPerson.student_id || '';
        document.getElementById('department').value = pendingRegisterPerson.department || '';
        document.getElementById('role').value = pendingRegisterPerson.role || 'student';
        document.getElementById('studentId').readOnly = true;
        showToast('已自动填入「' + pendingRegisterPerson.name + '」的信息，请拍照或上传照片', 'info');
        pendingRegisterPerson = null;
    }
}

function closePersonDetail() {
    document.getElementById('personDetailModal').classList.remove('show');
}

function goRegisterWithPerson(personId) {
    const person = state.persons.find(p => p.id === personId);
    if (!person) {
        showToast('未找到该人员', 'error');
        return;
    }
    pendingRegisterPerson = {
        name: person.name,
        student_id: person.student_id,
        department: person.department || '',
        role: person.role || 'student'
    };
    closePersonDetail();
    switchToRegister();
}
