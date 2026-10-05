from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import threading
from datetime import datetime, timedelta, timezone
from functools import wraps
import os
import secrets
import traceback

from models import db, init_db, Person, AccessLog, AbnormalRecord, Permission, User, Camera
from face_service_insightface import InsightFaceService
from liveness_service import LivenessDetector
from config import SIMILARITY_THRESHOLD, MIN_RECORD_INTERVAL, MAX_UPLOAD_SIZE, get_logger
from utils import decode_upload, decode_image

logger = get_logger(__name__)

app = Flask(__name__, static_folder='../frontend', static_url_path='')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE

CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

init_db(app)

face_service = InsightFaceService()
liveness_detector = LivenessDetector()

_tz_cn = timezone(timedelta(hours=8))

_last_record_lock = threading.Lock()
_last_record_time = {}


def _now():
    return datetime.now(_tz_cn)


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('user_id'):
            return jsonify({'success': False, 'error': '未登录', 'code': 'UNAUTHORIZED'}), 401
        return f(*args, **kwargs)
    return decorated


def _safe_emit(event, data):
    try:
        socketio.emit(event, data)
    except Exception as e:
        logger.warning(f"emit {event} failed: {e}")


def _get_feature(face_info, frame):
    feat = face_info.get('feature')
    if feat is not None:
        return feat
    face_roi = face_service.get_face_roi(frame, face_info['box'])
    return face_service.extract_feature(face_roi)


def _check_permission(person, camera_id=None):
    if person.status == 'disabled':
        return False
    perms = Permission.query.filter_by(person_id=person.id).all()
    if not perms:
        return True
    now = _now()
    current_time = now.strftime('%H:%M')
    current_date = now.date()
    for perm in perms:
        area_ok = True
        if camera_id:
            area_ok = (perm.area == camera_id)
        time_ok = True
        if perm.start_time and perm.end_time:
            time_ok = perm.start_time <= current_time <= perm.end_time
        date_ok = True
        if perm.valid_from and perm.valid_to:
            date_ok = perm.valid_from <= current_date <= perm.valid_to
        if area_ok and time_ok and date_ok:
            return True
    return False

_camera_name_cache = {}
def _get_camera_name(camera_id):
    if not camera_id:
        return '当前区域'
    if camera_id in _camera_name_cache:
        return _camera_name_cache[camera_id]
    cam = Camera.query.filter_by(camera_id=camera_id).first()
    name = cam.name if cam else camera_id
    _camera_name_cache[camera_id] = name
    return name


@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')


@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '')

        if not username or not password:
            return jsonify({'success': False, 'error': '用户名和密码不能为空'}), 400

        user = User.query.filter_by(username=username).first()
        if not user or not user.check_password(password):
            return jsonify({'success': False, 'error': '用户名或密码错误'}), 401

        session['user_id'] = user.id
        session['username'] = user.username
        session['role'] = user.role

        logger.info(f"User logged in: {username} ({user.role})")
        return jsonify({'success': True, 'data': user.to_dict(), 'message': '登录成功'})
    except Exception as e:
        logger.error(f"login: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    username = session.get('username', 'unknown')
    session.clear()
    logger.info(f"User logged out: {username}")
    return jsonify({'success': True, 'message': '已退出登录'})


@app.route('/api/auth/session', methods=['GET'])
def check_session():
    if session.get('user_id'):
        return jsonify({
            'success': True,
            'authenticated': True,
            'data': {
                'username': session.get('username'),
                'role': session.get('role')
            }
        })
    return jsonify({'success': True, 'authenticated': False})


@app.route('/api/persons', methods=['GET'])
@login_required
def get_persons():
    try:
        persons = Person.query.all()
        return jsonify({'success': True, 'data': [p.to_dict() for p in persons]})
    except Exception as e:
        logger.error(f"get_persons: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/persons', methods=['POST'])
@login_required
def create_person():
    try:
        data = request.json
        name = data.get('name')
        student_id = data.get('student_id')
        department = data.get('department', '')
        role = data.get('role', 'student')

        if not name or not student_id:
            return jsonify({'success': False, 'error': '姓名和学号不能为空'}), 400

        existing = Person.query.filter_by(student_id=student_id).first()
        if existing:
            return jsonify({'success': False, 'error': '学号已存在', 'person': existing.to_dict()}), 400

        person = Person(name=name, student_id=student_id, department=department, role=role)
        db.session.add(person)
        db.session.commit()
        logger.info(f"Created person: {name} ({student_id})")
        return jsonify({'success': True, 'data': person.to_dict()})
    except Exception as e:
        db.session.rollback()
        logger.error(f"create_person: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/persons/<int:person_id>', methods=['GET'])
@login_required
def get_person(person_id):
    try:
        person = Person.query.get(person_id)
        if not person:
            return jsonify({'success': False, 'error': '人员不存在'}), 404
        return jsonify({'success': True, 'data': person.to_dict()})
    except Exception as e:
        logger.error(f"get_person: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/persons/<int:person_id>', methods=['PUT'])
@login_required
def update_person(person_id):
    try:
        person = Person.query.get(person_id)
        if not person:
            return jsonify({'success': False, 'error': '人员不存在'}), 404
        data = request.json
        if 'name' in data:
            person.name = data['name']
        if 'department' in data:
            person.department = data['department']
        if 'role' in data:
            person.role = data['role']
        if 'status' in data:
            person.status = data['status']
        db.session.commit()
        return jsonify({'success': True, 'data': person.to_dict()})
    except Exception as e:
        db.session.rollback()
        logger.error(f"update_person: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/persons/<int:person_id>', methods=['DELETE'])
@login_required
def delete_person(person_id):
    try:
        person = Person.query.get(person_id)
        if not person:
            return jsonify({'success': False, 'error': '人员不存在'}), 404
        face_service.remove_face(person.student_id)
        AccessLog.query.filter_by(person_id=person_id).delete()
        Permission.query.filter_by(person_id=person_id).delete()
        db.session.delete(person)
        db.session.commit()
        logger.info(f"Deleted person: {person.name} ({person.student_id})")
        return jsonify({'success': True, 'message': '人员已删除'})
    except Exception as e:
        db.session.rollback()
        logger.error(f"delete_person: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/face/register', methods=['POST'])
@login_required
def register_face():
    try:
        frame = decode_upload(request)
        if frame is None:
            return jsonify({'success': False, 'error': '未提供有效图片'}), 400

        student_id = request.form.get('student_id')
        name = request.form.get('name', '')
        if not student_id:
            return jsonify({'success': False, 'error': '学号不能为空'}), 400

        faces = face_service.detect_faces_robust(frame)
        if not faces:
            return jsonify({'success': False, 'error': '未检测到人脸，请确保照片光线充足、人脸清晰'}), 400

        face_info = faces[0]
        feature = _get_feature(face_info, frame)
        face_roi = face_service.get_face_roi(frame, face_info['box'])

        face_service.register_face(student_id, feature, name)

        person = Person.query.filter_by(student_id=student_id).first()
        if not person and name:
            person = Person(name=name, student_id=student_id)
            db.session.add(person)
        if person:
            person.feature_vector = feature.tobytes()
            person.face_image = cv2.imencode('.jpg', face_roi)[1].tobytes()
            db.session.commit()

        logger.info(f"Face registered: {name} ({student_id}), confidence={face_info['confidence']:.3f}")
        return jsonify({
            'success': True,
            'message': '人脸注册成功',
            'confidence': float(face_info['confidence'])
        })
    except Exception as e:
        logger.error(f"register_face: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/face/match', methods=['POST'])
@login_required
def match_face():
    try:
        frame = decode_upload(request)
        if frame is None:
            return jsonify({'success': False, 'error': '未提供有效图片'}), 400

        faces = face_service.detect_faces_robust(frame)
        if not faces:
            return jsonify({'success': False, 'error': '未检测到人脸，请确保照片光线充足、人脸清晰可见'}), 400

        face_info = faces[0]
        feature = _get_feature(face_info, frame)

        match_threshold = face_service.similarity_threshold
        candidates = []
        best_score = -1
        best_match = None

        for student_id, data in face_service.features_db.items():
            features_list = face_service._get_features(student_id)
            max_score = max(
                (face_service.calculate_similarity(feature, f) for f in features_list),
                default=-1
            )
            person = Person.query.filter_by(student_id=student_id).first()
            if person:
                candidates.append({
                    'student_id': student_id,
                    'name': data.get('name', ''),
                    'person_id': person.id,
                    'department': person.department,
                    'role': person.role,
                    'match_score': float(max_score)
                })
            if max_score > best_score:
                best_score = max_score
                best_match = student_id

        candidates.sort(key=lambda x: x['match_score'], reverse=True)
        matched = best_match is not None and best_score >= match_threshold

        result = {
            'success': True,
            'detected_faces': len(faces),
            'matched': matched,
            'threshold': match_threshold,
            'top_score': float(best_score) if best_score > -1 else 0.0,
            'candidates': candidates[:10]
        }

        if not matched:
            if len(candidates) > 0:
                top_name = candidates[0]['name']
                top_pct = candidates[0]['match_score'] * 100
                result['message'] = f'最佳匹配: {top_name} (相似度 {top_pct:.1f}%), 低于阈值 {match_threshold*100:.0f}%'
            else:
                result['message'] = '数据库中无可匹配的人员'

        return jsonify(result)
    except Exception as e:
        logger.error(f"match_face: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/face/recognize', methods=['POST'])
@login_required
def recognize_face():
    try:
        frame = decode_upload(request)
        if frame is None:
            return jsonify({'success': False, 'error': '未提供有效图片'}), 400

        camera_id = request.args.get('camera_id') or request.form.get('camera_id')

        faces = face_service.detect_faces_robust(frame)

        if not faces:
            return jsonify({
                'success': True,
                'recognized': False,
                'reason': 'no_face',
                'message': '未检测到人脸',
                'confidence': 0.0
            })

        face_info = faces[0]
        feature = _get_feature(face_info, frame)
        matched_id, score = face_service.match_feature(feature)

        if matched_id:
            person = Person.query.filter_by(student_id=matched_id).first()
            if person:
                if not _check_permission(person, camera_id):
                    access_log = AccessLog(
                        person_id=person.id, name=person.name,
                        student_id=person.student_id, result='deny',
                        confidence=float(score),
                        camera_id=camera_id
                    )
                    db.session.add(access_log)
                    db.session.commit()
                    _safe_emit('access_log', access_log.to_dict())

                    cam_name = _get_camera_name(camera_id)
                    abnormal = AbnormalRecord(
                        type='权限不足',
                        description=f'{person.name} 无 {cam_name} 通行权限'
                    )
                    db.session.add(abnormal)
                    db.session.commit()
                    _safe_emit('abnormal_event', abnormal.to_dict())

                    return jsonify({
                        'success': True, 'recognized': False,
                        'reason': 'permission_denied',
                        'person': person.to_dict(),
                        'confidence': float(score),
                        'message': f'{person.name} 无{cam_name}区域通行权限'
                    })

                access_log = AccessLog(
                    person_id=person.id, name=person.name,
                    student_id=person.student_id, result='success',
                    confidence=float(score),
                    camera_id=camera_id
                )
                db.session.add(access_log)
                db.session.commit()
                _safe_emit('access_log', access_log.to_dict())

                logger.info(f"Recognized: {person.name} ({person.student_id}), score={score:.3f}")
                return jsonify({
                    'success': True, 'recognized': True,
                    'person': person.to_dict(),
                    'confidence': float(score),
                    'message': '识别成功'
                })

        abnormal = AbnormalRecord(
            type='未注册用户',
            description='检测到未注册人脸'
        )
        db.session.add(abnormal)
        db.session.commit()
        _safe_emit('abnormal_event', abnormal.to_dict())

        return jsonify({
            'success': True, 'recognized': False,
            'reason': 'not_found',
            'message': '未在数据库中找到匹配人员',
            'confidence': float(score) if score else 0.0
        })
    except Exception as e:
        logger.error(f"recognize_face: {e}\n{traceback.format_exc()}")
        return jsonify({
            'success': False, 'error': str(e),
            'recognized': False, 'message': '处理出错', 'confidence': 0.0
        }), 500


@app.route('/api/access-logs', methods=['GET'])
@login_required
def get_access_logs():
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        person_id = request.args.get('person_id')

        query = AccessLog.query
        if start_date:
            query = query.filter(AccessLog.access_time >= start_date)
        if end_date:
            query = query.filter(AccessLog.access_time <= end_date + ' 23:59:59')
        if person_id:
            query = query.filter(AccessLog.person_id == person_id)

        logs = query.order_by(AccessLog.access_time.desc()).limit(500).all()
        return jsonify({'success': True, 'data': [log.to_dict() for log in logs]})
    except Exception as e:
        logger.error(f"get_access_logs: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/access-logs/stats', methods=['GET'])
@login_required
def get_access_stats():
    try:
        today = _now().date()
        start_of_day = datetime.combine(today, datetime.min.time()).replace(tzinfo=_tz_cn)
        end_of_day = datetime.combine(today, datetime.max.time()).replace(tzinfo=_tz_cn)

        total_today = AccessLog.query.filter(
            AccessLog.access_time >= start_of_day,
            AccessLog.access_time <= end_of_day
        ).count()

        success_today = AccessLog.query.filter(
            AccessLog.access_time >= start_of_day,
            AccessLog.access_time <= end_of_day,
            AccessLog.result == 'success'
        ).count()

        abnormal_today = AbnormalRecord.query.filter(
            AbnormalRecord.timestamp >= start_of_day,
            AbnormalRecord.timestamp <= end_of_day
        ).count()

        return jsonify({
            'success': True,
            'data': {
                'total_today': total_today,
                'success_today': success_today,
                'abnormal_today': abnormal_today,
                'deny_today': total_today - success_today
            }
        })
    except Exception as e:
        logger.error(f"get_access_stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/abnormal-records', methods=['GET'])
@login_required
def get_abnormal_records():
    try:
        record_type = request.args.get('type')
        status = request.args.get('status')
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        query = AbnormalRecord.query
        if record_type:
            query = query.filter(AbnormalRecord.type == record_type)
        if status:
            query = query.filter(AbnormalRecord.status == status)
        if date_from:
            query = query.filter(AbnormalRecord.timestamp >= date_from)
        if date_to:
            query = query.filter(AbnormalRecord.timestamp <= date_to + ' 23:59:59')
        records = query.order_by(AbnormalRecord.timestamp.desc()).limit(200).all()
        return jsonify({'success': True, 'data': [r.to_dict() for r in records]})
    except Exception as e:
        logger.error(f"get_abnormal_records: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/abnormal-records', methods=['DELETE'])
@login_required
def clear_abnormal_records():
    try:
        AbnormalRecord.query.filter(AbnormalRecord.status != 'processed').delete()
        db.session.commit()
        return jsonify({'success': True, 'message': '未处理记录已清空'})
    except Exception as e:
        db.session.rollback()
        logger.error(f"clear_abnormal_records: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/abnormal-records/<int:record_id>', methods=['PUT'])
@login_required
def update_abnormal_record(record_id):
    try:
        record = AbnormalRecord.query.get(record_id)
        if not record:
            return jsonify({'success': False, 'error': '记录不存在'}), 404
        data = request.json
        if 'status' in data:
            record.status = data['status']
        db.session.commit()
        return jsonify({'success': True, 'data': record.to_dict()})
    except Exception as e:
        db.session.rollback()
        logger.error(f"update_abnormal_record: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/permissions', methods=['GET'])
@login_required
def get_permissions():
    try:
        person_id = request.args.get('person_id')
        query = Permission.query
        if person_id:
            query = query.filter(Permission.person_id == person_id)
        permissions = query.all()
        return jsonify({'success': True, 'data': [p.to_dict() for p in permissions]})
    except Exception as e:
        logger.error(f"get_permissions: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/permissions', methods=['POST'])
@login_required
def create_permission():
    try:
        data = request.json
        person_id = data.get('person_id')
        areas = data.get('areas')
        area = data.get('area')
        
        if not person_id:
            return jsonify({'success': False, 'error': '人员不能为空'}), 400
        
        if not areas and area:
            areas = [area]
        
        if not areas and not area:
            return jsonify({'success': False, 'error': '请选择通行区域'}), 400
        
        final_areas = []
        for a in areas:
            if a == '__ALL__':
                all_cams = Camera.query.all()
                for cam in all_cams:
                    final_areas.append(cam.camera_id)
            else:
                final_areas.append(a)
        
        if not final_areas:
            return jsonify({'success': False, 'error': '没有有效的通行区域'}), 400
        
        created = 0
        for a in final_areas:
            existing = Permission.query.filter_by(person_id=person_id, area=a).first()
            if existing:
                existing.start_time = data.get('start_time')
                existing.end_time = data.get('end_time')
                if data.get('valid_from'):
                    existing.valid_from = datetime.fromisoformat(data['valid_from']).date()
                if data.get('valid_to'):
                    existing.valid_to = datetime.fromisoformat(data['valid_to']).date()
            else:
                permission = Permission(
                    person_id=person_id, area=a,
                    start_time=data.get('start_time'),
                    end_time=data.get('end_time'),
                    valid_from=datetime.fromisoformat(data['valid_from']).date() if data.get('valid_from') else None,
                    valid_to=datetime.fromisoformat(data['valid_to']).date() if data.get('valid_to') else None
                )
                db.session.add(permission)
            created += 1
        
        db.session.commit()
        return jsonify({'success': True, 'count': created, 'message': f'已为 {created} 个区域设置权限'})
    except Exception as e:
        db.session.rollback()
        logger.error(f"create_permission: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/permissions/<int:perm_id>', methods=['DELETE'])
@login_required
def delete_permission(perm_id):
    try:
        perm = Permission.query.get(perm_id)
        if not perm:
            return jsonify({'success': False, 'error': '权限不存在'}), 404
        db.session.delete(perm)
        db.session.commit()
        return jsonify({'success': True, 'message': '权限已删除'})
    except Exception as e:
        db.session.rollback()
        logger.error(f"delete_permission: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/cameras', methods=['GET'])
@login_required
def get_cameras():
    try:
        cameras = Camera.query.all()
        return jsonify({'success': True, 'data': [c.to_dict() for c in cameras]})
    except Exception as e:
        logger.error(f"get_cameras: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/persons/import', methods=['POST'])
@login_required
def import_persons():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '未提供文件'}), 400
        file = request.files['file']
        content = file.read().decode('utf-8')
        lines = [l.strip() for l in content.strip().split('\n') if l.strip()]
        imported, skipped = 0, 0
        for line in lines[1:]:
            parts = line.split(',')
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            student_id = parts[1].strip()
            department = parts[2].strip() if len(parts) > 2 else ''
            role = parts[3].strip() if len(parts) > 3 else 'student'
            if Person.query.filter_by(student_id=student_id).first():
                skipped += 1
                continue
            person = Person(name=name, student_id=student_id, department=department, role=role)
            db.session.add(person)
            imported += 1
        db.session.commit()
        logger.info(f"Import: {imported} imported, {skipped} skipped")
        return jsonify({'success': True, 'message': f'成功导入 {imported} 人，跳过 {skipped} 人'})
    except Exception as e:
        db.session.rollback()
        logger.error(f"import_persons: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/export/csv', methods=['GET'])
@login_required
def export_csv():
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        query = AccessLog.query
        if start_date:
            query = query.filter(AccessLog.access_time >= start_date)
        if end_date:
            query = query.filter(AccessLog.access_time <= end_date + ' 23:59:59')
        logs = query.order_by(AccessLog.access_time.desc()).all()
        csv_lines = ["ID,人员ID,姓名,学号,通行时间,结果,置信度"]
        for log in logs:
            csv_lines.append(f"{log.id},{log.person_id or ''},{log.name or ''},{log.student_id or ''},{log.access_time},{log.result},{log.confidence or ''}")
        csv_content = '\n'.join(csv_lines)
        return csv_content, 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': 'attachment; filename=access_logs.csv'
        }
    except Exception as e:
        logger.error(f"export_csv: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    emit('connected', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')


@socketio.on('video_frame')
def handle_video_frame(data):
    try:
        if not session.get('user_id'):
            emit('error', {'error': '未登录'})
            return
        frame_data = data.get('frame')
        if not frame_data:
            return
        frame = decode_image(frame_data)
        if frame is None:
            return

        camera_id = data.get('camera_id')

        faces = face_service.detect_faces_robust(frame)
        result = {'face_count': len(faces), 'faces': []}

        for face_info in faces:
            box = face_info['box']
            box_list = [int(b) for b in box]
            face_result = {
                'box': box_list,
                'confidence': float(face_info['confidence']),
                'liveness': True,
                'liveness_results': [],
                'recognized': False
            }

            try:
                feature = _get_feature(face_info, frame)
                matched_id, score = face_service.match_feature(feature)

                if matched_id:
                    person = Person.query.filter_by(student_id=matched_id).first()
                    if person:
                        face_result['recognized'] = True
                        face_result['person'] = person.to_dict()
                        face_result['match_score'] = float(score)

                        if not _check_permission(person, camera_id):
                            face_result['recognized'] = False
                            face_result['reason'] = 'permission_denied'
                            face_result['person'] = person.to_dict()
                            face_result['match_score'] = float(score)
                            if camera_id:
                                cam_name = _get_camera_name(camera_id)
                                face_result['deny_detail'] = f'{person.name} 无 {cam_name} 区域通行权限'
                            
                            with _last_record_lock:
                                deny_key = f'deny_{matched_id}'
                                last = _last_record_time.get(deny_key)
                                now = _now()
                                if last is None or (now - last).seconds >= MIN_RECORD_INTERVAL:
                                    _last_record_time[deny_key] = now
                                    
                                    access_log = AccessLog(
                                        person_id=person.id, name=person.name,
                                        student_id=person.student_id,
                                        result='deny', confidence=float(score),
                                        camera_id=camera_id
                                    )
                                    db.session.add(access_log)
                                    db.session.commit()
                                    _safe_emit('access_log', access_log.to_dict())
                                    
                                    cam_name = _get_camera_name(camera_id)
                                    abnormal = AbnormalRecord(
                                        type='权限不足',
                                        description=f'{person.name} 无 {cam_name} 通行权限'
                                    )
                                    db.session.add(abnormal)
                                    db.session.commit()
                                    _safe_emit('abnormal_event', abnormal.to_dict())
                        else:
                            with _last_record_lock:
                                last = _last_record_time.get(matched_id)
                                now = _now()
                                if last is None or (now - last).seconds >= MIN_RECORD_INTERVAL:
                                    _last_record_time[matched_id] = now

                                    access_log = AccessLog(
                                        person_id=person.id, name=person.name,
                                        student_id=person.student_id,
                                        result='success', confidence=float(score),
                                        camera_id=camera_id
                                    )
                                    db.session.add(access_log)
                                    db.session.commit()
                                    _safe_emit('access_log', access_log.to_dict())
                else:
                    face_result['recognized'] = False
                    face_result['reason'] = 'not_found'

                    with _last_record_lock:
                        last = _last_record_time.get('__unknown__')
                        now = _now()
                        if last is None or (now - last).seconds >= MIN_RECORD_INTERVAL:
                            _last_record_time['__unknown__'] = now

                            abnormal = AbnormalRecord(
                                type='未注册用户',
                                description='实时识别检测到未注册人脸'
                            )
                            db.session.add(abnormal)
                            db.session.commit()
                            _safe_emit('abnormal_event', abnormal.to_dict())
            except Exception as e:
                logger.error(f"video_frame processing: {e}")

            result['faces'].append(face_result)

        emit('recognition_result', result)
    except Exception as e:
        logger.error(f"video_frame: {e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    logger.info("Starting face recognition server on port %d...", port)
    socketio.run(app, host='0.0.0.0', port=port, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)