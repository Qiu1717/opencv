from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from werkzeug.security import generate_password_hash, check_password_hash
import os
import secrets
import base64

from config import get_logger

logger = get_logger(__name__)

_tz_cn = timezone(timedelta(hours=8))

def _now():
    return datetime.now(_tz_cn)

db = SQLAlchemy()

def _migrate_status_column():
    conn = db.engine.connect()
    try:
        result = conn.execute(text("PRAGMA table_info(abnormal_records)"))
        columns = [row[1] for row in result]
        if 'status' not in columns:
            conn.execute(text("ALTER TABLE abnormal_records ADD COLUMN status VARCHAR(50) DEFAULT 'pending'"))
            conn.commit()
    except Exception:
        pass
    finally:
        conn.close()

def _create_default_admin():
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', role='admin')
        admin.set_password(os.environ.get('ADMIN_PASSWORD') or secrets.token_urlsafe(12))
        db.session.add(admin)
        db.session.commit()
        logger.info(
            "Created default admin user 'admin'. "
            "Set ADMIN_PASSWORD env var to choose your own initial password."
        )

class Person(db.Model):
    __tablename__ = 'persons'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    student_id = db.Column(db.String(50), unique=True, nullable=False)
    department = db.Column(db.String(100))
    role = db.Column(db.String(50), default='student')
    status = db.Column(db.String(50), default='active')
    face_image = db.Column(db.LargeBinary)
    feature_vector = db.Column(db.LargeBinary)
    created_at = db.Column(db.DateTime, default=_now)
    updated_at = db.Column(db.DateTime, default=_now, onupdate=_now)
    
    access_logs = db.relationship('AccessLog', backref='person', lazy=True)
    permissions = db.relationship('Permission', backref='person', lazy=True)
    
    def to_dict(self):
        result = {
            'id': self.id,
            'name': self.name,
            'student_id': self.student_id,
            'department': self.department,
            'role': self.role,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'permissions': [p.to_dict() for p in self.permissions]
        }
        
        if self.face_image:
            result['face_photo'] = 'data:image/jpeg;base64,' + base64.b64encode(self.face_image).decode('utf-8')
        else:
            result['face_photo'] = None
            
        return result

class AccessLog(db.Model):
    __tablename__ = 'access_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.Integer, db.ForeignKey('persons.id'))
    name = db.Column(db.String(100))
    student_id = db.Column(db.String(50))
    access_time = db.Column(db.DateTime, default=_now)
    result = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    camera_id = db.Column(db.String(50))
    
    def to_dict(self):
        return {
            'id': self.id,
            'person_id': self.person_id,
            'name': self.name,
            'student_id': self.student_id,
            'access_time': self.access_time.isoformat() if self.access_time else None,
            'result': self.result,
            'confidence': self.confidence,
            'camera_id': self.camera_id
        }

class AbnormalRecord(db.Model):
    __tablename__ = 'abnormal_records'
    
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=_now)
    image_path = db.Column(db.String(500))
    status = db.Column(db.String(50), default='pending')
    
    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'description': self.description,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'image_path': self.image_path,
            'status': self.status
        }

class Permission(db.Model):
    __tablename__ = 'permissions'
    
    id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.Integer, db.ForeignKey('persons.id'))
    area = db.Column(db.String(100), nullable=False)
    start_time = db.Column(db.String(20))
    end_time = db.Column(db.String(20))
    valid_from = db.Column(db.Date)
    valid_to = db.Column(db.Date)
    created_at = db.Column(db.DateTime, default=_now)
    
    def to_dict(self):
        return {
            'id': self.id,
            'person_id': self.person_id,
            'area': self.area,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'valid_from': self.valid_from.isoformat() if self.valid_from else None,
            'valid_to': self.valid_to.isoformat() if self.valid_to else None
        }

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), default='admin')
    created_at = db.Column(db.DateTime, default=_now)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'role': self.role,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Camera(db.Model):
    __tablename__ = 'cameras'
    
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(100))
    location = db.Column(db.String(200))
    status = db.Column(db.String(50), default='active')
    last_frame_time = db.Column(db.DateTime)
    
    def to_dict(self):
        return {
            'id': self.id,
            'camera_id': self.camera_id,
            'name': self.name,
            'location': self.location,
            'status': self.status,
            'last_frame_time': self.last_frame_time.isoformat() if self.last_frame_time else None
        }

def init_db(app):
    basedir = os.path.abspath(os.path.dirname(__file__))
    app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(basedir, 'face_system.db')}"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()
        _migrate_status_column()
        _create_default_admin()
        _seed_default_cameras()

def _seed_default_cameras():
    default_cameras = [
        ('east_gate',    '学校东门', '学校东门主入口'),
        ('south_gate',   '学校南门', '学校南门主入口'),
        ('north_gate',   '学校北门', '学校北门主入口'),
        ('library',      '图书馆',   '图书馆正门'),
        ('dorm_1',       '宿舍楼1',  '宿舍楼1入口'),
        ('dorm_2',       '宿舍楼2',  '宿舍楼2入口'),
        ('lab',          '实验楼',   '实验楼大厅'),
        ('art',          '艺术楼',   '艺术楼正门'),
        ('chemical',     '化工楼',   '化工楼入口'),
    ]
    valid_ids = {cid for cid, _, _ in default_cameras}
    for c in Camera.query.all():
        if c.camera_id not in valid_ids:
            db.session.delete(c)
    db.session.flush()
    for camera_id, name, location in default_cameras:
        existing = Camera.query.filter_by(camera_id=camera_id).first()
        if existing:
            existing.name = name
            existing.location = location
        else:
            db.session.add(Camera(camera_id=camera_id, name=name, location=location))
    db.session.commit()