import sqlite3
import os
import numpy as np
from config import DB_PATH, FEATURE_DB_PATH

class FaceDatabase:
    def __init__(self):
        self.conn = None
        self.features = {}
        self._init_db()
        self._load_features()
    
    def _init_db(self):
        self.conn = sqlite3.connect(DB_PATH)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                student_id TEXT UNIQUE NOT NULL,
                department TEXT,
                role TEXT DEFAULT 'student',
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                name TEXT,
                student_id TEXT,
                access_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                result TEXT,
                temperature REAL,
                FOREIGN KEY (person_id) REFERENCES persons(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS abnormal_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                description TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS permissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                area TEXT NOT NULL,
                start_time TEXT,
                end_time TEXT,
                valid_from DATE,
                valid_to DATE,
                FOREIGN KEY (person_id) REFERENCES persons(id)
            )
        ''')
        
        self.conn.commit()
    
    def _load_features(self):
        if os.path.exists(FEATURE_DB_PATH):
            self.features = np.load(FEATURE_DB_PATH, allow_pickle=True).item()
    
    def _save_features(self):
        np.save(FEATURE_DB_PATH, self.features)
    
    def add_person(self, name, student_id, department='', role='student'):
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO persons (name, student_id, department, role)
                VALUES (?, ?, ?, ?)
            ''', (name, student_id, department, role))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None
    
    def add_feature(self, student_id, feature):
        self.features[student_id] = feature
        self._save_features()
    
    def get_person_by_id(self, person_id):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM persons WHERE id = ?', (person_id,))
        return cursor.fetchone()
    
    def get_person_by_student_id(self, student_id):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM persons WHERE student_id = ?', (student_id,))
        return cursor.fetchone()
    
    def get_all_persons(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM persons')
        return cursor.fetchall()
    
    def add_access_log(self, person_id, name, student_id, result, temperature=None):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO access_logs (person_id, name, student_id, result, temperature)
            VALUES (?, ?, ?, ?, ?)
        ''', (person_id, name, student_id, result, temperature))
        self.conn.commit()
    
    def add_abnormal_record(self, type_, description, image_path=''):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO abnormal_records (type, description, image_path)
            VALUES (?, ?, ?)
        ''', (type_, description, image_path))
        self.conn.commit()
    
    def get_access_logs(self, start_date=None, end_date=None):
        cursor = self.conn.cursor()
        query = 'SELECT * FROM access_logs'
        params = []
        
        if start_date:
            query += ' WHERE access_time >= ?'
            params.append(start_date)
            if end_date:
                query += ' AND access_time <= ?'
                params.append(end_date)
        elif end_date:
            query += ' WHERE access_time <= ?'
            params.append(end_date)
        
        query += ' ORDER BY access_time DESC'
        cursor.execute(query, params)
        return cursor.fetchall()
    
    def get_abnormal_records(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM abnormal_records ORDER BY timestamp DESC')
        return cursor.fetchall()
    
    def close(self):
        if self.conn:
            self.conn.close()
