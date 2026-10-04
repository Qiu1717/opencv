import sys
import cv2
import numpy as np
import time
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QTabWidget, QTableWidget, QTableWidgetItem,
                             QDateEdit, QMessageBox, QFileDialog, QTextEdit,
                             QGroupBox, QGridLayout, QProgressBar)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import QTimer, QDate, Qt, pyqtSignal, QThread

from database import FaceDatabase
from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from liveness_detection import LivenessDetector
from tailgating_detection import TailgatingDetector

class VideoThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)
    fps_signal = pyqtSignal(int)
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.capture = cv2.VideoCapture(camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True
        self.fps_count = 0
        self.fps_start = time.time()
    
    def run(self):
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                self.frame_signal.emit(frame)
                self.fps_count += 1
                elapsed = time.time() - self.fps_start
                if elapsed >= 1.0:
                    self.fps_signal.emit(self.fps_count)
                    self.fps_count = 0
                    self.fps_start = time.time()
    
    def stop(self):
        self.running = False
        self.wait()
        self.capture.release()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能校园人脸识别门禁系统")
        self.setGeometry(100, 100, 1300, 850)
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f5f5; }
            QTabWidget::pane { border: 1px solid #ccc; background: white; }
            QTabBar::tab { background: #e0e0e0; padding: 8px 16px; margin-right: 2px; }
            QTabBar::tab:selected { background: white; border-bottom: 2px solid #1e90ff; }
            QPushButton { background: #1e90ff; color: white; border: none; padding: 8px 16px; border-radius: 4px; }
            QPushButton:hover { background: #4169e1; }
            QGroupBox { border: 1px solid #ccc; border-radius: 8px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        """)
        
        self.db = FaceDatabase()
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.liveness_detector = LivenessDetector()
        self.tailgating_detector = TailgatingDetector()
        
        self.prev_face = None
        self.recognized_names = []
        self.last_recognize_time = {}
        self.recognition_interval = 2.0
        
        self.setup_ui()
    
    def setup_ui(self):
        self.tab_widget = QTabWidget()
        
        self.register_tab = self.create_register_tab()
        self.recognize_tab = self.create_recognize_tab()
        self.alarm_tab = self.create_alarm_tab()
        self.report_tab = self.create_report_tab()
        self.permission_tab = self.create_permission_tab()
        
        self.tab_widget.addTab(self.register_tab, "人脸注册")
        self.tab_widget.addTab(self.recognize_tab, "实时识别")
        self.tab_widget.addTab(self.alarm_tab, "异常告警")
        self.tab_widget.addTab(self.report_tab, "统计报表")
        self.tab_widget.addTab(self.permission_tab, "权限管理")
        
        self.setCentralWidget(self.tab_widget)
    
    def create_register_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        form_layout = QHBoxLayout()
        
        left_layout = QVBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("姓名")
        left_layout.addWidget(QLabel("姓名:"))
        left_layout.addWidget(self.name_input)
        
        self.student_id_input = QLineEdit()
        self.student_id_input.setPlaceholderText("学号/工号")
        left_layout.addWidget(QLabel("学号/工号:"))
        left_layout.addWidget(self.student_id_input)
        
        self.department_input = QLineEdit()
        self.department_input.setPlaceholderText("院系")
        left_layout.addWidget(QLabel("院系:"))
        left_layout.addWidget(self.department_input)
        
        self.role_input = QLineEdit()
        self.role_input.setPlaceholderText("角色: student/teacher/staff")
        left_layout.addWidget(QLabel("角色:"))
        left_layout.addWidget(self.role_input)
        
        self.register_btn = QPushButton("注册人脸")
        self.register_btn.clicked.connect(self.register_face)
        left_layout.addWidget(self.register_btn)
        
        self.register_status = QLabel("")
        left_layout.addWidget(self.register_status)
        
        right_layout = QVBoxLayout()
        self.register_image_label = QLabel()
        self.register_image_label.setFixedSize(320, 240)
        self.register_image_label.setStyleSheet("border: 1px solid gray")
        right_layout.addWidget(self.register_image_label)
        
        self.capture_btn = QPushButton("拍照")
        self.capture_btn.clicked.connect(self.capture_face)
        right_layout.addWidget(self.capture_btn)
        
        self.upload_btn = QPushButton("上传照片")
        self.upload_btn.clicked.connect(self.upload_photo)
        right_layout.addWidget(self.upload_btn)
        
        form_layout.addLayout(left_layout)
        form_layout.addLayout(right_layout)
        layout.addLayout(form_layout)
        
        widget.setLayout(layout)
        return widget
    
    def capture_face(self):
        capture = cv2.VideoCapture(0)
        ret, frame = capture.read()
        capture.release()
        
        if ret:
            self.captured_frame = frame
            self.display_image(frame, self.register_image_label)
    
    def upload_photo(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择照片", "", "Image Files (*.jpg *.jpeg *.png)")
        if file_path:
            frame = cv2.imread(file_path)
            self.captured_frame = frame
            self.display_image(frame, self.register_image_label)
    
    def register_face(self):
        name = self.name_input.text()
        student_id = self.student_id_input.text()
        department = self.department_input.text()
        role = self.role_input.text() or "student"
        
        if not name or not student_id:
            self.register_status.setText("请填写姓名和学号")
            return
        
        if not hasattr(self, 'captured_frame'):
            self.register_status.setText("请先拍照或上传照片")
            return
        
        faces = self.face_detector.detect_faces(self.captured_frame)
        if not faces:
            self.register_status.setText("未检测到人脸")
            return
        
        face = self.face_detector.get_face_roi(self.captured_frame, faces[0]['box'])
        feature = self.face_recognizer.extract_feature(face)
        
        person_id = self.db.add_person(name, student_id, department, role)
        if person_id:
            self.db.add_feature(student_id, feature)
            self.register_status.setText(f"注册成功! ID: {person_id}")
        else:
            self.register_status.setText("学号已存在")
    
    def create_recognize_tab(self):
        widget = QWidget()
        layout = QHBoxLayout()
        
        left_layout = QVBoxLayout()
        
        video_group = QGroupBox("实时监控画面")
        video_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #1e90ff; background: black;")
        video_layout.addWidget(self.video_label)
        
        status_bar = QHBoxLayout()
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("color: red; font-size: 20px;")
        self.status_text = QLabel("等待启动")
        self.fps_label = QLabel("FPS: --")
        status_bar.addWidget(self.status_indicator)
        status_bar.addWidget(self.status_text)
        status_bar.addStretch()
        status_bar.addWidget(self.fps_label)
        video_layout.addLayout(status_bar)
        video_group.setLayout(video_layout)
        left_layout.addWidget(video_group)
        
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始识别")
        self.start_btn.clicked.connect(self.start_recognition)
        self.stop_btn = QPushButton("停止识别")
        self.stop_btn.clicked.connect(self.stop_recognition)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        left_layout.addLayout(control_layout)
        
        layout.addLayout(left_layout)
        
        right_layout = QVBoxLayout()
        
        result_group = QGroupBox("识别结果")
        result_layout = QVBoxLayout()
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(4)
        self.result_table.setHorizontalHeaderLabels(["姓名", "学号/工号", "通行时间", "状态"])
        self.result_table.setRowCount(5)
        result_layout.addWidget(self.result_table)
        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)
        
        stats_group = QGroupBox("今日通行统计")
        stats_layout = QGridLayout()
        
        self.total_count_label = QLabel("1,245")
        self.total_count_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.total_count_label.setStyleSheet("color: #333;")
        stats_layout.addWidget(QLabel("总通行人次"), 0, 0)
        stats_layout.addWidget(self.total_count_label, 1, 0)
        
        self.success_count_label = QLabel("1,187")
        self.success_count_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.success_count_label.setStyleSheet("color: #228b22;")
        stats_layout.addWidget(QLabel("正常通行"), 0, 1)
        stats_layout.addWidget(self.success_count_label, 1, 1)
        
        self.abnormal_count_label = QLabel("42")
        self.abnormal_count_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.abnormal_count_label.setStyleSheet("color: #ff8c00;")
        stats_layout.addWidget(QLabel("异常事件"), 0, 2)
        stats_layout.addWidget(self.abnormal_count_label, 1, 2)
        
        self.deny_count_label = QLabel("16")
        self.deny_count_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.deny_count_label.setStyleSheet("color: #dc143c;")
        stats_layout.addWidget(QLabel("禁止通行"), 0, 3)
        stats_layout.addWidget(self.deny_count_label, 1, 3)
        
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)
        
        layout.addLayout(right_layout)
        
        widget.setLayout(layout)
        return widget
    
    def start_recognition(self):
        self.video_thread = VideoThread()
        self.video_thread.frame_signal.connect(self.process_frame)
        self.video_thread.fps_signal.connect(self.update_fps)
        self.video_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_indicator.setStyleSheet("color: #228b22; font-size: 20px;")
        self.status_text.setText("识别中")
        self.update_today_stats()
    
    def stop_recognition(self):
        if hasattr(self, 'video_thread'):
            self.video_thread.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_indicator.setStyleSheet("color: red; font-size: 20px;")
        self.status_text.setText("已停止")
    
    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps}")
    
    def update_today_stats(self):
        today = datetime.now().strftime("%Y-%m-%d")
        start_date = f"{today} 00:00:00"
        end_date = f"{today} 23:59:59"
        
        logs = self.db.get_access_logs(start_date, end_date)
        total = len(logs)
        success = sum(1 for log in logs if log[5] == 'success')
        abnormal = len(self.db.get_abnormal_records())
        deny = total - success
        
        self.total_count_label.setText(f"{total:,}")
        self.success_count_label.setText(f"{success:,}")
        self.abnormal_count_label.setText(f"{abnormal:,}")
        self.deny_count_label.setText(f"{deny:,}")
    
    def add_recognize_result(self, name, student_id, status):
        current_time = datetime.now().strftime("%H:%M:%S")
        
        for i in range(self.result_table.rowCount()):
            if self.result_table.item(i, 0) is None or self.result_table.item(i, 0).text() == "":
                self.result_table.setItem(i, 0, QTableWidgetItem(name))
                self.result_table.setItem(i, 1, QTableWidgetItem(student_id))
                self.result_table.setItem(i, 2, QTableWidgetItem(current_time))
                self.result_table.setItem(i, 3, QTableWidgetItem(status))
                return
        
        for i in range(self.result_table.rowCount() - 1):
            for j in range(4):
                item = self.result_table.item(i + 1, j)
                if item:
                    self.result_table.setItem(i, j, QTableWidgetItem(item.text()))
                else:
                    self.result_table.setItem(i, j, QTableWidgetItem(""))
        
        self.result_table.setItem(self.result_table.rowCount() - 1, 0, QTableWidgetItem(name))
        self.result_table.setItem(self.result_table.rowCount() - 1, 1, QTableWidgetItem(student_id))
        self.result_table.setItem(self.result_table.rowCount() - 1, 2, QTableWidgetItem(current_time))
        self.result_table.setItem(self.result_table.rowCount() - 1, 3, QTableWidgetItem(status))
    
    def process_frame(self, frame):
        current_time = time.time()
        frame = self.face_detector.apply_clahe(frame)
        faces = self.face_detector.detect_faces(frame)
        
        for face_info in faces:
            box = face_info['box']
            face = self.face_detector.get_face_roi(frame, box)
            
            feature = self.face_recognizer.extract_feature(face)
            student_id, score = self.face_recognizer.match_feature(feature, self.db.features)
                
            if student_id:
                person = self.db.get_person_by_student_id(student_id)
                if person:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"{person[1]} {score:.2f}", 
                               (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    if student_id not in self.last_recognize_time or \
                       current_time - self.last_recognize_time[student_id] > self.recognition_interval:
                        self.db.add_access_log(person[0], person[1], student_id, "success")
                        self.add_recognize_result(person[1], student_id, "允许通行")
                        self.last_recognize_time[student_id] = current_time
                        self.update_today_stats()
            else:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                cv2.putText(frame, "未注册", (box[0], box[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                if "unknown" not in self.last_recognize_time or \
                   current_time - self.last_recognize_time["unknown"] > self.recognition_interval:
                    self.db.add_abnormal_record("未注册用户", "检测到未注册人脸")
                    self.add_recognize_result("未知人员", "-", "禁止通行")
                    self.last_recognize_time["unknown"] = current_time
        
        is_tailgating, _ = self.tailgating_detector.check_tailgating(frame)
        if is_tailgating:
            cv2.putText(frame, "尾随检测!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if "tailgating" not in self.last_recognize_time or \
               current_time - self.last_recognize_time["tailgating"] > self.recognition_interval:
                self.db.add_abnormal_record("尾随行为", "检测到尾随")
                self.add_recognize_result("未知人员", "-", "尾随告警")
                self.last_recognize_time["tailgating"] = current_time
                self.update_today_stats()
        
        self.display_image(frame, self.video_label)
    
    def create_alarm_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        filter_layout = QHBoxLayout()
        self.alarm_type_filter = QLineEdit()
        self.alarm_type_filter.setPlaceholderText("按类型筛选...")
        filter_layout.addWidget(QLabel("筛选:"))
        filter_layout.addWidget(self.alarm_type_filter)
        
        self.refresh_alarm_btn = QPushButton("刷新告警记录")
        self.refresh_alarm_btn.clicked.connect(self.refresh_alarm_table)
        filter_layout.addWidget(self.refresh_alarm_btn)
        
        self.clear_alarm_btn = QPushButton("清空记录")
        self.clear_alarm_btn.clicked.connect(self.clear_alarm_records)
        filter_layout.addWidget(self.clear_alarm_btn)
        
        filter_layout.addStretch()
        layout.addLayout(filter_layout)
        
        self.alarm_table = QTableWidget()
        self.alarm_table.setColumnCount(4)
        self.alarm_table.setHorizontalHeaderLabels(["类型", "描述", "时间", "图片路径"])
        self.alarm_table.setStyleSheet("QTableWidget { gridline-color: #ddd; }")
        layout.addWidget(self.alarm_table)
        
        self.alarm_summary = QGroupBox("告警统计")
        summary_layout = QHBoxLayout()
        
        self.unknown_count = QLabel("未注册用户: 0")
        self.unknown_count.setStyleSheet("color: #dc143c; font-weight: bold;")
        summary_layout.addWidget(self.unknown_count)
        
        self.liveness_count = QLabel("活体检测失败: 0")
        self.liveness_count.setStyleSheet("color: #ff8c00; font-weight: bold;")
        summary_layout.addWidget(self.liveness_count)
        
        self.tailgating_count = QLabel("尾随行为: 0")
        self.tailgating_count.setStyleSheet("color: #9370db; font-weight: bold;")
        summary_layout.addWidget(self.tailgating_count)
        
        self.alarm_summary.setLayout(summary_layout)
        layout.addWidget(self.alarm_summary)
        
        widget.setLayout(layout)
        return widget
    
    def refresh_alarm_table(self):
        records = self.db.get_abnormal_records()
        
        filter_text = self.alarm_type_filter.text().lower()
        if filter_text:
            records = [r for r in records if filter_text in r[1].lower()]
        
        self.alarm_table.setRowCount(len(records))
        
        unknown = 0
        liveness = 0
        tailgating = 0
        
        for i, record in enumerate(records):
            self.alarm_table.setItem(i, 0, QTableWidgetItem(record[1]))
            self.alarm_table.setItem(i, 1, QTableWidgetItem(record[2]))
            self.alarm_table.setItem(i, 2, QTableWidgetItem(record[3]))
            self.alarm_table.setItem(i, 3, QTableWidgetItem(record[4]))
            
            if "未注册" in record[1]:
                unknown += 1
            elif "活体检测" in record[1]:
                liveness += 1
            elif "尾随" in record[1]:
                tailgating += 1
        
        self.unknown_count.setText(f"未注册用户: {unknown}")
        self.liveness_count.setText(f"活体检测失败: {liveness}")
        self.tailgating_count.setText(f"尾随行为: {tailgating}")
    
    def clear_alarm_records(self):
        reply = QMessageBox.question(self, "确认清空", "确定要清空所有告警记录吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            cursor = self.db.conn.cursor()
            cursor.execute('DELETE FROM abnormal_records')
            self.db.conn.commit()
            self.refresh_alarm_table()
            QMessageBox.information(self, "操作成功", "告警记录已清空")
    
    def create_report_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        date_layout = QHBoxLayout()
        self.start_date = QDateEdit(QDate.currentDate())
        self.end_date = QDateEdit(QDate.currentDate())
        date_layout.addWidget(QLabel("开始日期:"))
        date_layout.addWidget(self.start_date)
        date_layout.addWidget(QLabel("结束日期:"))
        date_layout.addWidget(self.end_date)
        
        self.time_filter = QLineEdit()
        self.time_filter.setPlaceholderText("迟到时间阈值(如: 08:30)")
        date_layout.addWidget(QLabel("迟到阈值:"))
        date_layout.addWidget(self.time_filter)
        
        self.generate_btn = QPushButton("生成报表")
        self.generate_btn.clicked.connect(self.generate_report)
        date_layout.addWidget(self.generate_btn)
        
        layout.addLayout(date_layout)
        
        stats_group = QGroupBox("统计概览")
        stats_layout = QGridLayout()
        
        self.total_stat = QLabel("0")
        self.total_stat.setFont(QFont("Arial", 28, QFont.Bold))
        self.total_stat.setStyleSheet("color: #333;")
        stats_layout.addWidget(QLabel("总通行人次"), 0, 0)
        stats_layout.addWidget(self.total_stat, 1, 0)
        
        self.success_stat = QLabel("0")
        self.success_stat.setFont(QFont("Arial", 28, QFont.Bold))
        self.success_stat.setStyleSheet("color: #228b22;")
        stats_layout.addWidget(QLabel("成功通行"), 0, 1)
        stats_layout.addWidget(self.success_stat, 1, 1)
        
        self.fail_stat = QLabel("0")
        self.fail_stat.setFont(QFont("Arial", 28, QFont.Bold))
        self.fail_stat.setStyleSheet("color: #dc143c;")
        stats_layout.addWidget(QLabel("失败次数"), 0, 2)
        stats_layout.addWidget(self.fail_stat, 1, 2)
        
        self.late_rate_stat = QLabel("0%")
        self.late_rate_stat.setFont(QFont("Arial", 28, QFont.Bold))
        self.late_rate_stat.setStyleSheet("color: #ff8c00;")
        stats_layout.addWidget(QLabel("迟到率"), 0, 3)
        stats_layout.addWidget(self.late_rate_stat, 1, 3)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        layout.addWidget(self.report_text)
        
        button_layout = QHBoxLayout()
        self.export_btn = QPushButton("导出CSV")
        self.export_btn.clicked.connect(self.export_csv)
        button_layout.addWidget(self.export_btn)
        
        self.export_report_btn = QPushButton("导出报表")
        self.export_report_btn.clicked.connect(self.export_report)
        button_layout.addWidget(self.export_report_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        widget.setLayout(layout)
        return widget
    
    def generate_report(self):
        start_date = self.start_date.date().toString("yyyy-MM-dd 00:00:00")
        end_date = self.end_date.date().toString("yyyy-MM-dd 23:59:59")
        
        logs = self.db.get_access_logs(start_date, end_date)
        total_count = len(logs)
        success_count = sum(1 for log in logs if log[5] == 'success')
        fail_count = total_count - success_count
        
        late_threshold = self.time_filter.text() or "08:30"
        late_count = 0
        for log in logs:
            access_time = log[4]
            if ' ' in access_time:
                time_part = access_time.split(' ')[1]
                if time_part > late_threshold:
                    late_count += 1
        
        late_rate = (late_count / total_count * 100) if total_count > 0 else 0
        pass_rate = (success_count / total_count * 100) if total_count > 0 else 0
        
        self.total_stat.setText(f"{total_count:,}")
        self.success_stat.setText(f"{success_count:,}")
        self.fail_stat.setText(f"{fail_count:,}")
        self.late_rate_stat.setText(f"{late_rate:.1f}%")
        
        report = f"========== 门禁通行报表 ==========\n\n"
        report += f"统计时段: {start_date} 至 {end_date}\n"
        report += f"迟到时间阈值: {late_threshold}\n"
        report += "-----------------------------------\n"
        report += f"总通行人次: {total_count:,}\n"
        report += f"成功通行: {success_count:,} ({pass_rate:.2f}%)\n"
        report += f"失败次数: {fail_count:,}\n"
        report += f"迟到人数: {late_count:,} ({late_rate:.2f}%)\n"
        report += "-----------------------------------\n\n"
        report += "详细记录:\n"
        report += "-" * 60 + "\n"
        report += f"{'时间':<20} {'姓名':<10} {'学号':<12} {'状态':<8}\n"
        report += "-" * 60 + "\n"
        
        for log in logs:
            status = "成功" if log[5] == 'success' else "失败"
            report += f"{log[4]:<20} {log[2]:<10} {log[3]:<12} {status:<8}\n"
        
        self.report_text.setPlainText(report)
    
    def export_csv(self):
        logs = self.db.get_access_logs()
        csv_content = "ID,人员ID,姓名,学号,通行时间,结果,体温\n"
        
        for log in logs:
            csv_content += f"{log[0]},{log[1]},{log[2]},{log[3]},{log[4]},{log[5]},{log[6] or ''}\n"
        
        file_path, _ = QFileDialog.getSaveFileName(self, "保存CSV", "", "CSV Files (*.csv)")
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(csv_content)
            QMessageBox.information(self, "导出成功", "CSV文件已导出")
    
    def export_report(self):
        start_date = self.start_date.date().toString("yyyy-MM-dd 00:00:00")
        end_date = self.end_date.date().toString("yyyy-MM-dd 23:59:59")
        
        logs = self.db.get_access_logs(start_date, end_date)
        total_count = len(logs)
        success_count = sum(1 for log in logs if log[5] == 'success')
        fail_count = total_count - success_count
        
        late_threshold = self.time_filter.text() or "08:30"
        late_count = sum(1 for log in logs if ' ' in log[4] and log[4].split(' ')[1] > late_threshold)
        
        late_rate = (late_count / total_count * 100) if total_count > 0 else 0
        pass_rate = (success_count / total_count * 100) if total_count > 0 else 0
        
        report = f"========== 智能校园门禁系统 - 通行报表 ==========\n\n"
        report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"统计时段: {start_date} 至 {end_date}\n"
        report += f"迟到时间阈值: {late_threshold}\n\n"
        report += "【统计摘要】\n"
        report += f"  ├─ 总通行人次: {total_count:,}\n"
        report += f"  ├─ 成功通行: {success_count:,} ({pass_rate:.2f}%)\n"
        report += f"  ├─ 失败次数: {fail_count:,}\n"
        report += f"  └─ 迟到人数: {late_count:,} ({late_rate:.2f}%)\n\n"
        report += "【详细记录】\n"
        report += "-" * 60 + "\n"
        report += f"{'序号':<6} {'时间':<20} {'姓名':<10} {'学号':<12} {'状态':<8}\n"
        report += "-" * 60 + "\n"
        
        for i, log in enumerate(logs, 1):
            status = "成功" if log[5] == 'success' else "失败"
            report += f"{i:<6} {log[4]:<20} {log[2]:<10} {log[3]:<12} {status:<8}\n"
        
        report += "\n" + "=" * 60 + "\n"
        report += "报表结束\n"
        
        file_path, _ = QFileDialog.getSaveFileName(self, "保存报表", "", "文本文件 (*.txt)")
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report)
            QMessageBox.information(self, "导出成功", "报表文件已导出")
    
    def create_permission_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        button_layout = QHBoxLayout()
        
        self.refresh_person_btn = QPushButton("刷新人员列表")
        self.refresh_person_btn.clicked.connect(self.refresh_person_table)
        button_layout.addWidget(self.refresh_person_btn)
        
        self.import_btn = QPushButton("批量导入名单")
        self.import_btn.clicked.connect(self.import_persons)
        button_layout.addWidget(self.import_btn)
        
        self.delete_btn = QPushButton("删除选中人员")
        self.delete_btn.clicked.connect(self.delete_person)
        button_layout.addWidget(self.delete_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        self.person_table = QTableWidget()
        self.person_table.setColumnCount(6)
        self.person_table.setHorizontalHeaderLabels(["ID", "姓名", "学号", "院系", "角色", "状态"])
        self.person_table.setStyleSheet("QTableWidget { gridline-color: #ddd; }")
        layout.addWidget(self.person_table)
        
        permission_group = QGroupBox("权限设置")
        perm_layout = QGridLayout()
        
        perm_layout.addWidget(QLabel("选择人员:"), 0, 0)
        self.selected_person_label = QLabel("未选择")
        perm_layout.addWidget(self.selected_person_label, 0, 1)
        
        perm_layout.addWidget(QLabel("通行区域:"), 1, 0)
        self.area_input = QLineEdit()
        self.area_input.setPlaceholderText("如: 教学楼A区")
        perm_layout.addWidget(self.area_input, 1, 1)
        
        time_layout = QHBoxLayout()
        perm_layout.addWidget(QLabel("通行时段:"), 2, 0)
        self.start_time_input = QLineEdit()
        self.start_time_input.setPlaceholderText("开始时间: 08:00")
        time_layout.addWidget(self.start_time_input)
        self.end_time_input = QLineEdit()
        self.end_time_input.setPlaceholderText("结束时间: 18:00")
        time_layout.addWidget(self.end_time_input)
        perm_layout.addLayout(time_layout, 2, 1)
        
        date_layout = QHBoxLayout()
        perm_layout.addWidget(QLabel("有效期:"), 3, 0)
        self.valid_from_input = QLineEdit()
        self.valid_from_input.setPlaceholderText("开始日期: 2024-01-01")
        date_layout.addWidget(self.valid_from_input)
        self.valid_to_input = QLineEdit()
        self.valid_to_input.setPlaceholderText("结束日期: 2024-12-31")
        date_layout.addWidget(self.valid_to_input)
        perm_layout.addLayout(date_layout, 3, 1)
        
        self.set_permission_btn = QPushButton("设置权限")
        self.set_permission_btn.clicked.connect(self.set_permission)
        perm_layout.addWidget(self.set_permission_btn, 4, 0, 1, 2)
        
        permission_group.setLayout(perm_layout)
        layout.addWidget(permission_group)
        
        self.person_table.itemSelectionChanged.connect(self.on_person_select)
        
        widget.setLayout(layout)
        return widget
    
    def refresh_person_table(self):
        persons = self.db.get_all_persons()
        self.person_table.setRowCount(len(persons))
        
        for i, person in enumerate(persons):
            for j in range(6):
                self.person_table.setItem(i, j, QTableWidgetItem(str(person[j])))
    
    def on_person_select(self):
        selected_rows = self.person_table.selectedItems()
        if selected_rows:
            row = selected_rows[0].row()
            name = self.person_table.item(row, 1).text()
            student_id = self.person_table.item(row, 2).text()
            self.selected_person_label.setText(f"{name} ({student_id})")
    
    def delete_person(self):
        selected_rows = self.person_table.selectedItems()
        if not selected_rows:
            QMessageBox.warning(self, "提示", "请先选择要删除的人员")
            return
        
        reply = QMessageBox.question(self, "确认删除", "确定要删除选中的人员吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            row = selected_rows[0].row()
            person_id = int(self.person_table.item(row, 0).text())
            student_id = self.person_table.item(row, 2).text()
            
            if student_id in self.db.features:
                del self.db.features[student_id]
                self.db._save_features()
            
            cursor = self.db.conn.cursor()
            cursor.execute('DELETE FROM persons WHERE id = ?', (person_id,))
            cursor.execute('DELETE FROM permissions WHERE person_id = ?', (person_id,))
            self.db.conn.commit()
            
            self.refresh_person_table()
            QMessageBox.information(self, "删除成功", "人员信息已删除")
    
    def import_persons(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择CSV文件", "", "CSV Files (*.csv)")
        if not file_path:
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            imported_count = 0
            skipped_count = 0
            
            for i, line in enumerate(lines[1:], 2):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) < 2:
                    continue
                
                name = parts[0].strip()
                student_id = parts[1].strip()
                department = parts[2].strip() if len(parts) > 2 else ""
                role = parts[3].strip() if len(parts) > 3 else "student"
                
                if self.db.get_person_by_student_id(student_id):
                    skipped_count += 1
                    continue
                
                person_id = self.db.add_person(name, student_id, department, role)
                if person_id:
                    imported_count += 1
            
            self.refresh_person_table()
            QMessageBox.information(self, "导入完成", 
                                   f"成功导入 {imported_count} 人，跳过 {skipped_count} 人(已存在)")
        except Exception as e:
            QMessageBox.critical(self, "导入失败", f"导入过程中发生错误: {str(e)}")
    
    def set_permission(self):
        selected_rows = self.person_table.selectedItems()
        if not selected_rows:
            QMessageBox.warning(self, "提示", "请先选择人员")
            return
        
        row = selected_rows[0].row()
        person_id = int(self.person_table.item(row, 0).text())
        
        area = self.area_input.text()
        start_time = self.start_time_input.text()
        end_time = self.end_time_input.text()
        valid_from = self.valid_from_input.text()
        valid_to = self.valid_to_input.text()
        
        if not area:
            QMessageBox.warning(self, "提示", "请填写通行区域")
            return
        
        cursor = self.db.conn.cursor()
        cursor.execute('''
            INSERT INTO permissions (person_id, area, start_time, end_time, valid_from, valid_to)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (person_id, area, start_time, end_time, valid_from, valid_to))
        self.db.conn.commit()
        
        QMessageBox.information(self, "设置成功", "权限已添加")
        
        self.area_input.clear()
        self.start_time_input.clear()
        self.end_time_input.clear()
        self.valid_from_input.clear()
        self.valid_to_input.clear()
    
    def display_image(self, frame, label):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q_image))
    
    def closeEvent(self, event):
        self.db.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
