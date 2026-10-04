import cv2
import os

def collect_faces(person_name, num_samples=50):
    """使用摄像头采集人脸数据"""
    print(f"=== 人脸采集 - {person_name} ===")
    print(f"将采集 {num_samples} 张照片")
    print("按 's' 保存照片，按 'q' 退出")
    
    output_dir = f'./dataset/{person_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    count = 0
    saved_count = 0
    
    while saved_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"已保存: {saved_count}/{num_samples}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Face Collection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s') and len(faces) > 0:
            x, y, w, h = faces[0]
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (256, 256))
            
            filename = f'{output_dir}/{saved_count:04d}.jpg'
            cv2.imwrite(filename, face_img)
            saved_count += 1
            print(f"已保存: {saved_count}/{num_samples}")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n采集完成！共保存 {saved_count} 张照片")
    print(f"照片保存在: {output_dir}")

if __name__ == '__main__':
    person_name = input("请输入采集人的姓名: ")
    num_samples = int(input("请输入采集数量 (默认50): ") or 50)
    
    collect_faces(person_name, num_samples)