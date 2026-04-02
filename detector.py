from ultralytics import YOLO
import cv2
import time
import torch

# --- 1. SETUP DEVICE & MODEL ---
# Pakai GPU kalau sudah instal versi CUDA, kalau belum otomatis ke CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Running on: {device.upper()}")

# Load model YOLOv8 Nano (Sangat ringan untuk i5-12450HX)
model = YOLO('yolov8n.pt').to(device) 

cap = cv2.VideoCapture(0)
# Set resolusi standar
cap.set(3, 640)
cap.set(4, 480)

window_name = "Human Detector - Rayan UPN"
prev_time = 0

print("✅ Program Aktif. Mencari objek manusia...")

while True:
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1) # Mirroring

    # --- 2. DETEKSI YOLO ---
    # imgsz=320 untuk menjaga FPS tetap stabil di CPU
    results = model(frame, imgsz=320, stream=True, verbose=False)

    human_count = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Class 0 adalah 'person' dalam dataset COCO
            if int(box.cls[0]) == 0: 
                human_count += 1
                
                # Ambil koordinat kotak
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0]) # Tingkat kepercayaan (0.0 - 1.0)
                
                # Gambar Bounding Box & Label
                # Warna biru (255, 0, 0) biar beda dari yang kemarin
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"HUMAN {conf:.2%}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # --- 3. UI & FPS ---
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    # Tampilkan FPS dan Jumlah Manusia terdeteksi
    cv2.rectangle(frame, (10, 10), (250, 80), (0, 0, 0), -1) # Background hitam buat info
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Count: {human_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # --- 4. DISPLAY & EXIT ---
    cv2.imshow(window_name, frame)

    # Close pakai tombol 'Q' atau tombol silang [X]
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()