# 👤 AI Human Detector: Real-Time Object Detection

**Human Detector** adalah aplikasi visi komputer (Computer Vision) berbasis *Deep Learning* yang dirancang untuk mendeteksi keberadaan manusia secara real-time melalui kamera. Proyek ini menggunakan arsitektur **YOLOv8 (You Only Look Once)** yang dioptimasi untuk performa tinggi pada perangkat konsumen.



## 🚀 Fitur Utama
- **High-Precision Detection:** Menggunakan model YOLOv8n (Nano) yang dilatih pada dataset COCO (80 kelas).
- **Real-Time FPS Counter:** Monitor performa pemrosesan frame per detik secara langsung.
- **Human Counter:** Menghitung jumlah individu yang terdeteksi dalam satu frame.
- **Confidence Scoring:** Menampilkan tingkat keyakinan AI dalam mendeteksi objek manusia.
- **Smart Exit:** Mendukung penutupan aplikasi via tombol 'Q' atau tombol silang [X] pada jendela.

## 💻 Spesifikasi Sistem (Tested On)
- **OS:** Windows 11
- **Python Version:** 3.12.x
- **Inference Speed:** ~8-12 FPS (CPU Mode)

## 📦 Persiapan & Instalasi
Aplikasi ini membutuhkan library `ultralytics` untuk menjalankan mesin YOLO. Pastikan kamu menggunakan Python 3.12.

```powershell
# 1. Update/Instal library utama
pip install ultralytics opencv-python torch torchvision

# 2. (Opsional) Untuk performa maksimal menggunakan GPU RTX 2050:
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
