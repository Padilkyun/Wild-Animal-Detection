# 🎯 YOLO Detection Studio — Streamlit Dashboard

Dashboard real-time object detection menggunakan model ONNX hasil training ulang YOLO.

## ✅ Fitur
- Upload model `.onnx` langsung dari browser
- Upload video (MP4, AVI, MOV, MKV, WebM)
- Live detection feed dengan bounding box stylized
- Statistik per frame: FPS, jumlah deteksi, inference time
- Class counts kumulatif selama video berjalan
- Quick image test untuk satu gambar
- Download hasil video yang sudah dianotasi
- Support YOLOv5 / YOLOv8 / YOLOv9 / YOLOv10 format output
- Auto-detect GPU (CUDA) atau fallback ke CPU

---

## 🚀 Cara Menjalankan

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Pakai GPU NVIDIA?** Ganti `onnxruntime` dengan `onnxruntime-gpu`:
> ```bash
> pip install onnxruntime-gpu opencv-python-headless numpy Pillow pandas streamlit
> ```

### 2. Jalankan Streamlit

```bash
streamlit run app.py
```

Dashboard akan terbuka di `http://localhost:8501`

---

## 📂 Struktur File

```
yolo_dashboard/
├── app.py              # Aplikasi utama
├── requirements.txt    # Dependencies
└── README.md           # Dokumentasi ini
```

---

## 🔧 Konfigurasi Sidebar

| Setting | Keterangan |
|---|---|
| **Upload model .onnx** | File model hasil training YOLO |
| **Confidence threshold** | Filter deteksi di bawah nilai ini diabaikan |
| **IoU threshold** | Untuk NMS (Non-Maximum Suppression) |
| **Frame skip** | Skip frame untuk mempercepat processing (0 = semua frame) |
| **Class Names** | Isi manual jika model tidak menyimpan nama kelas |
| **Save output** | Simpan video hasil anotasi untuk didownload |

---

## 🧩 Export Model ke ONNX

### Dari YOLOv8 (Ultralytics):
```python
from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="onnx", opset=12, simplify=True)
```

### Dari YOLOv5:
```bash
python export.py --weights best.pt --include onnx --opset 12
```

---

## ⚙️ Format Output ONNX yang Didukung

| Format | Shape | Keterangan |
|---|---|---|
| YOLOv8/v9/v10 | `[1, 4+nc, anchors]` | Default Ultralytics export |
| YOLOv5 | `[1, anchors, 5+nc]` | Default YOLOv5 export |

---

## 🐛 Troubleshooting

**Model tidak terdeteksi kelasnya?**
→ Isi nama kelas secara manual di sidebar "Class Names"

**Deteksi lambat?**
→ Naikkan nilai "Frame Skip" di sidebar
→ Install `onnxruntime-gpu` jika punya GPU NVIDIA

**Error saat load model?**
→ Pastikan model diekspor dengan `opset=12` atau `opset=11`
→ Coba simplify: `model.export(format="onnx", simplify=True)`

---

## 📦 Deploy ke Streamlit Cloud

1. Push ke GitHub repository
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → pilih `app.py`
4. Deploy!

> Note: Streamlit Cloud tidak support GPU. Untuk performa optimal, deploy di server dengan NVIDIA GPU.