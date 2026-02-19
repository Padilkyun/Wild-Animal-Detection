import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from pathlib import Path
import onnxruntime as ort
from PIL import Image
import pandas as pd

st.set_page_config(
    page_title="Wild Animal Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

    :root {
        --bg-dark: #0a0a0f;
        --bg-card: #12121a;
        --bg-card2: #1a1a26;
        --accent: #00ff88;
        --accent2: #ff6b35;
        --accent3: #7c3aed;
        --text: #e8e8f0;
        --text-muted: #6b6b80;
        --border: rgba(255,255,255,0.06);
    }

    html, body, .stApp {
        background-color: var(--bg-dark) !important;
        color: var(--text) !important;
        font-family: 'Syne', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background: var(--bg-card) !important;
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        letter-spacing: -0.02em;
    }

    [data-testid="metric-container"] {
        background: var(--bg-card2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }
    [data-testid="metric-container"] label {
        color: var(--text-muted) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--accent) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 28px !important;
        font-weight: 700;
    }

    .stButton > button {
        background: var(--accent) !important;
        color: #000 !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 10px 24px !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: #00cc6a !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(0,255,136,0.3) !important;
    }

    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: var(--accent) !important;
    }

    .stProgress > div > div {
        background: linear-gradient(90deg, var(--accent3), var(--accent)) !important;
    }

    [data-testid="stFileUploader"] {
        background: var(--bg-card2) !important;
        border: 2px dashed rgba(0,255,136,0.2) !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent) !important;
    }

    .stSelectbox [data-baseweb="select"] > div {
        background: var(--bg-card2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
    }

    .stDataFrame {
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: var(--bg-card2) !important;
    }

    .stAlert {
        border-radius: 10px !important;
        border: none !important;
    }

    hr {
        border-color: var(--border) !important;
        margin: 24px 0;
    }

    code, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .badge-green { background: rgba(0,255,136,0.15); color: #00ff88; border: 1px solid rgba(0,255,136,0.3); }
    .badge-orange { background: rgba(255,107,53,0.15); color: #ff6b35; border: 1px solid rgba(255,107,53,0.3); }
    .badge-purple { background: rgba(124,58,237,0.15); color: #a78bfa; border: 1px solid rgba(124,58,237,0.3); }

    .detect-log {
        background: var(--bg-card2);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        max-height: 220px;
        overflow-y: auto;
        line-height: 1.8;
    }

    .hero {
        background: linear-gradient(135deg, var(--bg-card2) 0%, rgba(124,58,237,0.1) 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(0,255,136,0.05) 0%, transparent 70%);
        border-radius: 50%;
    }

    footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    button[kind="header"] { display: none; }
</style>
""", unsafe_allow_html=True)


MODELS = {
    "YOLOv5": {
        "Babi Hutan": "YoloV5/yolov5-babi-hutan/yolov5_babihutan.onnx",
        "Gajah":     "YoloV5/yolov5-gajah/yolov5_gajah.onnx",
        "Harimau":   "YoloV5/yolov5-harimau/yolov5_harimau.onnx",
        "Orangutan": "YoloV5/Yolov5-orangutan/yolov5_orangutan.onnx",
    },
    "YOLOv8": {
        "Babi Hutan": "YoloV8/YoloV8-babihutan/yolov8_babihutan.onnx",
        "Gajah":      "YoloV8/YoloV8-gajah/yolov8_gajah.onnx",
        "Harimau":    "YoloV8/YoloV8-Harimau/yolov8_harimau.onnx",
        "Orangutan":  "YoloV8/Yolov8-orangutan/yolov8_orangutan.onnx",
    },
    "YOLOv11": {
        "Babi Hutan": "YoloV11/YoloV11-babihutan/yolov11_babihutan.onnx",
        "Gajah":      "YoloV11/YoloV11-gajah/yolov11_gajah.onnx",
        "Harimau":    "YoloV11/YoloV11-Harimau/yolov11_harimau.onnx",
        "Orangutan":  "YoloV11/YoloV11-Orangutan/yolov11_orangutan.onnx",
    },
}

ANIMAL_CLASSES = {
    "Babi Hutan": ["babi_hutan"],
    "Gajah":      ["gajah"],
    "Harimau":    ["harimau"],
    "Orangutan":  ["orangutan"],
}


class YOLODetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
        except Exception:
            self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])

        meta = self.session.get_inputs()[0]
        self.input_name = meta.name
        self.input_shape = meta.shape

        model_meta = self.session.get_modelmeta()
        if hasattr(model_meta, 'custom_metadata_map') and 'names' in model_meta.custom_metadata_map:
            import ast
            raw = model_meta.custom_metadata_map['names']
            try:
                names_dict = ast.literal_eval(raw)
                self.class_names = [names_dict[i] for i in sorted(names_dict.keys())]
            except Exception:
                self.class_names = None
        else:
            self.class_names = None

        h = self.input_shape[2] if isinstance(self.input_shape[2], int) and self.input_shape[2] > 0 else 416
        w = self.input_shape[3] if isinstance(self.input_shape[3], int) and self.input_shape[3] > 0 else 416
        self.input_h = h
        self.input_w = w

    def _letterbox(self, frame: np.ndarray):
        orig_h, orig_w = frame.shape[:2]
        scale = min(self.input_w / orig_w, self.input_h / orig_h)

        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = self.input_w - new_w
        pad_h = self.input_h - new_h

        top    = pad_h // 2
        bottom = pad_h - top
        left   = pad_w // 2
        right  = pad_w - left

        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))

        meta = {"scale": scale, "pad_left": left, "pad_top": top}
        return padded, meta

    def preprocess(self, frame: np.ndarray):
        img_bgr, self._lb_meta = self._letterbox(frame)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_norm, (2, 0, 1))
        return np.expand_dims(img_chw, axis=0)

    def postprocess(self, outputs, orig_h: int, orig_w: int):
        predictions = outputs[0]
        scale    = self._lb_meta["scale"]
        pad_left = self._lb_meta["pad_left"]
        pad_top  = self._lb_meta["pad_top"]

        if predictions.ndim == 3:
            predictions = predictions[0]

            is_yolov8 = predictions.shape[0] < predictions.shape[1]

            if is_yolov8:
                predictions = predictions.T

                cx = predictions[:, 0]
                cy = predictions[:, 1]
                bw = predictions[:, 2]
                bh = predictions[:, 3]
                scores_all = predictions[:, 4:]

                x1_lb = cx - bw / 2
                y1_lb = cy - bh / 2
                x2_lb = cx + bw / 2
                y2_lb = cy + bh / 2

                class_ids   = np.argmax(scores_all, axis=1)
                confidences = scores_all[np.arange(len(scores_all)), class_ids]

            else:
                obj_conf    = predictions[:, 4]
                cls_scores  = predictions[:, 5:]
                class_ids   = np.argmax(cls_scores, axis=1)
                class_confs = cls_scores[np.arange(len(cls_scores)), class_ids]
                confidences = obj_conf * class_confs

                cx = predictions[:, 0]
                cy = predictions[:, 1]
                bw = predictions[:, 2]
                bh = predictions[:, 3]
                x1_lb = cx - bw / 2
                y1_lb = cy - bh / 2
                x2_lb = cx + bw / 2
                y2_lb = cy + bh / 2

        else:
            return [], [], []

        mask = confidences >= self.conf_threshold
        x1_lb, y1_lb = x1_lb[mask], y1_lb[mask]
        x2_lb, y2_lb = x2_lb[mask], y2_lb[mask]
        confidences = confidences[mask]
        class_ids   = class_ids[mask]

        if len(confidences) == 0:
            return [], [], []

        x1 = np.clip((x1_lb - pad_left) / scale, 0, orig_w)
        y1 = np.clip((y1_lb - pad_top)  / scale, 0, orig_h)
        x2 = np.clip((x2_lb - pad_left) / scale, 0, orig_w)
        y2 = np.clip((y2_lb - pad_top)  / scale, 0, orig_h)

        valid = (x2 - x1 > 1) & (y2 - y1 > 1)
        x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
        confidences = confidences[valid]
        class_ids   = class_ids[valid]

        if len(confidences) == 0:
            return [], [], []

        boxes_pixel = np.stack([x1, y1, x2, y2], axis=1).astype(int)

        boxes_for_nms = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes_pixel]
        indices = cv2.dnn.NMSBoxes(boxes_for_nms, confidences.tolist(), self.conf_threshold, self.iou_threshold)

        if len(indices) == 0:
            return [], [], []

        indices = indices.flatten()
        return boxes_pixel[indices].tolist(), confidences[indices].tolist(), class_ids[indices].tolist()

    def detect(self, frame: np.ndarray):
        orig_h, orig_w = frame.shape[:2]
        blob = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: blob})
        return self.postprocess(outputs, orig_h, orig_w)

    def draw(self, frame: np.ndarray, boxes, scores, class_ids, class_names=None):
        colors = [
            (0, 255, 136), (255, 107, 53), (124, 58, 237),
            (255, 220, 0), (0, 191, 255), (255, 0, 128),
            (57, 255, 20), (255, 165, 0), (138, 43, 226), (0, 255, 255)
        ]
        annotated = frame.copy()
        names = class_names or self.class_names

        for box, score, cid in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            color = colors[cid % len(colors)]
            label = (names[cid] if names and cid < len(names) else f"cls_{cid}")
            label_text = f"{label} {score:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
            for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                cv2.line(annotated, (cx, cy), (cx + dx * corner_len, cy), color, 3)
                cv2.line(annotated, (cx, cy), (cx, cy + dy * corner_len), color, 3)

            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            lx1, ly1 = x1, max(0, y1 - th - 8)
            lx2, ly2 = x1 + tw + 10, y1
            cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), color, -1)
            cv2.putText(annotated, label_text, (lx1 + 5, ly2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        return annotated


@st.cache_resource
def load_model_cached(model_path, conf, iou):
    return YOLODetector(model_path, conf, iou)


def process_video(detector, video_path, class_names, skip_frames=1, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0
    detect_counts = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        if frame_idx % (skip_frames + 1) == 0:
            t0 = time.time()
            boxes, scores, class_ids = detector.detect(frame)
            infer_ms = (time.time() - t0) * 1000
            annotated = detector.draw(frame, boxes, scores, class_ids, class_names)

            for cid in class_ids:
                name = (class_names[cid] if class_names and cid < len(class_names) else f"cls_{cid}")
                detect_counts[name] = detect_counts.get(name, 0) + 1

            stats = {
                "frame": frame_idx,
                "total": total,
                "fps": fps,
                "n_detect": len(boxes),
                "infer_ms": infer_ms,
                "counts": dict(detect_counts),
            }
            yield annotated, stats

        frame_idx += 1

    cap.release()


with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:24px'>
        <div style='font-family:Syne,sans-serif;font-size:22px;font-weight:800;letter-spacing:-0.03em;color:#e8e8f0'>
            🎯 Konfigurasi Model
        </div>
        <div style='font-family:JetBrains Mono,monospace;font-size:11px;color:#6b6b80;margin-top:4px'>
            Real Time Detection
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**🦾 Versi YOLO**")
    yolo_version = st.selectbox("Pilih versi YOLO", list(MODELS.keys()), index=2)

    st.markdown("**🐾 Hewan**")
    animal_name = st.selectbox("Pilih hewan", list(MODELS[yolo_version].keys()))

    selected_model_path = MODELS[yolo_version][animal_name]
    model_exists = os.path.exists(selected_model_path)

    if model_exists:
        st.markdown(f"""
        <div style='background:rgba(0,255,136,0.08);border:1px solid rgba(0,255,136,0.25);border-radius:8px;padding:10px 12px;margin:8px 0;font-family:JetBrains Mono,monospace;font-size:11px;color:#00ff88'>
            ✅ Model ditemukan<br><span style='color:#6b6b80'>{selected_model_path}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:rgba(255,107,53,0.08);border:1px solid rgba(255,107,53,0.25);border-radius:8px;padding:10px 12px;margin:8px 0;font-family:JetBrains Mono,monospace;font-size:11px;color:#ff6b35'>
            ❌ File tidak ditemukan<br><span style='color:#6b6b80'>{selected_model_path}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**⚙️ Detection Config**")
    conf_thresh = st.slider("Confidence threshold", 0.10, 0.95, 0.40, 0.01)
    iou_thresh = st.slider("IoU threshold (NMS)", 0.10, 0.90, 0.45, 0.01)
    skip_frames = st.selectbox("Frame skip (speed ↑ quality ↓)", [0, 1, 2, 4], index=1)

    st.markdown("---")
    st.markdown("**📐 Input Size**")
    override_size = st.checkbox("Override input size", value=False)
    if override_size:
        input_size = st.selectbox(
            "Input size (H = W)",
            [320, 416, 512, 608, 640, 736, 832],
            index=1,
        )
    else:
        input_size = 416

    st.markdown("---")
    st.markdown("**🏷️ Class Names** *(optional override)*")
    class_input = st.text_area(
        "One class per line",
        placeholder="babi_hutan\ngajah\nharimau\norangutan",
        height=100,
    )

    st.markdown("---")
    st.markdown("**🎬 Output**")
    save_output = st.checkbox("Save annotated video", value=True)
    show_log = st.checkbox("Show detection log", value=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#6b6b80;line-height:1.8'>
        Universitas Andalas Copyright@2026<br>
        YOLOv5 / v8 / v11 compatible
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div class='hero'>
    <div style='font-family:Syne,sans-serif;font-size:36px;font-weight:800;letter-spacing:-0.04em;margin-bottom:6px'>
        Animal Real-Time Detection
    </div>
    <div style='margin-top:14px;display:flex;gap:8px'>
        <span class='badge badge-green'>ONNX Runtime</span>
        <span class='badge badge-purple'>YOLOv5/v8/v11</span>
        <span class='badge badge-orange'>GPU / CPU</span>
    </div>
</div>
""", unsafe_allow_html=True)

custom_class_names = None
if class_input.strip():
    custom_class_names = [c.strip() for c in class_input.strip().split("\n") if c.strip()]
else:
    custom_class_names = ANIMAL_CLASSES.get(animal_name)

if "detector" not in st.session_state:
    st.session_state.detector = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "loaded_model_key" not in st.session_state:
    st.session_state.loaded_model_key = None

current_model_key = (yolo_version, animal_name, input_size)

if model_exists and st.session_state.loaded_model_key != current_model_key:
    with st.spinner(f"Loading {yolo_version} — {animal_name} ({input_size}×{input_size})..."):
        try:
            detector_obj = YOLODetector(selected_model_path, conf_thresh, iou_thresh)
            detector_obj.input_h = input_size
            detector_obj.input_w = input_size
            st.session_state.detector = detector_obj
            st.session_state.model_loaded = True
            st.session_state.loaded_model_key = current_model_key
            st.toast(f"✅ {yolo_version} {animal_name} loaded", icon="🎯")
        except Exception as e:
            st.error(f"❌ Gagal memuat model: {e}")
            st.session_state.model_loaded = False
elif model_exists and st.session_state.detector:
    st.session_state.detector.conf_threshold = conf_thresh
    st.session_state.detector.iou_threshold = iou_thresh
    st.session_state.detector.input_h = input_size
    st.session_state.detector.input_w = input_size

col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

with col_stat1:
    status_txt = "READY" if st.session_state.model_loaded else "NO MODEL"
    color = "#00ff88" if st.session_state.model_loaded else "#ff6b35"
    st.markdown(f"""
    <div style='background:var(--bg-card2);border:1px solid var(--border);border-radius:12px;padding:16px'>
        <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em'>Model Status</div>
        <div style='font-family:JetBrains Mono,monospace;font-size:22px;font-weight:700;color:{color};margin-top:6px'>{status_txt}</div>
        <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#6b6b80;margin-top:2px'>{yolo_version} · {animal_name}</div>
    </div>""", unsafe_allow_html=True)

with col_stat2:
    if st.session_state.model_loaded and st.session_state.detector:
        det = st.session_state.detector
        inp = f"{det.input_h}×{det.input_w}"
        inp_sub = "from model" if not override_size else "overridden"
    else:
        inp = f"{input_size}×{input_size}"
        inp_sub = "default"
    st.markdown(f"""
    <div style='background:var(--bg-card2);border:1px solid var(--border);border-radius:12px;padding:16px'>
        <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em'>Input Size</div>
        <div style='font-family:JetBrains Mono,monospace;font-size:22px;font-weight:700;color:#a78bfa;margin-top:6px'>{inp}</div>
        <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#6b6b80;margin-top:2px'>{inp_sub}</div>
    </div>""", unsafe_allow_html=True)

with col_stat3:
    if custom_class_names:
        nc = len(custom_class_names)
    elif st.session_state.model_loaded and st.session_state.detector and st.session_state.detector.class_names:
        nc = len(st.session_state.detector.class_names)
    else:
        nc = "?"
    st.markdown(f"""
    <div style='background:var(--bg-card2);border:1px solid var(--border);border-radius:12px;padding:16px'>
        <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em'>Classes</div>
        <div style='font-family:JetBrains Mono,monospace;font-size:22px;font-weight:700;color:#00ff88;margin-top:6px'>{nc}</div>
    </div>""", unsafe_allow_html=True)

with col_stat4:
    providers = ort.get_available_providers() if st.session_state.model_loaded else []
    hw = "CUDA" if "CUDAExecutionProvider" in providers else "CPU"
    hw_color = "#00ff88" if hw == "CUDA" else "#ff6b35"
    st.markdown(f"""
    <div style='background:var(--bg-card2);border:1px solid var(--border);border-radius:12px;padding:16px'>
        <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em'>Hardware</div>
        <div style='font-family:JetBrains Mono,monospace;font-size:22px;font-weight:700;color:{hw_color};margin-top:6px'>{hw}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

st.markdown("### 🎬 Upload Video")
video_file = st.file_uploader("Pilih file video", type=["mp4", "avi", "mov", "mkv", "webm"])

if not st.session_state.model_loaded:
    if not model_exists:
        st.error(f"❌ File model tidak ditemukan: `{selected_model_path}` — pastikan folder model sudah ada.")
    else:
        st.info("⏳ Memuat model...")

if video_file and st.session_state.model_loaded:
    det = st.session_state.detector
    class_names = custom_class_names or det.class_names

    with tempfile.NamedTemporaryFile(suffix="." + video_file.name.split(".")[-1], delete=False) as tmp_vid:
        tmp_vid.write(video_file.read())
        tmp_vid_path = tmp_vid.name

    cap_info = cv2.VideoCapture(tmp_vid_path)
    total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_info = cap_info.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_info.release()

    st.markdown("---")
    st.markdown(f"""
    <div style='font-family:JetBrains Mono,monospace;font-size:12px;color:var(--text-muted);margin-bottom:12px'>
        📹 {video_file.name} &nbsp;|&nbsp; {vid_w}×{vid_h} &nbsp;|&nbsp; {fps_info:.1f} fps &nbsp;|&nbsp; {total_frames} frames
    </div>
    """, unsafe_allow_html=True)

    col_run, col_preview = st.columns([1, 3])
    with col_run:
        run_btn = st.button("▶  Run Detection", use_container_width=True)

    if run_btn:
        st.markdown("### 📡 Live Detection Feed")

        vid_col, metrics_col = st.columns([3, 1])

        with vid_col:
            video_placeholder = st.empty()

        with metrics_col:
            fps_display = st.empty()
            det_display = st.empty()
            count_display = st.empty()

        prog = st.progress(0, text="Processing frames…")
        log_placeholder = st.empty()

        tmp_out_path = None
        writer = None
        if save_output:
            tmp_out = tempfile.NamedTemporaryFile(suffix="_detected.mp4", delete=False)
            tmp_out_path = tmp_out.name
            tmp_out.close()

        log_lines = []
        all_counts = {}
        processed = 0
        t_start = time.time()

        for annotated, stats in process_video(det, tmp_vid_path, class_names, skip_frames=skip_frames):
            processed += 1
            progress_val = min(stats["frame"] / max(stats["total"], 1), 1.0)
            prog.progress(progress_val, text=f"Frame {stats['frame']}/{stats['total']}")

            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, use_container_width=True)

            elapsed = time.time() - t_start
            real_fps = processed / max(elapsed, 0.001)
            fps_display.markdown(f"""
            <div style='background:var(--bg-card2);border:1px solid var(--border);border-radius:10px;padding:12px;text-align:center;margin-bottom:8px'>
                <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:var(--text-muted)'>PROCESSING FPS</div>
                <div style='font-family:JetBrains Mono,monospace;font-size:28px;font-weight:700;color:#00ff88'>{real_fps:.1f}</div>
            </div>""", unsafe_allow_html=True)

            det_display.markdown(f"""
            <div style='background:var(--bg-card2);border:1px solid var(--border);border-radius:10px;padding:12px;text-align:center;margin-bottom:8px'>
                <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:var(--text-muted)'>DETECTIONS</div>
                <div style='font-family:JetBrains Mono,monospace;font-size:28px;font-weight:700;color:#ff6b35'>{stats["n_detect"]}</div>
                <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:var(--text-muted)'>{stats["infer_ms"]:.0f} ms/frame</div>
            </div>""", unsafe_allow_html=True)

            for cls, cnt in stats["counts"].items():
                all_counts[cls] = cnt
            if all_counts:
                counts_html = "".join([
                    f"<div style='display:flex;justify-content:space-between;font-family:JetBrains Mono,monospace;font-size:11px;margin:3px 0'>"
                    f"<span style='color:var(--text-muted)'>{k}</span><span style='color:#00ff88;font-weight:700'>{v}</span></div>"
                    for k, v in sorted(all_counts.items(), key=lambda x: -x[1])[:8]
                ])
                count_display.markdown(f"""
                <div style='background:var(--bg-card2);border:1px solid var(--border);border-radius:10px;padding:12px;margin-bottom:8px'>
                    <div style='font-family:JetBrains Mono,monospace;font-size:10px;color:var(--text-muted);margin-bottom:8px'>CLASS COUNTS</div>
                    {counts_html}
                </div>""", unsafe_allow_html=True)

            if show_log and stats["n_detect"] > 0:
                log_lines.append(f"[f{stats['frame']:05d}] → {stats['n_detect']} obj | {stats['infer_ms']:.0f}ms")
                if len(log_lines) > 40:
                    log_lines = log_lines[-40:]
                log_html = "<br>".join(
                    f"<span style='color:{'#00ff88' if i == len(log_lines)-1 else '#6b6b80'}'>{l}</span>"
                    for i, l in enumerate(log_lines)
                )
                log_placeholder.markdown(f"<div class='detect-log'>{log_html}</div>", unsafe_allow_html=True)

            if save_output and tmp_out_path:
                if writer is None:
                    h_a, w_a = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(tmp_out_path, fourcc, max(fps_info, 1), (w_a, h_a))
                writer.write(annotated)

        if writer:
            writer.release()

        prog.progress(1.0, text="✅ Detection complete!")

        st.markdown("---")
        st.markdown("### 📊 Detection Summary")

        s1, s2, s3, s4 = st.columns(4)
        elapsed_total = time.time() - t_start
        s1.metric("Frames Processed", processed)
        s2.metric("Total Detections", sum(all_counts.values()))
        s3.metric("Unique Classes", len(all_counts))
        s4.metric("Processing Time", f"{elapsed_total:.1f}s")

        if all_counts:
            df = pd.DataFrame(
                sorted(all_counts.items(), key=lambda x: -x[1]),
                columns=["Class", "Total Detections"]
            )
            st.markdown("#### Detections per Class")
            st.bar_chart(df.set_index("Class"))
            st.dataframe(df, use_container_width=True, hide_index=True)

        if save_output and tmp_out_path and os.path.exists(tmp_out_path):
            with open(tmp_out_path, "rb") as f:
                vid_bytes = f.read()
            st.markdown("#### ⬇️ Download Result")
            out_name = Path(video_file.name).stem + "_detected.mp4"
            st.download_button(
                label="Download Annotated Video",
                data=vid_bytes,
                file_name=out_name,
                mime="video/mp4",
                use_container_width=False,
            )

elif video_file and not st.session_state.model_loaded:
    st.warning("⚠️ Model belum berhasil dimuat. Periksa apakah file .onnx tersedia di folder yang benar.")

st.markdown("---")
st.markdown("### 🖼️ Quick Image Test")
img_file = st.file_uploader("Atau uji pada satu gambar", type=["jpg", "jpeg", "png", "bmp", "webp"], key="img_test")

if img_file and st.session_state.model_loaded:
    det = st.session_state.detector
    class_names = custom_class_names or det.class_names

    img_pil = Image.open(img_file).convert("RGB")
    img_np = np.array(img_pil)
    frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    t0 = time.time()
    boxes, scores, class_ids = det.detect(frame_bgr)
    infer_ms = (time.time() - t0) * 1000
    annotated = det.draw(frame_bgr, boxes, scores, class_ids, class_names)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    col_orig, col_ann = st.columns(2)
    with col_orig:
        st.markdown("**Original**")
        st.image(img_pil, use_container_width=True)
    with col_ann:
        st.markdown(f"**Detected — {len(boxes)} objects | {infer_ms:.0f} ms**")
        st.image(annotated_rgb, use_container_width=True)

    if boxes:
        det_data = []
        for box, score, cid in zip(boxes, scores, class_ids):
            name = (class_names[cid] if class_names and cid < len(class_names) else f"cls_{cid}")
            det_data.append({"Class": name, "Confidence": f"{score:.3f}",
                             "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]})
        st.dataframe(pd.DataFrame(det_data), use_container_width=True, hide_index=True)