import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# Paths and runtime config
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", os.path.join("models", "yolo", "best.pt"))
CRNN_MODEL_PATH = os.getenv("CRNN_MODEL_PATH", os.path.join("models", "crnn", "34e_crnn_simple_best.pth"))

BASE_OUTPUT = os.path.join(os.getcwd(), "TrainingData")
TEMP_DIR = os.path.join("Temp")
CRNN_TRAIN_DIR = os.path.join(BASE_OUTPUT, "CRNN_Data_Training")
OUT_JSON = "crnn_results.json"

TARGET_WIDTH = 700
TARGET_HEIGHT = 300
INTERP = cv2.INTER_CUBIC

IMG_HEIGHT = 48
IMG_WIDTH = 192
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_VEHICLE_NUMBER_LENGTH = 10

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK_IDX = 0
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(ALPHABET)}

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(CRNN_TRAIN_DIR, exist_ok=True)

_YOLO_MODEL = None
_CRNN_MODEL = None


class SimpleCRNN(nn.Module):
    def __init__(self, img_h: int, n_classes: int, hidden: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_h, IMG_WIDTH)
            feats = self.cnn(dummy)
            _, c, h, _ = feats.size()
            self.feat_dim = c * h

        self.rnn = nn.LSTM(
            input_size=self.feat_dim,
            hidden_size=hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        feats = self.cnn(x)
        b, c, h, w = feats.size()
        feats = feats.permute(0, 3, 1, 2)
        feats = feats.contiguous().view(b, w, c * h)
        seq, _ = self.rnn(feats)
        logits = self.fc(seq)
        return logits.permute(1, 0, 2)


INFER_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def _load_yolo() -> YOLO | None:
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"[ERR] YOLO model not found: {YOLO_MODEL_PATH}")
        return None
    try:
        _YOLO_MODEL = YOLO(YOLO_MODEL_PATH)
        return _YOLO_MODEL
    except Exception as exc:
        print(f"[ERR] Failed to load YOLO model: {exc}")
        return None


def _load_crnn() -> nn.Module | None:
    global _CRNN_MODEL
    if _CRNN_MODEL is not None:
        return _CRNN_MODEL
    if not os.path.exists(CRNN_MODEL_PATH):
        print(f"[ERR] CRNN model not found: {CRNN_MODEL_PATH}")
        return None
    try:
        model = SimpleCRNN(IMG_HEIGHT, len(ALPHABET) + 1).to(DEVICE)
        model.load_state_dict(torch.load(CRNN_MODEL_PATH, map_location=DEVICE))
        model.eval()
        _CRNN_MODEL = model
        return _CRNN_MODEL
    except Exception as exc:
        print(f"[ERR] Failed to load CRNN model: {exc}")
        return None


def detect_and_crop(image_path: str) -> np.ndarray | None:
    yolo_model = _load_yolo()
    if yolo_model is None:
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERR] Cannot read image: {image_path}")
        return None

    h, w = img.shape[:2]
    results = yolo_model.predict(img, imgsz=640, verbose=False)[0]

    if results.boxes is None or len(results.boxes) == 0:
        print(f"[WARN] No plate detected in: {image_path}")
        return None

    best = results.boxes[0]
    xmin, ymin, xmax, ymax = best.xyxy[0].cpu().numpy().astype(int)
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w - 1, xmax)
    ymax = min(h - 1, ymax)

    if xmin >= xmax or ymin >= ymax:
        print(f"[WARN] Invalid crop box for: {image_path}")
        return None

    return img[ymin:ymax, xmin:xmax]


def process_image(image_path: str, plate_text: str) -> tuple[np.ndarray | None, str | None]:
    crop = detect_and_crop(image_path)
    if crop is None:
        return None, None

    name = "".join(c for c in plate_text if c.isalnum())
    if not name:
        name = "unknown_plate"
    return crop, name


def resize_plate(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(TARGET_WIDTH / w, TARGET_HEIGHT / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=INTERP)

    canvas = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=img.dtype)
    x_offset = (TARGET_WIDTH - new_w) // 2
    y_offset = (TARGET_HEIGHT - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas


def _clear_temp_folder() -> None:
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def save_images_and_label(crop_img: np.ndarray, base_name: str, plate_text: str) -> str | None:
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_3ch = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    resized = resize_plate(bw_3ch)

    img_filename = f"{base_name}.jpg"
    txt_filename = f"{base_name}.txt"

    temp_img_path = os.path.join(TEMP_DIR, img_filename)
    train_img_path = os.path.join(CRNN_TRAIN_DIR, img_filename)
    train_txt_path = os.path.join(CRNN_TRAIN_DIR, txt_filename)

    _clear_temp_folder()
    cv2.imwrite(temp_img_path, resized)

    if "unknown" not in base_name.lower():
        cv2.imwrite(train_img_path, resized)
        with open(train_txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(plate_text.strip())

    return temp_img_path


def _ctc_greedy_decode(logits: torch.Tensor) -> str:
    log_probs = logits.log_softmax(2)
    _, preds = log_probs.max(2)
    preds = preds.transpose(0, 1)

    seq = preds[0].tolist()
    prev = BLANK_IDX
    chars = []
    for p in seq:
        if p != BLANK_IDX and p != prev:
            chars.append(IDX_TO_CHAR.get(p, ""))
        prev = p
    return "".join(chars)


def run_inference(img_folder: str = TEMP_DIR, out_json: str = OUT_JSON) -> list[dict[str, str]]:
    model = _load_crnn()
    if model is None:
        return []

    results = []
    with torch.no_grad():
        for name in sorted(os.listdir(img_folder)):
            if not name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            path = os.path.join(img_folder, name)
            img = Image.open(path).convert("L")
            img_t = INFER_TRANSFORM(img).unsqueeze(0).to(DEVICE)

            logits = model(img_t)
            plate = _ctc_greedy_decode(logits)
            plate = "".join(ch for ch in plate.upper() if ch.isalnum())[:MAX_VEHICLE_NUMBER_LENGTH]

            results.append({"image": name, "prediction": plate})

    with open(out_json, "w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=2)
    return results


def detect_vehicle_number(image_path: str) -> str | None:
    try:
        if not image_path or not Path(image_path).exists():
            return None

        crop_img = detect_and_crop(image_path)
        if crop_img is None:
            return None

        save_images_and_label(crop_img, "unknown_plate", "")
        predictions = run_inference(TEMP_DIR, OUT_JSON)
        if not predictions:
            return None

        detected = predictions[0].get("prediction", "").strip().upper()
        detected = "".join(ch for ch in detected if ch.isalnum())[:MAX_VEHICLE_NUMBER_LENGTH]
        return detected or None
    except Exception as exc:
        print(f"[ERR] Detection pipeline failed: {exc}")
        return None


if __name__ == "__main__":
    run_inference()
