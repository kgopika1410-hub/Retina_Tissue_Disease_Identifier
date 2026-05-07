from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative Diabetic Retinopathy",
]


def crop_black_borders(image: np.ndarray, threshold: int = 7) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    if not np.any(mask):
        return image

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image[y0:y1, x0:x1]


def enhance_image(image: np.ndarray) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 7, 21)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_for_backbone(image: np.ndarray, backbone: str) -> np.ndarray:
    image_255 = image * 255.0
    if backbone == "resnet50":
        image_255 = tf.keras.applications.resnet50.preprocess_input(image_255)
    elif backbone == "vgg16":
        image_255 = tf.keras.applications.vgg16.preprocess_input(image_255)
    else:
        image_255 = tf.keras.applications.efficientnet.preprocess_input(image_255)
    return image_255.astype(np.float32)


def resolve_model_path(root_dir: Path) -> Path:
    env_model_path = os.getenv("MODEL_PATH", "").strip()
    if env_model_path:
        env_path = Path(env_model_path)
        if env_path.exists():
            return env_path

    candidates = [
        root_dir / "outputs_improved" / "best_model.keras",
        root_dir / "outputs_improved" / "final_model.keras",
        root_dir / "outputs_run2" / "best_model.keras",
        root_dir / "outputs_run2" / "final_model.keras",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("No trained model found in outputs_improved or outputs_run2")


_MODEL: tf.keras.Model | None = None
_MODEL_PATH: Path | None = None


def get_model(root_dir: Path) -> tuple[tf.keras.Model, Path]:
    global _MODEL, _MODEL_PATH
    model_path = resolve_model_path(root_dir)
    if _MODEL is None or _MODEL_PATH != model_path:
        _MODEL = tf.keras.models.load_model(model_path)
        _MODEL_PATH = model_path
    return _MODEL, model_path


def prepare_image(
    image_rgb: np.ndarray,
    image_size: int = 224,
    backbone: str = "efficientnetb0",
    use_crop: bool = True,
    use_enhance: bool = False,
) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if use_crop:
        image_bgr = crop_black_borders(image_bgr)

    if use_enhance:
        image_bgr = enhance_image(image_bgr)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image_rgb = image_rgb.astype(np.float32) / 255.0
    image_ready = preprocess_for_backbone(image_rgb, backbone)
    return np.expand_dims(image_ready, axis=0)


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return np.array(pil_image)


def predict_image(
    image_rgb: np.ndarray,
    root_dir: Path,
    use_crop: bool = True,
    use_enhance: bool = False,
    use_tta: bool = True,
    backbone: str = "efficientnetb0",
) -> dict:
    model, model_path = get_model(root_dir)
    x = prepare_image(
        image_rgb=image_rgb,
        image_size=224,
        backbone=backbone,
        use_crop=use_crop,
        use_enhance=use_enhance,
    )

    if use_tta:
        x_hflip = x[:, :, ::-1, :]
        batch = np.concatenate([x, x_hflip], axis=0)
        preds = model.predict(batch, verbose=0)
        probabilities = np.mean(preds, axis=0)
    else:
        probabilities = model.predict(x, verbose=0)[0]

    pred_idx = int(np.argmax(probabilities))
    pred_conf = float(probabilities[pred_idx])

    top2 = np.argsort(probabilities)[::-1][:2]
    second_idx = int(top2[1])
    expected_severity = float(np.sum(np.arange(len(CLASS_NAMES)) * probabilities))
    severity_idx = int(np.clip(np.round(expected_severity), 0, len(CLASS_NAMES) - 1))

    return {
        "predicted_stage": pred_idx,
        "predicted_label": CLASS_NAMES[pred_idx],
        "confidence": pred_conf,
        "second_stage": second_idx,
        "second_label": CLASS_NAMES[second_idx],
        "second_confidence": float(probabilities[second_idx]),
        "severity_stage": severity_idx,
        "severity_label": CLASS_NAMES[severity_idx],
        "probabilities": {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))},
        "model_path": str(model_path),
    }
