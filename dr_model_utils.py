from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path

import cv2
import numpy as np
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
    # Lazy-import TensorFlow to avoid heavy imports at module import time
    import tensorflow as tf

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

    # Default priority: final_model first, then best_model.
    candidates = [
        root_dir / "outputs_improved" / "final_model.keras",
        root_dir / "outputs_improved" / "best_model.keras",
        root_dir / "outputs_run2" / "final_model.keras",
        root_dir / "outputs_run2" / "best_model.keras",
        root_dir / "outputs" / "final_model.keras",
        root_dir / "outputs" / "best_model.keras",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("No trained model found in outputs_improved or outputs_run2")


_MODEL: tf.keras.Model | None = None
_MODEL_PATH: Path | None = None


def get_model(root_dir: Path) -> tuple["tf.keras.Model", Path]:
    # Lazy-import TensorFlow and load model with compile=False for Keras 3 compatibility
    global _MODEL, _MODEL_PATH
    import tensorflow as tf
    try:
        from keras.src.models.functional import Functional
    except Exception:
        Functional = None

    def _strip_renorm(config: dict) -> dict:
        cfg = dict(config)
        cfg.pop("renorm", None)
        cfg.pop("renorm_clipping", None)
        cfg.pop("renorm_momentum", None)
        return cfg

    def _strip_quantization(config: dict) -> dict:
        cfg = dict(config)
        cfg.pop("quantization_config", None)
        return cfg

    def _patch_layer_quantization_deserialization() -> None:
        dense_classes = [tf.keras.layers.Dense]
        try:
            from keras.src.layers.core.dense import Dense as KerasSrcDense

            dense_classes.append(KerasSrcDense)
        except Exception:
            pass

        seen = set()
        for dense_cls in dense_classes:
            if dense_cls in seen:
                continue
            seen.add(dense_cls)

            if getattr(dense_cls, "_quant_patch_applied", False):
                continue

            original_init = dense_cls.__init__
            original_from_config = dense_cls.from_config.__func__

            def patched_init(self, *args, __orig_init=original_init, **kwargs):
                kwargs.pop("quantization_config", None)
                return __orig_init(self, *args, **kwargs)

            def patched_from_config(cls, config, __orig_from_config=original_from_config):
                return __orig_from_config(cls, _strip_quantization(config))

            dense_cls.__init__ = patched_init
            dense_cls.from_config = classmethod(patched_from_config)
            dense_cls._quant_patch_applied = True

    def _patch_batchnorm_deserialization() -> None:
        bn_classes = [tf.keras.layers.BatchNormalization]
        try:
            from keras.src.layers.normalization.batch_normalization import BatchNormalization as KerasSrcBN

            bn_classes.append(KerasSrcBN)
        except Exception:
            pass

        seen = set()
        for bn_cls in bn_classes:
            if bn_cls in seen:
                continue
            seen.add(bn_cls)

            if getattr(bn_cls, "_renorm_patch_applied", False):
                continue

            original_init = bn_cls.__init__
            original_from_config = bn_cls.from_config.__func__

            def patched_init(self, *args, __orig_init=original_init, **kwargs):
                kwargs.pop("renorm", None)
                kwargs.pop("renorm_clipping", None)
                kwargs.pop("renorm_momentum", None)
                return __orig_init(self, *args, **kwargs)

            def patched_from_config(cls, config, __orig_from_config=original_from_config):
                return __orig_from_config(cls, _strip_renorm(config))

            bn_cls.__init__ = patched_init
            bn_cls.from_config = classmethod(patched_from_config)
            bn_cls._renorm_patch_applied = True

    model_path = resolve_model_path(root_dir)
    if _MODEL is None or _MODEL_PATH != model_path:
        _patch_batchnorm_deserialization()
        _patch_layer_quantization_deserialization()

        class CompatBatchNormalization(tf.keras.layers.BatchNormalization):
            def __init__(self, *args, **kwargs):
                kwargs.pop("renorm", None)
                kwargs.pop("renorm_clipping", None)
                kwargs.pop("renorm_momentum", None)
                super().__init__(*args, **kwargs)

            @classmethod
            def from_config(cls, config):
                return super().from_config(_strip_renorm(config))

        class CompatDense(tf.keras.layers.Dense):
            def __init__(self, *args, **kwargs):
                kwargs.pop("quantization_config", None)
                super().__init__(*args, **kwargs)

            @classmethod
            def from_config(cls, config):
                return super().from_config(_strip_quantization(config))

        custom_objects = {
            "BatchNormalization": CompatBatchNormalization,
            "keras.layers.BatchNormalization": CompatBatchNormalization,
            "keras.src.layers.normalization.batch_normalization.BatchNormalization": CompatBatchNormalization,
            "Dense": CompatDense,
            "keras.layers.Dense": CompatDense,
            "keras.src.layers.core.dense.Dense": CompatDense,
        }
        if Functional is not None:
            custom_objects["Functional"] = Functional
            custom_objects["keras.src.models.functional.Functional"] = Functional
            custom_objects["keras.models.Functional"] = Functional

        attempts = [
            {"compile": False},
            {"compile": False, "safe_mode": False},
            {"compile": False, "custom_objects": custom_objects},
            {"compile": False, "custom_objects": custom_objects, "safe_mode": False},
        ]

        last_exc = None
        for kwargs in attempts:
            try:
                _MODEL = tf.keras.models.load_model(model_path, **kwargs)
                break
            except TypeError as exc:
                last_exc = exc
            except Exception as exc:
                last_exc = exc

        if _MODEL is None:
            try:
                import tf_keras as legacy_keras

                class LegacyCompatBatchNormalization(legacy_keras.layers.BatchNormalization):
                    def __init__(self, *args, **kwargs):
                        kwargs.pop("renorm", None)
                        kwargs.pop("renorm_clipping", None)
                        kwargs.pop("renorm_momentum", None)
                        super().__init__(*args, **kwargs)

                    @classmethod
                    def from_config(cls, config):
                        return super().from_config(_strip_renorm(config))

                class LegacyCompatDense(legacy_keras.layers.Dense):
                    def __init__(self, *args, **kwargs):
                        kwargs.pop("quantization_config", None)
                        super().__init__(*args, **kwargs)

                    @classmethod
                    def from_config(cls, config):
                        return super().from_config(_strip_quantization(config))

                legacy_custom_objects = {
                    "BatchNormalization": LegacyCompatBatchNormalization,
                    "keras.layers.BatchNormalization": LegacyCompatBatchNormalization,
                    "keras.src.layers.normalization.batch_normalization.BatchNormalization": LegacyCompatBatchNormalization,
                    "Dense": LegacyCompatDense,
                    "keras.layers.Dense": LegacyCompatDense,
                    "keras.src.layers.core.dense.Dense": LegacyCompatDense,
                }
                if Functional is not None:
                    legacy_custom_objects["Functional"] = Functional
                    legacy_custom_objects["keras.src.models.functional.Functional"] = Functional

                _MODEL = legacy_keras.models.load_model(
                    model_path,
                    compile=False,
                    custom_objects=legacy_custom_objects,
                )
            except Exception as exc:
                last_exc = exc

        if _MODEL is None:
            raise RuntimeError(f"Failed to load model from {model_path}: {last_exc}")
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
