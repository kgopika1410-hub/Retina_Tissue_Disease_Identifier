import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np


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


def load_model(model_path: Path) -> "tf.keras.Model":
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    # Import TensorFlow lazily so importing this module doesn't require TF at startup
    import tensorflow as tf
    try:
        from keras.src.models.functional import Functional
    except Exception:
        Functional = None
    import traceback

    def _strip_renorm(config: dict) -> dict:
        cfg = dict(config)
        cfg.pop("renorm", None)
        cfg.pop("renorm_clipping", None)
        cfg.pop("renorm_momentum", None)
        return cfg

    def _patch_batchnorm_deserialization() -> None:
        # Some legacy models persist renorm fields; newer Keras rejects those kwargs.
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

    print(f"Attempting to load model from {model_path}")
    try:
        print("TensorFlow version:", tf.__version__, "Keras version:", tf.keras.__version__)
    except Exception:
        pass

    _patch_batchnorm_deserialization()

    class CompatBatchNormalization(tf.keras.layers.BatchNormalization):
        def __init__(self, *args, **kwargs):
            kwargs.pop("renorm", None)
            kwargs.pop("renorm_clipping", None)
            kwargs.pop("renorm_momentum", None)
            super().__init__(*args, **kwargs)

        @classmethod
        def from_config(cls, config):
            return super().from_config(_strip_renorm(config))

    custom_objects = {
        "BatchNormalization": CompatBatchNormalization,
        "keras.layers.BatchNormalization": CompatBatchNormalization,
        "keras.src.layers.normalization.batch_normalization.BatchNormalization": CompatBatchNormalization,
    }
    if Functional is not None:
        custom_objects["Functional"] = Functional
        custom_objects["keras.src.models.functional.Functional"] = Functional
        custom_objects["keras.models.Functional"] = Functional

    attempts = [
        ("standard", {"compile": False}),
        ("safe_mode_false", {"compile": False, "safe_mode": False}),
        ("compat_objects", {"compile": False, "custom_objects": custom_objects}),
        (
            "compat_objects_safe_mode_false",
            {"compile": False, "custom_objects": custom_objects, "safe_mode": False},
        ),
    ]

    last_exc = None
    for attempt_name, kwargs in attempts:
        try:
            print(f"Attempting model load using {attempt_name}")
            return tf.keras.models.load_model(model_path, **kwargs)
        except TypeError as exc:
            # Older/newer Keras builds may not accept safe_mode; try the next option.
            print(f"Model load attempt '{attempt_name}' rejected kwargs:", exc)
            last_exc = exc
        except Exception as exc:
            print(f"Model load attempt '{attempt_name}' failed:", exc)
            traceback.print_exc()
            last_exc = exc

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

        legacy_custom_objects = {
            "BatchNormalization": LegacyCompatBatchNormalization,
            "keras.layers.BatchNormalization": LegacyCompatBatchNormalization,
            "keras.src.layers.normalization.batch_normalization.BatchNormalization": LegacyCompatBatchNormalization,
        }
        if Functional is not None:
            legacy_custom_objects["Functional"] = Functional
            legacy_custom_objects["keras.src.models.functional.Functional"] = Functional

        print("Attempting model load using tf_keras compatibility path")
        return legacy_keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=legacy_custom_objects,
        )
    except Exception as exc:
        print("Model load attempt 'tf_keras' failed:", exc)
        traceback.print_exc()
        last_exc = exc

    raise RuntimeError(f"All model load attempts failed for {model_path}: {last_exc}")


ROOT_DIR = Path(__file__).resolve().parent

# Global model variables (lazy loaded)
MODEL = None
MODEL_PATH = None
LOADED_MODEL_PATH = None
BACKBONE = "efficientnetb0"


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
    return None


def ensure_model_loaded() -> None:
    """Load model on first use (lazy loading)."""
    global MODEL, MODEL_PATH, LOADED_MODEL_PATH

    if MODEL is not None:
        return

    model_path = resolve_model_path(ROOT_DIR)
    if model_path is None:
        raise FileNotFoundError(
            "No trained model found in outputs_improved or outputs_run2. "
            "Please upload the model files to the service."
        )

    try:
        MODEL = load_model(model_path)
        MODEL_PATH = model_path
        LOADED_MODEL_PATH = model_path
        print(f"Loaded model: {model_path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to load model: {exc}") from exc


def predict_with_tta(x: np.ndarray) -> np.ndarray:
    # Average predictions across original and horizontally flipped views.
    x_hflip = x[:, :, ::-1, :]
    batch = np.concatenate([x, x_hflip], axis=0)
    preds = MODEL.predict(batch, verbose=0)
    return np.mean(preds, axis=0)


def build_analysis_text(probabilities: np.ndarray, pred_idx: int, severity_idx: int, model_path: Path) -> str:
    ordered = np.argsort(probabilities)[::-1]
    lines = [
        f"### Analysis",
        f"Most likely class: **{CLASS_NAMES[pred_idx]}** ({probabilities[pred_idx] * 100:.1f}%)",
        f"Estimated severity: **{CLASS_NAMES[severity_idx]}**",
        f"Model used: `{model_path.name}`",
        "",
        "#### Class probabilities",
    ]

    for idx in ordered:
        lines.append(f"- {CLASS_NAMES[idx]}: {probabilities[idx] * 100:.1f}%")

    return "\n".join(lines)


def predict_dr(
    image: np.ndarray,
    use_crop: bool,
    use_enhance: bool,
) -> tuple[str, dict[str, float], str]:
    if image is None:
        return "Please upload an image.", {}, ""

    try:
        ensure_model_loaded()
    except (FileNotFoundError, RuntimeError) as exc:
        return f"Error: {str(exc)}", {}, f"### Analysis\n\n{str(exc)}"

    x = prepare_image(
        image_rgb=image,
        image_size=224,
        backbone=BACKBONE,
        use_crop=use_crop,
        use_enhance=use_enhance,
    )

    probabilities = predict_with_tta(x)
    pred_idx = int(np.argmax(probabilities))
    pred_label = CLASS_NAMES[pred_idx]
    pred_conf = float(probabilities[pred_idx])

    expected_severity = float(np.sum(np.arange(len(CLASS_NAMES)) * probabilities))
    severity_idx = int(np.clip(np.round(expected_severity), 0, len(CLASS_NAMES) - 1))

    top2 = np.argsort(probabilities)[::-1][:2]
    second_label = CLASS_NAMES[int(top2[1])]
    second_conf = float(probabilities[int(top2[1])])

    scores = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
    analysis_text = build_analysis_text(probabilities, pred_idx, severity_idx, LOADED_MODEL_PATH or MODEL_PATH or resolve_model_path(ROOT_DIR))

    if pred_conf < 0.55:
        message = (
            f"Predicted Level (low confidence): {CLASS_NAMES[severity_idx]} "
            f"(top class: {pred_label}, confidence: {pred_conf:.4f}; "
            f"second class: {second_label}, confidence: {second_conf:.4f})"
        )
    else:
        message = (
            f"Predicted Level: {pred_label} (confidence: {pred_conf:.4f}; "
            f"second class: {second_label}, confidence: {second_conf:.4f})"
        )

    return message, scores, analysis_text


def create_demo() -> gr.Blocks:
    with gr.Blocks(title="Diabetic Retinopathy Classifier") as demo:
        gr.Markdown(
            "## Diabetic Retinopathy Detection\n"
            "Upload a retinal fundus image to predict disease severity.\n"
            "Tip: keep image enhancement OFF for faster and usually more stable predictions."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="numpy", label="Upload Retinal Image")
                crop_checkbox = gr.Checkbox(value=True, label="Crop black borders")
                enhance_checkbox = gr.Checkbox(value=False, label="Apply image enhancement (slower)")
                predict_btn = gr.Button("Predict", variant="primary")

            with gr.Column(scale=1):
                prediction_text = gr.Textbox(label="Prediction", interactive=False)
                class_scores = gr.Label(label="Class Probabilities")

        with gr.Row():
            analysis_md = gr.Markdown(value="### Analysis\nNo prediction yet.", label="Analysis")

        predict_btn.click(
            fn=predict_dr,
            inputs=[input_image, crop_checkbox, enhance_checkbox],
            outputs=[prediction_text, class_scores, analysis_md],
        )

    return demo


demo = create_demo()