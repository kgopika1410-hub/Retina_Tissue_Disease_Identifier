import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import tensorflow as tf


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


def load_model(model_path: Path) -> tf.keras.Model:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return tf.keras.models.load_model(model_path)


ROOT_DIR = Path(__file__).resolve().parent

# Global model variables (lazy loaded)
MODEL = None
MODEL_PATH = None
LOADED_MODEL_PATH = None
BACKBONE = "efficientnetb0"


def resolve_model_path(root_dir: Path) -> Path:
    improved_best_path = root_dir / "outputs_improved" / "best_model.keras"
    improved_final_path = root_dir / "outputs_improved" / "final_model.keras"
    best_path = root_dir / "outputs_run2" / "best_model.keras"
    final_path = root_dir / "outputs_run2" / "final_model.keras"
    if improved_best_path.exists():
        return improved_best_path
    if improved_final_path.exists():
        return improved_final_path
    if best_path.exists():
        return best_path
    if final_path.exists():
        return final_path
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
        print(f"✓ Loaded model: {model_path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to load model: {exc}") from exc


def predict_with_tta(x: np.ndarray) -> np.ndarray:
    # Average predictions across original and horizontally flipped views.
    x_hflip = x[:, :, ::-1, :]
    batch = np.concatenate([x, x_hflip], axis=0)
    preds = MODEL.predict(batch, verbose=0)
    return np.mean(preds, axis=0)


def predict_dr(
    image: np.ndarray,
    use_crop: bool,
    use_enhance: bool,
) -> tuple[str, dict]:
    if image is None:
        return "Please upload an image.", {}

    try:
        ensure_model_loaded()
    except (FileNotFoundError, RuntimeError) as exc:
        return f"Error: {str(exc)}", {}

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

    return message, scores


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

    predict_btn.click(
        fn=predict_dr,
        inputs=[input_image, crop_checkbox, enhance_checkbox],
        outputs=[prediction_text, class_scores],
    )


if __name__ == "__main__":
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    # For Render deployment, bind to 0.0.0.0 to accept external connections
    if os.getenv("RENDER"):
        server_name = "0.0.0.0"
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        show_error=True,
        prevent_thread_lock=False,
        quiet=False,
    )
