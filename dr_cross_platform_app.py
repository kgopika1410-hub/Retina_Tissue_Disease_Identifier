from __future__ import annotations

from pathlib import Path
import hashlib

import requests
import streamlit as st

from dr_model_utils import decode_image_bytes, predict_image


ROOT_DIR = Path(__file__).resolve().parent


SEVERITY_TEXT = {
    0: "No visible diabetic retinopathy",
    1: "Mild non-proliferative diabetic retinopathy",
    2: "Moderate non-proliferative diabetic retinopathy",
    3: "Severe non-proliferative diabetic retinopathy",
    4: "Proliferative diabetic retinopathy",
}


def render_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(135deg, #f4fbff 0%, #eef8ff 40%, #f9fcff 100%);
            }
            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 2.0rem;
                max-width: 1200px;
            }
            h1, h2, h3, h4, p, label, span, div {
                color: #12384d;
            }
            .hero-card {
                background: linear-gradient(120deg, #0b6b8f 0%, #118ab2 45%, #3eaed0 100%);
                border: 1px solid #0c6f93;
                border-radius: 14px;
                padding: 1rem 1.2rem;
                margin-bottom: 0.9rem;
                box-shadow: 0 8px 22px rgba(14, 85, 116, 0.18);
            }
            .hero-title {
                color: #ffffff;
                font-size: 1.35rem;
                font-weight: 700;
                margin-bottom: 0.2rem;
            }
            .hero-sub {
                color: #eaf7ff;
                font-size: 0.95rem;
            }
            .section-card {
                border: 1px solid #c8ddea;
                border-radius: 14px;
                padding: 1rem 1.1rem;
                background: #ffffff;
                box-shadow: 0 8px 24px rgba(29, 84, 116, 0.06);
            }
            .upload-card {
                border: 1px solid #c3d9e7;
                border-radius: 16px;
                padding: 1rem 1.1rem;
                background: linear-gradient(180deg, #ffffff 0%, #f3faff 100%);
                box-shadow: 0 10px 24px rgba(22, 92, 129, 0.10);
            }
            .result-card {
                border: 1px solid #bfe0ee;
                border-radius: 12px;
                padding: 0.8rem 1rem;
                background: linear-gradient(180deg, #f0fbff 0%, #f8fcff 100%);
                margin-bottom: 0.8rem;
            }
            .result-title {
                font-size: 1rem;
                color: #0c5d7f;
                font-weight: 700;
            }
            .result-main {
                font-size: 1.05rem;
                color: #0e3950;
                font-weight: 650;
                margin-top: 0.15rem;
            }
            .tiny-note {
                color: #3e6e86;
                font-size: 0.84rem;
                margin-top: 0.45rem;
            }
            .stButton > button {
                border-radius: 10px;
                border: 1px solid #0f7ca6;
                background: linear-gradient(90deg, #118ab2 0%, #1ea3cf 100%);
                color: #ffffff;
                font-weight: 600;
                padding: 0.45rem 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def predict_online(
    image_bytes: bytes,
    backend_url: str,
    use_crop: bool,
    use_enhance: bool,
    use_tta: bool,
) -> dict:
    files = {"file": ("retina_image.jpg", image_bytes, "image/jpeg")}
    data = {
        "use_crop": str(use_crop).lower(),
        "use_enhance": str(use_enhance).lower(),
        "use_tta": str(use_tta).lower(),
    }
    response = requests.post(f"{backend_url.rstrip('/')}/predict", files=files, data=data, timeout=120)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload


def show_result(result: dict) -> None:
    stage = result["predicted_stage"]
    label = result["predicted_label"]
    confidence = result["confidence"]
    secondary_label = result["second_label"]
    secondary_confidence = result["second_confidence"]

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-title">Primary Prediction</div>
            <div class="result-main">Stage {stage}: {label}</div>
            <div class="tiny-note">{SEVERITY_TEXT.get(stage, "Unknown severity")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Primary Confidence", f"{confidence:.2%}")
    with m2:
        st.metric("Secondary Class", secondary_label)
    with m3:
        st.metric("Secondary Confidence", f"{secondary_confidence:.2%}")

    st.caption(
        f"Severity-smoothed estimate: Stage {result['severity_stage']} - {result['severity_label']}"
    )

    st.subheader("Class Probability Distribution")
    st.bar_chart(result["probabilities"])

    if "model_path" in result:
        st.caption(f"Model in use: {result['model_path']}")


def main() -> None:
    st.set_page_config(page_title="DR Cross-Platform Detector", layout="wide")
    render_styles()
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Automated Diabetic Retinopathy Detection</div>
            <div class="hero-sub">
                Cross-platform screening interface for mobile and desktop.
                Upload or capture retinal images and receive stage-level predictions instantly.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Inference Settings")
        mode = st.radio("Mode", ["Offline (Local)", "Online (Backend API)"])
        backend_url = st.text_input("Backend URL", value="http://127.0.0.1:8000")
        use_crop = st.checkbox("Crop black borders", value=True)
        use_enhance = st.checkbox("Image enhancement (slower)", value=False)
        use_tta = st.checkbox("Test-time augmentation", value=True)
        st.markdown("---")
        st.caption("For faster and usually stable results, keep enhancement OFF.")

    col_left, col_right = st.columns([1.2, 0.9], gap="large")
    image_obj = None
    run_prediction = False

    with col_left:
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        st.subheader("Retinal Image Input")
        source = st.radio("Image Source", ["Upload from gallery/files", "Capture from camera"], horizontal=True)
        uploaded = st.file_uploader("Select retinal image", type=["jpg", "jpeg", "png"])
        camera_image = st.camera_input("Capture retinal image") if source == "Capture from camera" else None
        image_obj = camera_image if camera_image is not None else uploaded
        if image_obj is not None:
            st.image(image_obj, caption="Selected retinal image", use_container_width=True)
        run_prediction = st.button("Run Prediction", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Clinical Notes")
        st.write("Prediction provides screening support and does not replace medical diagnosis.")
        st.write("Use high-quality, centered retinal images for better reliability.")
        st.markdown('</div>', unsafe_allow_html=True)

    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "predicted_image_hash" not in st.session_state:
        st.session_state["predicted_image_hash"] = None

    if run_prediction:
        if image_obj is None:
            st.error("Please upload or capture an image first.")
            return

        try:
            image_bytes = image_obj.getvalue()
            image_hash = hashlib.md5(image_bytes).hexdigest()

            if mode == "Offline (Local)":
                image_rgb = decode_image_bytes(image_bytes)
                result = predict_image(
                    image_rgb=image_rgb,
                    root_dir=ROOT_DIR,
                    use_crop=use_crop,
                    use_enhance=use_enhance,
                    use_tta=use_tta,
                )
            else:
                result = predict_online(
                    image_bytes=image_bytes,
                    backend_url=backend_url,
                    use_crop=use_crop,
                    use_enhance=use_enhance,
                    use_tta=use_tta,
                )

            st.session_state["last_result"] = result
            st.session_state["predicted_image_hash"] = image_hash

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    show_results = False
    if image_obj is not None and st.session_state.get("last_result") is not None:
        current_hash = hashlib.md5(image_obj.getvalue()).hexdigest()
        show_results = current_hash == st.session_state.get("predicted_image_hash")

    if show_results:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Prediction Output")
        show_result(st.session_state["last_result"])
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
