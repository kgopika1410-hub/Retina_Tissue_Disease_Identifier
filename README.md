# Diabetic Retinopathy Classifier

A deep learning application for detecting and classifying diabetic retinopathy from retinal fundus images.

## Features

- **Web Interface**: Gradio-based UI for easy image upload and prediction
- **REST API**: FastAPI backend for integration with other systems  
- **Multiple Models**: Support for EfficientNet, ResNet50, VGG16 backbones
- **Image Enhancement**: Optional preprocessing for improved predictions
- **TTA (Test Time Augmentation)**: Ensemble predictions for better confidence

## Classification Levels

- No Diabetic Retinopathy
- Mild
- Moderate
- Severe
- Proliferative Diabetic Retinopathy

## Quick Start

### Local Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run Gradio web interface
python dr_inference_app.py
# Visit: http://127.0.0.1:7860

# OR run FastAPI backend
python dr_backend_api.py
# Docs: http://127.0.0.1:8000/docs
```

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for Render deployment instructions.

## Project Structure

```
├── dr_inference_app.py       # Gradio web interface
├── dr_backend_api.py         # FastAPI REST API
├── dr_model_utils.py         # Shared model utilities
├── dr_cross_platform_app.py  # Streamlit alternative UI
├── train_dr_model.py         # Model training script
├── requirements.txt          # Python dependencies
├── outputs_improved/         # Trained models
├── render.yaml              # Render deployment config
└── Procfile                 # Process type definition
```

## Model Files

The trained models (`best_model.keras`, `final_model.keras`) should be placed in:
- `outputs_improved/` (preferred)
- `outputs_run2/` (fallback)

For deployment, upload these files to your Render service or host on cloud storage.

## API Endpoints (FastAPI)

### Health Check
```bash
GET /health
```

### Predict
```bash
POST /predict
Content-Type: multipart/form-data

- file: image file (JPG, PNG)
- use_crop: bool (default: true)
- use_enhance: bool (default: false)
- use_tta: bool (default: true)
```

## Configuration

Environment variables:
- `GRADIO_SERVER_PORT`: Port for Gradio interface (default: 7860)
- `GRADIO_SERVER_NAME`: Host binding (default: 127.0.0.1)
- `MODEL_PATH`: Custom path to model file
- `RENDER`: Set to "true" for Render deployment

## Requirements

- Python 3.9+
- TensorFlow 2.13+
- OpenCV, NumPy, Pandas
- Gradio 4.0+
- FastAPI, Uvicorn

## Training

To train or retrain the model:
```bash
python train_dr_model.py --epochs 100 --batch-size 32
```

## License

[Add your license here]

## References

- APTOS 2019 Blindness Detection Dataset
- EfficientNet: https://arxiv.org/abs/1905.11946
- TensorFlow Keras: https://www.tensorflow.org/
