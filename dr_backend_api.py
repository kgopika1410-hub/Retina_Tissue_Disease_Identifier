from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

from dr_model_utils import decode_image_bytes, predict_image


app = FastAPI(title="DR Detection API", version="1.0.0")
ROOT_DIR = Path(__file__).resolve().parent

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    use_crop: bool = Form(True),
    use_enhance: bool = Form(False),
    use_tta: bool = Form(True),
) -> JSONResponse:
    try:
        content = await file.read()
        image_rgb = decode_image_bytes(content)
        result = predict_image(
            image_rgb=image_rgb,
            root_dir=ROOT_DIR,
            use_crop=use_crop,
            use_enhance=use_enhance,
            use_tta=use_tta,
        )
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.get("/")
def root() -> HTMLResponse:
    html = (
        "<h2>Diabetic Retinopathy Service</h2>"
        "<p>This endpoint serves the API only. The web UI (Gradio) is not being served by the current process.</p>"
        "<p>To expose the web interface at the site root, set the service Start Command to <code>python dr_inference_app.py</code> in the Render dashboard and redeploy.</p>"
        "<p>Or use the API endpoints: <a href=\"/health\">/health</a> and POST /predict (multipart form).</p>"
    )
    return HTMLResponse(content=html, status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("dr_backend_api:app", host="127.0.0.1", port=8000, reload=False)
