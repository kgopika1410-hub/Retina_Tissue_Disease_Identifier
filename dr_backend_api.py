import os
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dr_model_utils import decode_image_bytes, predict_image
from dr_web_core import create_demo


ROOT_DIR = Path(__file__).resolve().parent


app = FastAPI(title="DR Detection API", version="1.0.0")

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


demo = create_demo()
app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    server_port = int(os.getenv("PORT", "8000"))
    uvicorn.run("dr_backend_api:app", host="0.0.0.0", port=server_port, reload=False)
