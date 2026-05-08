import os
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dr_model_utils import decode_image_bytes, predict_image
from dr_web_core import create_demo, ensure_model_loaded


ROOT_DIR = Path(__file__).resolve().parent


app = FastAPI(title="DR Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.state.model_loaded = False


@app.on_event("startup")
def warm_model() -> None:
    try:
        ensure_model_loaded()
        app.state.model_loaded = True
    except Exception:
        app.state.model_loaded = False


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/status")
def status() -> JSONResponse:
    import subprocess
    import tensorflow as tf

    from dr_model_utils import resolve_model_path
    import dr_web_core

    commit = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        commit = None

    model_path = None
    model_loaded = False
    model_available = False
    model_state = "unknown"
    try:
        resolved_model_path = resolve_model_path(ROOT_DIR)
        model_available = resolved_model_path.exists()
        model_path = str(resolved_model_path)
    except Exception:
        model_path = None

    try:
        model_loaded = bool(app.state.model_loaded or dr_web_core.MODEL is not None)
        if model_loaded:
            model_state = "loaded"
        elif model_available:
            model_state = "available_not_loaded"
        else:
            model_state = "missing"
    except Exception:
        model_loaded = False
        model_state = "unknown"

    info = {
        "commit": commit,
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "model_path": model_path,
        "model_available": model_available,
        "model_loaded": model_loaded,
        "model_state": model_state,
    }
    return JSONResponse(info)


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
        # attempt to surface which model is loaded for diagnostics
        try:
            from dr_web_core import LOADED_MODEL_PATH
            loaded_path = str(LOADED_MODEL_PATH) if LOADED_MODEL_PATH is not None else None
        except Exception:
            loaded_path = None

        result = predict_image(
            image_rgb=image_rgb,
            root_dir=ROOT_DIR,
            use_crop=use_crop,
            use_enhance=use_enhance,
            use_tta=use_tta,
        )
        # attach diagnostics for easier debugging of deployment mismatches
        if isinstance(result, dict):
            result.setdefault("_debug", {})
            result["_debug"]["loaded_model_path"] = loaded_path
            result["_debug"]["model_candidates_env"] = os.getenv("MODEL_PATH", "")
            result["_debug"]["model_loaded"] = bool(loaded_path)
            result["_debug"]["backend_warmed"] = bool(app.state.model_loaded)
        return JSONResponse(result)
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        return JSONResponse({"error": str(exc), "traceback": tb}, status_code=400)


demo = create_demo()
app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    server_port = int(os.getenv("PORT", "8000"))
    uvicorn.run("dr_backend_api:app", host="0.0.0.0", port=server_port, reload=False)
