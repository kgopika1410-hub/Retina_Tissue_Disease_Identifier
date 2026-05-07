if __name__ == "__main__":
    import os

    import uvicorn

    server_port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    uvicorn.run("dr_backend_api:app", host="0.0.0.0", port=server_port, reload=False)
