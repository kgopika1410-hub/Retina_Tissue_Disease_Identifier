if __name__ == "__main__":
    import os
    import sys
    import traceback

    import uvicorn

    port_env = os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860"))
    try:
        server_port = int(port_env)
    except Exception:
        server_port = 7860

    print(
        f"Starting uvicorn for dr_backend_api:app on 0.0.0.0:{server_port} (PORT={os.getenv('PORT')}, GRADIO_SERVER_PORT={os.getenv('GRADIO_SERVER_PORT')})"
    )
    try:
        uvicorn.run("dr_backend_api:app", host="0.0.0.0", port=server_port, reload=False)
    except Exception:
        print("Unhandled exception while starting server:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
