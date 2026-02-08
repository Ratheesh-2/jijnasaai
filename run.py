"""Launch both backend (FastAPI) and frontend (Streamlit) processes."""
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path(__file__).parent

    # Ensure data directories exist
    (root / "data" / "vectorstore").mkdir(parents=True, exist_ok=True)
    (root / "data" / "uploads").mkdir(parents=True, exist_ok=True)

    # Detect environment: DOCKER=1 env var or check if running inside a container
    is_docker = os.environ.get("DOCKER", "0") == "1"
    host = "0.0.0.0" if is_docker else "127.0.0.1"

    # Backend: no --reload in production (Docker)
    backend_cmd = [
        sys.executable, "-m", "uvicorn", "backend.main:app",
        "--host", host, "--port", "8000",
    ]
    if not is_docker:
        backend_cmd.append("--reload")

    print(f"Starting backend (FastAPI) on http://{host}:8000 ...")
    backend = subprocess.Popen(backend_cmd, cwd=str(root))

    time.sleep(2)

    print(f"Starting frontend (Streamlit) on http://{host}:8501 ...")
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(root / "frontend" / "app.py"),
         "--server.port", "8501", "--server.address", host],
        cwd=str(root),
    )

    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend.terminate()
        frontend.terminate()
        backend.wait()
        frontend.wait()


if __name__ == "__main__":
    main()
