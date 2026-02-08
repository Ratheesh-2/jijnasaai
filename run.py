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

    # Railway sets PORT env var — the service MUST listen on this port.
    # We bind Streamlit (the user-facing frontend) to PORT,
    # and FastAPI (the internal backend API) to a fixed internal port.
    railway_port = os.environ.get("PORT", "")
    backend_port = "8000"
    frontend_port = railway_port if railway_port else "8501"

    # Guard against port collision (unlikely but possible on Railway)
    if frontend_port == backend_port:
        backend_port = "8001"

    # Startup banner -- visible in Railway deploy logs to confirm run.py is executing
    print("=" * 60)
    print("  JijnasaAI -- run.py starting")
    print(f"  DOCKER={os.environ.get('DOCKER', '(not set)')}")
    print(f"  PORT={railway_port or '(not set, using default 8501)'}")
    print(f"  Backend  (FastAPI)   -> http://{host}:{backend_port}")
    print(f"  Frontend (Streamlit) -> http://{host}:{frontend_port}")
    print("=" * 60)

    # Backend: no --reload in production (Docker)
    backend_cmd = [
        sys.executable, "-m", "uvicorn", "backend.main:app",
        "--host", host, "--port", backend_port,
    ]
    if not is_docker:
        backend_cmd.append("--reload")

    print(f"Starting backend (FastAPI) on http://{host}:{backend_port} ...")
    backend = subprocess.Popen(backend_cmd, cwd=str(root))

    time.sleep(2)

    print(f"Starting frontend (Streamlit) on http://{host}:{frontend_port} ...")
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(root / "frontend" / "app.py"),
         "--server.port", frontend_port, "--server.address", host],
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
