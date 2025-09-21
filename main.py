# Fallback entrypoint for platforms that auto-detect a root-level FastAPI/ASGI app.
# Delegates to app.main.
from app.main import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
