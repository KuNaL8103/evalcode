# evalcode sandbox image: runs generated pytest files in an isolated container.
# Build once:  docker build -t evalcode-sandbox:latest .
# The agent only bind-mounts /work (solution.py + test_solution.py) at runtime.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Unprivileged user: the runtime keeps --cap-drop=ALL and no-new-privileges.
RUN useradd --create-home --uid 10001 sandbox \
    && pip install --no-cache-dir "pytest>=8,<9"

WORKDIR /work
USER sandbox

# The host supplies the real command (python -m pytest ...); this is a safe default.
CMD ["python", "-m", "pytest", "--version"]
