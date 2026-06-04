# ── Base image ─────────────────────────────────────────────
FROM python:3.10-slim

# ── System dependencies ─────────────────────────────────────
RUN apt-get update && apt-get install -y \
    openslide-tools \
    libopenslide-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ───────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ─────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=120 --retries=5 -r requirements.txt

# ── Copy project code and assets ────────────────────────────
COPY src/ ./src/
COPY .streamlit/ ./.streamlit/
COPY outputs/models/ ./outputs/models/
COPY outputs/results/ ./outputs/results/
COPY data/metadata/ ./data/metadata/
COPY data/features/ ./data/features/
COPY tests/ ./tests/

# ── Expose Streamlit port ───────────────────────────────────
EXPOSE 8501

# ── Healthcheck ─────────────────────────────────────────────
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ── Run dashboard ───────────────────────────────────────────
CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]