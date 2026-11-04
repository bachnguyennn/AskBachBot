FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only PyTorch first. The default Linux wheel bundles CUDA libraries (several GB)
# that this CPU-only container never uses. `torch==2.14.1` in requirements.txt is then
# already satisfied by 2.14.1+cpu, so no CUDA packages are pulled.
COPY app/requirements.txt .
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so a cold start doesn't download it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY app/src/ ./src/

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
ENV HF_HUB_OFFLINE=1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
