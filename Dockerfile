FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home --uid 10001 appuser
ENV HF_HOME=/home/appuser/.cache/huggingface
RUN mkdir -p "$HF_HOME" && chown -R appuser:appuser /home/appuser /app
COPY --chown=appuser:appuser app app
COPY --chown=appuser:appuser inference inference
COPY --chown=appuser:appuser scheduler scheduler
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
