FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY signalrank ./signalrank
RUN pip install --no-cache-dir .
ENV SIGNALRANK_MODEL=/app/artifacts/model.joblib
COPY artifacts ./artifacts
EXPOSE 8000
CMD ["uvicorn", "signalrank.api:app", "--host", "0.0.0.0", "--port", "8000"]

