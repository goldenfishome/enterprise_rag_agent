# Dockerfile
# Multi-stage build to reduce image size

FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy dependencies
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy project code
COPY . .

# Run as a non-root user (security best practice)
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Production startup with multiple Uvicorn workers
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--log-level", "info"]
