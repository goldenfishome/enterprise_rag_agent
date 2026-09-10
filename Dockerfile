# Dockerfile
# 多阶段构建，减小镜像体积

FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# 复制依赖
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# 复制项目代码
COPY . .

# 非root用户运行（安全最佳实践）
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# 生产启动：uvicorn多进程
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--log-level", "info"]
