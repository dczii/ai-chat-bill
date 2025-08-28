# ---------- builder ----------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
# uv + python are preinstalled

# Non-root for reproducibility
RUN useradd -m -u 1000 user
USER user
WORKDIR /app

# Copy only lock/manifest to leverage layer caching
# If you have uv.lock, copy it too for reproducible & smaller installs
COPY --chown=user pyproject.toml uv.lock* ./

# Create project venv and sync deps (no dev, no editable)
ENV PATH="/home/user/.venv/bin:${PATH}"
RUN uv venv && uv sync --frozen --no-dev || uv sync --no-dev

# Wipe build caches to keep the final copy small
RUN rm -rf /home/user/.cache/*

# ---------- runtime ----------
FROM python:3.12-slim AS runtime

# Minimal tools only if you need to unzip at runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends unzip \
 && rm -rf /var/lib/apt/lists/*

# Non-root
RUN useradd -m -u 1000 user
USER user
WORKDIR /app

# Put the venv on PATH
ENV PATH="/home/user/.venv/bin:${PATH}"

# Copy the resolved, compact venv from builder
# COPY --from=builder --chown=user /home/user/.venv /home/user/.venv

# Copy only what you need to run
# (Avoid copying big data, node_modules, .git, etc.)
COPY --chown=user app.py ./app.py
# Optional: unzip bills.zip at *runtime start* if present (no layer cost).
# Create a tiny entrypoint that unzips if found, then launches Chainlit.
COPY --chown=user <<'SH' /app/entrypoint.sh
#!/usr/bin/env sh
set -eu
if [ -f /app/bills.zip ]; then
  mkdir -p /app/bills
  unzip -o /app/bills.zip -d /app/bills >/dev/null 2>&1 || true
fi
exec uv run chainlit run /app/app.py -h 0.0.0.0 -p "${PORT:-8080}" --headless
SH
RUN chmod +x /app/entrypoint.sh

EXPOSE 8080
CMD ["/app/entrypoint.sh"]
