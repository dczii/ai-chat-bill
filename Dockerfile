FROM python:3.12-slim

# tools needed: curl + unzip
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl unzip \
 && rm -rf /var/lib/apt/lists/*

# non-root user
RUN useradd -m -u 1000 user
USER user
WORKDIR /app

# make sure uv ends up on PATH
ENV PATH="/home/user/.local/bin:/home/user/.venv/bin:${PATH}"

# --- Install uv (no flags!) ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# deps layer
COPY --chown=user pyproject.toml ./
COPY --chown=user static ./static

# chainlit via pip; create venv; sync deps (use uv.lock if present; otherwise resolve)
RUN pip install --user --no-cache-dir chainlit \
 && uv venv \
 && uv sync --frozen || uv sync

# app code
COPY --chown=user . /app

# unzip bills if present (don’t fail if missing)
RUN [ -f /app/bills.zip ] && unzip -o /app/bills.zip -d /app/bills || true

EXPOSE 8080
CMD ["sh", "-c", "uv run chainlit run app.py -h 0.0.0.0 -p ${PORT:-8080} --headless"]
