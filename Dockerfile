# Python 3.12 to satisfy requires-python >=3.12 from pyproject
FROM python:3.12-slim

# Make uv behave well in containers
ENV UV_LINK_MODE=copy \
    UV_PYTHON=python3 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/root/.local/bin:$PATH"

# System deps: curl (for uv installer), unzip (to extract bills.zip)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl unzip && \
    rm -rf /var/lib/apt/lists/*

# Install uv (https://astral.sh/uv)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv

# App files
WORKDIR /app

# Sync Python deps first for better Docker layer caching
COPY pyproject.toml /app/pyproject.toml
# If you have a uv.lock, copy it too for reproducible builds:
# COPY uv.lock /app/uv.lock
RUN uv sync --no-dev --frozen

# Bring in the rest of the code (including app.py, bills.zip, static/, etc.)
COPY . /app

# Extract bills.zip into /app/bills if it exists
RUN mkdir -p /app/bills && \
    if [ -f /app/bills.zip ]; then unzip -o /app/bills.zip -d /app/bills; fi

# Heroku provides $PORT; use Chainlit headless mode
# NOTE: use sh -c so $PORT expands
CMD ["sh", "-c", "uv run chainlit run app.py -h 0.0.0.0 -p $PORT --headless"]
