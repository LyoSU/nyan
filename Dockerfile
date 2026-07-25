# Build the dependencies in a separate stage so the compilers they need do not
# ship in the final image.
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


FROM python:3.12-slim

# curl is what download_models.sh needs at runtime; the entrypoint fetches the
# models when the mounted volume is empty. tzdata is required for
# ZoneInfo("Europe/Kyiv") — the slim image ships no timezone database, and
# without it every reader-facing time would silently fall back to UTC.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY . .

# Models are not baked into the image: docker-compose mounts ./models over this
# path anyway, so downloading them here would only make the image bigger.
RUN mkdir -p data models \
    && chmod +x *.sh \
    && cp docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
