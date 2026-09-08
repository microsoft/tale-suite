FROM vllm/vllm-openai:latest

USER root

# System dependencies (Java needed for ScienceWorld and TextWorldExpress)
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre-headless \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install Python dependencies first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the project
COPY . .
RUN pip install --no-cache-dir -e .

# Validate all frameworks load
RUN python -c "import tales; print(f'Loaded {len(tales.tasks)} tasks, {len(tales.envs)} envs'); assert len(tales.envs) > 0"

# Download game data at build time
RUN python -m tales.download

# Entrypoint handles: start vLLM, wait, run benchmark
RUN chmod +x /app/docker/entrypoint.sh
ENTRYPOINT ["/app/docker/entrypoint.sh"]
