# Docker Infrastructure

All-in-one images that bundle an inference server (vLLM or SGLang) with the TALES benchmark suite. The entrypoint starts the server, waits for readiness, and runs the benchmark — no external orchestration needed.

## Quick Start

```bash
# Build (from repo root)
docker build -f docker/Dockerfile -t tales .

# Run a benchmark with a local model
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e MODEL_NAME=Qwen/Qwen3-4B \
  -e ENVS="jericho" \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e WANDB=true \
  tales
```

Or use the helper script (see `run_experiment.sh` in the repo root).

## Images

| Image | Base | Use when |
|-------|------|----------|
| `docker/Dockerfile` | `vllm/vllm-openai:latest` | Default. Works with most models. |
| `docker/Dockerfile.sglang` | `lmsysorg/sglang:latest` | Model is incompatible with vLLM (e.g., MiniMax-M2.7 FP8). |

Build the SGLang variant:

```bash
docker build -f docker/Dockerfile.sglang -t tales-sglang .
```

## Files

- `Dockerfile` — vLLM-based all-in-one image
- `Dockerfile.sglang` — SGLang-based all-in-one image
- `entrypoint.sh` — Unified entrypoint: starts server, waits, runs benchmark

## Environment Variables

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | *(required)* | HuggingFace model ID (e.g., `Qwen/Qwen3-30B-A3B`) |
| `SERVER_TYPE` | `vllm` | Inference backend: `vllm` or `sglang` |
| `SERVER_PORT` | `8000` | Port for the inference server |
| `SERVER_URL` | *(auto)* | Set to skip local server startup and use an external endpoint |
| `SERVER_ARGS` | | Extra CLI args passed to the server |
| `CHAT_TEMPLATE` | *(auto)* | Path to Jinja2 chat template file. Auto-detected for DeepSeek-V4 |
| `SERVER_WAIT_TIMEOUT` | `600` | Abort if server produces no log output for this many seconds |
| `GPU_COUNT` | `1` | Number of GPUs (sets tensor parallelism) |
| `DP_SIZE` | | Data parallel size (TP = GPU_COUNT / DP_SIZE) |
| `MAX_MODEL_LEN` | | Max context length (vLLM: `--max-model-len`, SGLang: `--context-length`) |

### vLLM-Specific

| Variable | Default | Description |
|----------|---------|-------------|
| `REASONING_PARSER` | | Reasoning parser for vLLM (e.g., `deepseek_r1`) |
| `VLLM_MOE_BACKEND` | | MoE backend override (`triton`, `auto`, etc.) |

### Benchmark Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_TYPE` | `zero-shot` | Agent type: `zero-shot` or `reasoning` |
| `AGENT_FILE` | *(auto)* | Agent script path (auto-detected from AGENT_TYPE) |
| `NB_STEPS` | `100` | Max steps per game |
| `CONTEXT_LIMIT` | `100` | Context window limit |
| `ENVS` | *(all)* | Space-separated env/task names |
| `N_SEEDS` | `5` | Number of seeds to run |
| `SEED_PREFIX` | `20241106` | Seed prefix (seeds = prefix+1, prefix+2, ...) |
| `PARALLEL` | `0` | Concurrency: 0=all seeds at once, 1=sequential, N=cap |
| `WANDB` | `false` | Enable WandB logging (`true`/`false`) |
| `CONTINUE_FROM` | | WandB run ID to resume from (or `auto`) |
| `FORCE_FAILED` | `false` | Re-run only failed experiments |
| `REASONING_EFFORT` | | Max thinking tokens for reasoning agent |
| `CONVERSATION` | `true` | Use conversation mode |
| `EXTRA_ARGS` | | Additional args appended to the benchmark command |

### Secrets (via env or mounted files)

| Variable | Description |
|----------|-------------|
| `WANDB_API_KEY` | WandB API key (required if `WANDB=true`) |
| `HF_TOKEN` | HuggingFace token (for gated models) |

### Debugging

| Variable | Default | Description |
|----------|---------|-------------|
| `PAUSE_ON_FAILURE` | `600` | Seconds to keep container alive after failure (for log inspection) |

## Usage Examples

### Run with vLLM (default)

```bash
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e MODEL_NAME=Qwen/Qwen3-30B-A3B \
  -e GPU_COUNT=4 \
  -e ENVS="JerichoEnvZork1 JerichoEnvZork3" \
  -e N_SEEDS=3 \
  -e WANDB=true \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e HF_TOKEN=$HF_TOKEN \
  tales
```

### Run with SGLang

```bash
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e MODEL_NAME=MiniMaxAI/MiniMax-M2.7 \
  -e SERVER_TYPE=sglang \
  -e GPU_COUNT=8 \
  -e DP_SIZE=2 \
  -e ENVS="jericho" \
  -e WANDB=true \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  tales-sglang
```

### Reasoning agent with thinking budget

```bash
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e MODEL_NAME=Qwen/Qwen3-30B-A3B \
  -e AGENT_TYPE=reasoning \
  -e REASONING_EFFORT=1024 \
  -e GPU_COUNT=4 \
  -e ENVS="jericho" \
  tales
```

### Use an external server

```bash
# Point to an already-running vLLM/SGLang instance
docker run \
  -e MODEL_NAME=Qwen/Qwen3-30B-A3B \
  -e SERVER_URL=http://host.docker.internal:8000/v1 \
  -e ENVS="JerichoEnvZork1" \
  tales
```

### Pass custom args directly

```bash
# Override entrypoint env-var logic entirely
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e MODEL_NAME=Qwen/Qwen3-4B \
  tales \
  --agent agents/llm.py zero-shot --llm Qwen/Qwen3-4B --nb-steps 50 --envs JerichoEnvZork1 --seed 1
```

## Architecture

```
┌─────────────────────────────────────────┐
│  Container                              │
│                                         │
│  entrypoint.sh                          │
│    │                                    │
│    ├─ Start server (vLLM or SGLang)     │
│    │    └─ Background process           │
│    │                                    │
│    ├─ Wait for /v1/models endpoint      │
│    │    └─ Activity-based timeout       │
│    │                                    │
│    └─ Run benchmark.py (N seeds × envs) │
│         └─ Parallel work units          │
│                                         │
└─────────────────────────────────────────┘
```
