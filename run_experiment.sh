#!/bin/bash
# Run TALES benchmark experiment using local Docker.
# Builds and runs the all-in-one container with the specified model and environments.
#
# Usage:
#   ./run_experiment.sh --model Qwen/Qwen3-4B --envs JerichoEnvZork1 --gpu-count 1
#   ./run_experiment.sh --model Qwen/Qwen3-30B-A3B --all --gpu-count 4 --wandb
#   ./run_experiment.sh --model MiniMaxAI/MiniMax-M2.7 --server sglang --gpu-count 8 --all
#
# Prerequisites:
#   - Docker with NVIDIA GPU support (nvidia-container-toolkit)
#   - HuggingFace cache at ~/.cache/huggingface (or set --hf-cache)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
NB_STEPS=100
N_SEEDS=5
SEED_PREFIX=20241106
AGENT_TYPE="zero-shot"
CONTEXT_LIMIT=100
GPU_COUNT=1
WANDB=false
WANDB_PROJECT="${WANDB_PROJECT:-tales}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
REASONING_PARSER=""
REASONING_EFFORT=""
MAX_MODEL_LEN=""
DP_SIZE=""
MOE_BACKEND=""
SERVER_TYPE="vllm"
SERVER_ARGS=""
FORCE_FAILED=false
CONTINUE_FROM=""
PARALLEL=0
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"
DOCKER_IMAGE=""
BUILD=true
DETACH=false
EXTRA_DOCKER_ARGS=""

# Parse args
ENVS=""
MODEL_NAME=""
ALL_ENVS=false

usage() {
    echo "Usage: $0 --model MODEL [options]"
    echo ""
    echo "Required:"
    echo "  --model MODEL          HuggingFace model name"
    echo ""
    echo "Environment selection (one required):"
    echo "  --envs ENV [ENV...]    Specific game environments"
    echo "  --framework FW         All games in a framework (jericho, scienceworld, etc.)"
    echo "  --all                  All 122 games"
    echo ""
    echo "Benchmark options:"
    echo "  --nb-steps N           Max steps per game (default: 100)"
    echo "  --n-seeds N            Number of seeds (default: 5)"
    echo "  --seed-prefix PREFIX   Seed prefix (default: 20241106)"
    echo "  --agent-type TYPE      Agent type: zero-shot, reasoning (default: zero-shot)"
    echo "  --context N            Context limit (default: 100)"
    echo "  --reasoning-effort E   Max thinking tokens for reasoning agent"
    echo "  --parallel N           Concurrency (0=all seeds, 1=sequential, N=cap)"
    echo "  --continue-from ID     Resume from WandB run (or 'auto')"
    echo "  --force-failed         Re-run only failed experiments"
    echo ""
    echo "Server options:"
    echo "  --server TYPE          Inference server: vllm or sglang (default: vllm)"
    echo "  --gpu-count N          Number of GPUs (default: 1)"
    echo "  --dp-size N            Data parallel size (TP = gpu-count / dp-size)"
    echo "  --max-model-len N      Max context length for the server"
    echo "  --moe-backend BACKEND  vLLM MoE backend (triton, auto)"
    echo "  --server-args ARGS     Extra CLI args for the inference server"
    echo "  --reasoning-parser P   vLLM reasoning parser (e.g., deepseek_r1)"
    echo ""
    echo "Docker options:"
    echo "  --docker-image IMG     Use specific image (skip build)"
    echo "  --no-build             Don't rebuild image"
    echo "  --hf-cache PATH        HuggingFace cache dir (default: ~/.cache/huggingface)"
    echo "  --detach               Run container in background"
    echo ""
    echo "Logging:"
    echo "  --wandb                Enable WandB logging"
    echo "  --no-wandb             Disable WandB logging (default)"
    echo ""
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_NAME="$2"; shift 2 ;;
        --envs) shift; while [[ $# -gt 0 && ! "$1" == --* ]]; do ENVS="${ENVS} $1"; shift; done; ENVS="${ENVS# }" ;;
        --framework) ENVS="$2"; shift 2 ;;
        --all) ALL_ENVS=true; shift ;;
        --nb-steps) NB_STEPS="$2"; shift 2 ;;
        --n-seeds) N_SEEDS="$2"; shift 2 ;;
        --seed-prefix) SEED_PREFIX="$2"; shift 2 ;;
        --agent-type) AGENT_TYPE="$2"; shift 2 ;;
        --context) CONTEXT_LIMIT="$2"; shift 2 ;;
        --gpu-count) GPU_COUNT="$2"; shift 2 ;;
        --dp-size) DP_SIZE="$2"; shift 2 ;;
        --docker-image) DOCKER_IMAGE="$2"; BUILD=false; shift 2 ;;
        --no-build) BUILD=false; shift ;;
        --reasoning-parser) REASONING_PARSER="$2"; shift 2 ;;
        --reasoning-effort) REASONING_EFFORT="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --moe-backend) MOE_BACKEND="$2"; shift 2 ;;
        --server) SERVER_TYPE="$2"; shift 2 ;;
        --server-args) SERVER_ARGS="$2"; shift 2 ;;
        --continue-from) CONTINUE_FROM="$2"; shift 2 ;;
        --force-failed) FORCE_FAILED=true; shift ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --wandb) WANDB=true; shift ;;
        --no-wandb) WANDB=false; shift ;;
        --hf-cache) HF_CACHE="$2"; shift 2 ;;
        --detach) DETACH=true; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "$MODEL_NAME" ]; then
    echo "ERROR: --model is required"
    usage
fi

# Validate server type
if [ "$SERVER_TYPE" != "vllm" ] && [ "$SERVER_TYPE" != "sglang" ]; then
    echo "ERROR: --server must be 'vllm' or 'sglang', got '${SERVER_TYPE}'"
    exit 1
fi

# Resolve environments
if [ "$ALL_ENVS" = true ]; then
    ENVS="jericho scienceworld textworld textworld_express alfworld"
fi

if [ -z "$ENVS" ]; then
    echo "ERROR: No environments specified. Use --envs, --framework, or --all"
    usage
fi

# Determine Docker image
if [ -z "$DOCKER_IMAGE" ]; then
    if [ "$SERVER_TYPE" = "sglang" ]; then
        DOCKER_IMAGE="tales-sglang"
    else
        DOCKER_IMAGE="tales"
    fi
fi

# Build image if requested
DOCKERFILE="docker/Dockerfile"
if [ "$SERVER_TYPE" = "sglang" ]; then
    DOCKERFILE="docker/Dockerfile.sglang"
fi

if [ "$BUILD" = true ]; then
    echo "Building ${DOCKER_IMAGE} from ${DOCKERFILE}..."
    docker build -f "$SCRIPT_DIR/$DOCKERFILE" -t "$DOCKER_IMAGE" "$SCRIPT_DIR"
    echo ""
fi

# Build docker run command
DOCKER_CMD="docker run --rm"

if [ "$DETACH" = true ]; then
    DOCKER_CMD="${DOCKER_CMD} -d"
else
    DOCKER_CMD="${DOCKER_CMD} -it"
fi

# GPU access
DOCKER_CMD="${DOCKER_CMD} --gpus all"

# Mount HuggingFace cache
DOCKER_CMD="${DOCKER_CMD} -v ${HF_CACHE}:/root/.cache/huggingface"

# Environment variables
DOCKER_CMD="${DOCKER_CMD} -e MODEL_NAME=${MODEL_NAME}"
DOCKER_CMD="${DOCKER_CMD} -e SERVER_TYPE=${SERVER_TYPE}"
DOCKER_CMD="${DOCKER_CMD} -e GPU_COUNT=${GPU_COUNT}"
DOCKER_CMD="${DOCKER_CMD} -e NB_STEPS=${NB_STEPS}"
DOCKER_CMD="${DOCKER_CMD} -e N_SEEDS=${N_SEEDS}"
DOCKER_CMD="${DOCKER_CMD} -e SEED_PREFIX=${SEED_PREFIX}"
DOCKER_CMD="${DOCKER_CMD} -e AGENT_TYPE=${AGENT_TYPE}"
DOCKER_CMD="${DOCKER_CMD} -e CONTEXT_LIMIT=${CONTEXT_LIMIT}"
DOCKER_CMD="${DOCKER_CMD} -e ENVS=${ENVS}"
DOCKER_CMD="${DOCKER_CMD} -e PARALLEL=${PARALLEL}"
DOCKER_CMD="${DOCKER_CMD} -e CONVERSATION=true"

if [ -n "$DP_SIZE" ]; then
    DOCKER_CMD="${DOCKER_CMD} -e DP_SIZE=${DP_SIZE}"
fi
if [ -n "$MAX_MODEL_LEN" ]; then
    DOCKER_CMD="${DOCKER_CMD} -e MAX_MODEL_LEN=${MAX_MODEL_LEN}"
fi
if [ -n "$MOE_BACKEND" ]; then
    DOCKER_CMD="${DOCKER_CMD} -e VLLM_MOE_BACKEND=${MOE_BACKEND}"
fi
if [ -n "$SERVER_ARGS" ]; then
    DOCKER_CMD="${DOCKER_CMD} -e SERVER_ARGS=${SERVER_ARGS}"
fi
if [ -n "$REASONING_PARSER" ]; then
    DOCKER_CMD="${DOCKER_CMD} -e REASONING_PARSER=${REASONING_PARSER}"
fi
if [ -n "$REASONING_EFFORT" ]; then
    DOCKER_CMD="${DOCKER_CMD} -e REASONING_EFFORT=${REASONING_EFFORT}"
fi
if [ "$FORCE_FAILED" = true ]; then
    DOCKER_CMD="${DOCKER_CMD} -e FORCE_FAILED=true"
fi
if [ -n "$CONTINUE_FROM" ]; then
    DOCKER_CMD="${DOCKER_CMD} -e CONTINUE_FROM=${CONTINUE_FROM}"
fi

# WandB
if [ "$WANDB" = true ]; then
    DOCKER_CMD="${DOCKER_CMD} -e WANDB=true"
    if [ -n "$WANDB_API_KEY" ]; then
        DOCKER_CMD="${DOCKER_CMD} -e WANDB_API_KEY=${WANDB_API_KEY}"
    fi
    if [ -n "$WANDB_PROJECT" ]; then
        DOCKER_CMD="${DOCKER_CMD} -e WANDB_PROJECT=${WANDB_PROJECT}"
    fi
    if [ -n "$WANDB_ENTITY" ]; then
        DOCKER_CMD="${DOCKER_CMD} -e WANDB_ENTITY=${WANDB_ENTITY}"
    fi
fi

# HuggingFace token
if [ -n "$HF_TOKEN" ]; then
    DOCKER_CMD="${DOCKER_CMD} -e HF_TOKEN=${HF_TOKEN}"
fi

# SHM size for multi-GPU (NCCL needs shared memory)
if [ "$GPU_COUNT" -gt 1 ] 2>/dev/null; then
    DOCKER_CMD="${DOCKER_CMD} --shm-size=16g"
fi

# Image name
DOCKER_CMD="${DOCKER_CMD} ${DOCKER_IMAGE}"

echo "=== TALES Experiment (local Docker) ==="
echo "Model:      ${MODEL_NAME}"
echo "Agent:      ${AGENT_TYPE}"
echo "Server:     ${SERVER_TYPE}"
echo "Steps:      ${NB_STEPS}"
echo "Seeds:      ${N_SEEDS} (prefix: ${SEED_PREFIX})"
echo "Envs:       ${ENVS}"
echo "Image:      ${DOCKER_IMAGE}"
echo "GPUs:       ${GPU_COUNT}"
[ -n "$DP_SIZE" ] && echo "DP size:    ${DP_SIZE} (TP=$((GPU_COUNT / DP_SIZE)))"
echo "Parallel:   ${PARALLEL}"
echo "Wandb:      ${WANDB}"
[ -n "$REASONING_EFFORT" ] && echo "Reasoning:  effort=${REASONING_EFFORT}"
[ -n "$MAX_MODEL_LEN" ] && echo "MaxLen:     ${MAX_MODEL_LEN}"
echo ""
echo "Command:"
echo "  ${DOCKER_CMD}"
echo ""

exec ${DOCKER_CMD}
