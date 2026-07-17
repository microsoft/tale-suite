#!/bin/bash
set -e

# Sleep before exiting on failure so logs can be inspected
PAUSE_ON_FAILURE="${PAUSE_ON_FAILURE:-600}"
trap 'EXIT_CODE=$?; if [ $EXIT_CODE -ne 0 ]; then echo "=== FAILED (exit $EXIT_CODE). Pod will stay for ${PAUSE_ON_FAILURE}s for log inspection ==="; sleep $PAUSE_ON_FAILURE; fi' EXIT

# --- Persist logs to PVC (survives pod deletion / preemption) ---
if [ -d "/data" ] && [ -n "$MODEL_NAME" ]; then
    LOG_DIR="/data/tales-logs"
    mkdir -p "$LOG_DIR"
    TIMESTAMP=$(date +%Y%m%dT%H%M%S)
    MODEL_SLUG=$(echo "${MODEL_NAME}" | tr '/' '-' | tr '[:upper:]' '[:lower:]')
    LOG_FILE="${LOG_DIR}/${MODEL_SLUG}-${TIMESTAMP}.log"
    echo "Logging to ${LOG_FILE}"
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

# --- Server type: vllm (default) or sglang ---
SERVER_TYPE="${SERVER_TYPE:-vllm}"
# Neutral naming with legacy fallbacks
SERVER_PORT="${SERVER_PORT:-${VLLM_PORT:-8000}}"
SERVER_URL="${SERVER_URL:-${VLLM_URL:-}}"
SERVER_ARGS="${SERVER_ARGS:-${VLLM_ARGS:-}}"

# --- Start inference server (if MODEL_NAME is set and SERVER_URL is not) ---
if [ -n "$MODEL_NAME" ] && [ -z "$SERVER_URL" ]; then
    SERVER_LOG="/tmp/server.log"

    # --- Auto-detect chat template for models that don't ship one ---
    CHAT_TEMPLATE_ARG=""
    if [ -n "$CHAT_TEMPLATE" ]; then
        CHAT_TEMPLATE_ARG="--chat-template ${CHAT_TEMPLATE}"
    elif echo "${MODEL_NAME}" | grep -qi "DeepSeek-V4"; then
        CHAT_TEMPLATE_ARG="--chat-template /app/docker/templates/deepseek_v4.jinja2"
        echo "Auto-detected DeepSeek-V4 model — using custom chat template"
    fi

    if [ "$SERVER_TYPE" = "sglang" ]; then
        # ===================== SGLang =====================
        echo "Starting SGLang server for ${MODEL_NAME} on port ${SERVER_PORT}..."
        SGLANG_EXTRA_ARGS="--trust-remote-code"

        # GPU parallelism (SGLang DP requires router mode — only TP supported here)
        if [ -n "$GPU_COUNT" ] && [ "$GPU_COUNT" -gt 1 ] 2>/dev/null; then
            if [ -n "$DP_SIZE" ] && [ "$DP_SIZE" -gt 1 ] 2>/dev/null; then
                TP_SIZE=$((GPU_COUNT / DP_SIZE))
                SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS} --tp ${TP_SIZE} --dp ${DP_SIZE}"
            else
                SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS} --tp ${GPU_COUNT}"
            fi
        fi

        if [ -n "$MAX_MODEL_LEN" ]; then
            SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS} --context-length ${MAX_MODEL_LEN}"
        fi

        if [ -n "$SERVER_ARGS" ]; then
            SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS} ${SERVER_ARGS}"
        fi

        if [ -n "$CHAT_TEMPLATE_ARG" ]; then
            SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS} ${CHAT_TEMPLATE_ARG}"
        fi

        if [ -n "$LOG_DIR" ]; then
            SERVER_LOG_PERSIST="${LOG_DIR}/${MODEL_SLUG}-${TIMESTAMP}-sglang.log"
            python -m sglang.launch_server \
                --model-path "${MODEL_NAME}" \
                --served-model-name "${MODEL_NAME}" \
                --port "${SERVER_PORT}" \
                --host 0.0.0.0 \
                ${SGLANG_EXTRA_ARGS} \
                > >(tee "$SERVER_LOG" >> "$SERVER_LOG_PERSIST") 2>&1 &
        else
            python -m sglang.launch_server \
                --model-path "${MODEL_NAME}" \
                --served-model-name "${MODEL_NAME}" \
                --port "${SERVER_PORT}" \
                --host 0.0.0.0 \
                ${SGLANG_EXTRA_ARGS} \
                > "$SERVER_LOG" 2>&1 &
        fi

    else
        # ===================== vLLM (default) =====================
        echo "Starting vLLM server for ${MODEL_NAME} on port ${SERVER_PORT}..."
        VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---trust-remote-code}"
        # --- Auto-detect reasoning parser for known model families ---
        if [ -z "$REASONING_PARSER" ] && [ "$AGENT_TYPE" = "reasoning" ]; then
            case "$MODEL_NAME" in
                *DeepSeek-R1*|*deepseek-r1*) REASONING_PARSER="deepseek_r1" ;;
                *DeepSeek-V3*|*DeepSeek-V4*|*deepseek-v3*|*deepseek-v4*) REASONING_PARSER="deepseek_v3" ;;
                *Qwen3*|*qwen3*) REASONING_PARSER="qwen3" ;;
                *gpt-oss*) REASONING_PARSER="openai_gptoss" ;;
                *Magistral*|*Mistral-Small*|*Mistral-Large*) REASONING_PARSER="mistral" ;;
                *MiniMax-M2*|*minimax-m2*) REASONING_PARSER="minimax_m2_append_think" ;;
                *gemma-4*|*gemma4*) REASONING_PARSER="gemma4" ;;
                # Nemotron: uses <think> text tags but Llama-3.3 tokenizer lacks
                # special <think>/<think> token IDs — no vLLM parser works.
                # Thinking is triggered via THINKING_INSTRUCTION in the prompt.
            esac
            if [ -n "$REASONING_PARSER" ]; then
                echo "Auto-detected reasoning parser: ${REASONING_PARSER}"
            fi
        fi
        if [ -n "$REASONING_PARSER" ]; then
            # Enable vLLM's reasoning parser to separate thinking from content.
            # Always add it (even for reasoning agent) — Mistral-native models
            # require the parser for ANY thinking output.
            VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS} --reasoning-config {\"reasoning_parser\":\"${REASONING_PARSER}\"}"
        fi
        if [ "$AGENT_TYPE" = "reasoning" ]; then
            # Reasoning agent may send custom chat templates to control thinking mode
            VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS} --trust-request-chat-template"
        fi
        if [ -n "$GPU_COUNT" ] && [ "$GPU_COUNT" -gt 1 ] 2>/dev/null; then
            TP_SIZE="${GPU_COUNT}"
            if [ -n "$DP_SIZE" ] && [ "$DP_SIZE" -gt 1 ] 2>/dev/null; then
                TP_SIZE=$((GPU_COUNT / DP_SIZE))
                VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS} --data-parallel-size ${DP_SIZE}"
            fi
            VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS} --tensor-parallel-size ${TP_SIZE}"
        fi
        if [ -n "$VLLM_MOE_BACKEND" ]; then
            VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS} --moe-backend ${VLLM_MOE_BACKEND}"
        fi
        if [ -n "$SERVER_ARGS" ]; then
            VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS} ${SERVER_ARGS}"
        fi
        if [ -n "$CHAT_TEMPLATE_ARG" ]; then
            VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS} ${CHAT_TEMPLATE_ARG}"
        fi
        if [ -n "$MAX_MODEL_LEN" ]; then
            VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS} --max-model-len ${MAX_MODEL_LEN}"
        fi
        if [ -n "$LOG_DIR" ]; then
            SERVER_LOG_PERSIST="${LOG_DIR}/${MODEL_SLUG}-${TIMESTAMP}-vllm.log"
            python -m vllm.entrypoints.openai.api_server \
                --model "${MODEL_NAME}" \
                --served-model-name "${MODEL_NAME}" \
                --port "${SERVER_PORT}" \
                --host 0.0.0.0 \
                ${VLLM_EXTRA_ARGS} \
                > >(tee "$SERVER_LOG" >> "$SERVER_LOG_PERSIST") 2>&1 &
        else
            python -m vllm.entrypoints.openai.api_server \
                --model "${MODEL_NAME}" \
                --served-model-name "${MODEL_NAME}" \
                --port "${SERVER_PORT}" \
                --host 0.0.0.0 \
                ${VLLM_EXTRA_ARGS} \
                > "$SERVER_LOG" 2>&1 &
        fi
    fi

    SERVER_PID=$!
    SERVER_URL="http://localhost:${SERVER_PORT}/v1"
fi

# Export vars so the Python benchmark process can detect vLLM mode
export SERVER_URL SERVER_TYPE REASONING_PARSER

# --- Configure LLM model endpoint ---
if [ -n "$SERVER_URL" ] && [ -n "$MODEL_NAME" ]; then
    CONFIG_DIR="${HOME}/.config/io.datasette.llm"
    mkdir -p "$CONFIG_DIR"
    cat > "$CONFIG_DIR/extra-openai-models.yaml" << EOF
- model_id: ${MODEL_NAME}
  model_name: ${MODEL_NAME}
  api_base: "${SERVER_URL}"
EOF
    echo "Configured llm to use ${MODEL_NAME} at ${SERVER_URL}"
fi

# --- Wait for server readiness ---
if [ -n "$SERVER_URL" ]; then
    echo "Waiting for ${SERVER_TYPE} server at ${SERVER_URL}..."
    INACTIVITY_TIMEOUT=${SERVER_WAIT_TIMEOUT:-${VLLM_WAIT_TIMEOUT:-600}}
    elapsed=0
    last_activity=$(date +%s)
    # Stream server logs during startup so user can see progress
    tail -f "$SERVER_LOG" 2>/dev/null &
    TAIL_PID=$!
    until curl -sf "${SERVER_URL}/models" > /dev/null 2>&1; do
        # Check if server process is still alive
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            kill $TAIL_PID 2>/dev/null || true
            echo "ERROR: ${SERVER_TYPE} process died during startup."
            tail -50 "$SERVER_LOG"
            exit 1
        fi
        # Check for log activity (file modification time)
        if [ -f "$SERVER_LOG" ]; then
            log_mtime=$(stat -c %Y "$SERVER_LOG" 2>/dev/null || echo "$last_activity")
            if [ "$log_mtime" -gt "$last_activity" ]; then
                last_activity=$log_mtime
            fi
        fi
        now=$(date +%s)
        inactive_secs=$((now - last_activity))
        if [ $inactive_secs -ge $INACTIVITY_TIMEOUT ]; then
            kill $TAIL_PID 2>/dev/null || true
            echo "ERROR: ${SERVER_TYPE} produced no output for ${INACTIVITY_TIMEOUT}s, aborting."
            tail -50 "$SERVER_LOG"
            exit 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    # Stop streaming logs once server is ready
    kill $TAIL_PID 2>/dev/null || true
    wait $TAIL_PID 2>/dev/null || true
    echo ""
    echo "${SERVER_TYPE} server is ready (took ${elapsed}s)."
fi

# --- Sanity check: verify thinking traces before full benchmark ---
if [ -n "$SERVER_URL" ] && [ "$AGENT_TYPE" = "reasoning" ] && [ "${SKIP_SANITY_CHECK}" != "true" ]; then
    echo ""
    echo "Running thinking trace sanity check..."
    if python scripts/sanity_check.py --model "${MODEL_NAME}" --server-url "${SERVER_URL}" --reasoning-parser "${REASONING_PARSER:-}"; then
        echo "Sanity check passed, proceeding with benchmark."
    else
        echo "WARNING: Sanity check FAILED — model may not produce thinking traces."
        echo "Set SKIP_SANITY_CHECK=true to bypass. Continuing anyway..."
    fi
    echo ""
fi

# --- Run benchmark ---
# If args are passed directly, use them; otherwise build from env vars.
if [ $# -gt 0 ]; then
    echo "Running: python benchmark.py $@"
    exec python benchmark.py "$@"
fi

# Build command from env vars
# Build command from env vars
AGENT_TYPE="${AGENT_TYPE:-zero-shot}"
# Auto-switch to reasoning agent when REASONING_EFFORT is set
if [ -n "$REASONING_EFFORT" ] && [ "$AGENT_TYPE" = "zero-shot" ]; then
    AGENT_TYPE="reasoning"
fi
# Auto-detect agent file from type if not explicitly set
if [ -z "$AGENT_FILE" ]; then
    case "$AGENT_TYPE" in
        reasoning) AGENT_FILE="agents/reasoning.py" ;;
        *) AGENT_FILE="agents/llm.py" ;;
    esac
fi
NB_STEPS="${NB_STEPS:-100}"
CONTEXT_LIMIT="${CONTEXT_LIMIT:-100}"
SEED_PREFIX="${SEED_PREFIX:-20241106}"
N_SEEDS="${N_SEEDS:-5}"

# Base command (without seed or --envs - added per work unit)
BASE_CMD="python benchmark.py --agent ${AGENT_FILE} ${AGENT_TYPE}"
BASE_CMD="${BASE_CMD} --llm ${MODEL_NAME}"
BASE_CMD="${BASE_CMD} --nb-steps ${NB_STEPS}"
BASE_CMD="${BASE_CMD} --context ${CONTEXT_LIMIT}"

# Conversation mode (default: on)
if [ "${CONVERSATION}" != "false" ]; then
    BASE_CMD="${BASE_CMD} --conversation"
fi

# Wandb logging (default: off)
if [ "${WANDB}" = "true" ]; then
    BASE_CMD="${BASE_CMD} --wandb"
fi

# Optional flags
if [ "${FORCE_ALL}" = "true" ]; then
    BASE_CMD="${BASE_CMD} --force-all"
fi

if [ "${FORCE_FAILED}" = "true" ]; then
    BASE_CMD="${BASE_CMD} --force-failed"
fi

if [ -n "$CONTINUE_FROM" ]; then
    BASE_CMD="${BASE_CMD} --continue-from ${CONTINUE_FROM}"
fi

if [ "${ADMISSIBLE_COMMANDS}" = "true" ]; then
    BASE_CMD="${BASE_CMD} --admissible-commands"
fi

if [ -n "$REASONING_EFFORT" ]; then
    BASE_CMD="${BASE_CMD} --reasoning-effort ${REASONING_EFFORT}"
elif [ "$AGENT_TYPE" = "reasoning" ]; then
    # Default thinking budget for reasoning agent when not explicitly set
    BASE_CMD="${BASE_CMD} --cot-max-tokens 1024"
fi

if [ -n "$EXTRA_ARGS" ]; then
    BASE_CMD="${BASE_CMD} ${EXTRA_ARGS}"
fi

# --- Parallelism ---
# PARALLEL=0 or unset: fully sequential (one game at a time).
# PARALLEL=N: run N games concurrently within each seed; seeds are always sequential.
# Strategy: complete all games for seed 1, then seed 2, etc.
PARALLEL="${PARALLEL:-0}"

# Resolve the full environment list.
resolve_envs() {
    if [ -z "$ENVS" ]; then
        python -c "import tales; print(' '.join(tales.envs))"
    else
        # Expand task names (e.g., "jericho") into individual envs
        python -c "
import tales
envs = [e for t in '''$ENVS'''.split() for e in (tales.envs_per_task[t] if t in tales.tasks else [t])]
print(' '.join(envs))
"
    fi
}

# Split a space-separated list into N chunks, printing chunk at index K (0-based).
get_chunk() {
    local items="$1"
    local n_chunks="$2"
    local chunk_idx="$3"
    python -c "
items = '''$items'''.split()
n = int('$n_chunks')
k = int('$chunk_idx')
chunk_size = len(items) // n
remainder = len(items) % n
start = k * chunk_size + min(k, remainder)
end = start + chunk_size + (1 if k < remainder else 0)
print(' '.join(items[start:end]))
"
}

run_work_unit() {
    local seed="$1"
    local envs_subset="$2"
    local label="$3"
    local cmd="${BASE_CMD} --seed ${seed}"
    if [ -n "$envs_subset" ]; then
        cmd="${cmd} --envs ${envs_subset}"
    fi
    echo "[${label}] Running: ${cmd}"
    ${cmd}
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[${label}] FAILED (exit code $rc)"
    else
        echo "[${label}] Done."
    fi
    return $rc
}

# Determine parallelism (within each seed)
if [ "$PARALLEL" -le 1 ] 2>/dev/null; then
    EFFECTIVE_PARALLEL=1
else
    EFFECTIVE_PARALLEL=$PARALLEL
fi

# Resolve envs to split across parallel workers
RESOLVED_ENVS=$(resolve_envs)
N_ENVS=$(echo "$RESOLVED_ENVS" | wc -w)

# Number of env chunks per seed = min(EFFECTIVE_PARALLEL, N_ENVS)
if [ "$EFFECTIVE_PARALLEL" -gt "$N_ENVS" ]; then
    N_CHUNKS=$N_ENVS
else
    N_CHUNKS=$EFFECTIVE_PARALLEL
fi

echo "Parallelism: PARALLEL=${EFFECTIVE_PARALLEL} workers per seed, seeds=${N_SEEDS} (sequential), envs=${N_ENVS}, chunks=${N_CHUNKS}"

# --- Execute work units: one seed at a time, parallel games within ---
FAILURES=0
TOTAL_UNITS=0

for i in $(seq 1 ${N_SEEDS}); do
    SEED="${SEED_PREFIX}${i}"
    echo "=== Starting seed ${i}/${N_SEEDS} (${SEED}) ==="
    RUNNING=0

    if [ "$N_CHUNKS" -le 1 ]; then
        # Single worker: run all envs sequentially
        TOTAL_UNITS=$((TOTAL_UNITS + 1))
        LABEL="seed${i}"
        ENVS_ARG=""
        [ -n "$ENVS" ] && ENVS_ARG="$ENVS"
        run_work_unit "$SEED" "$ENVS_ARG" "$LABEL" || FAILURES=$((FAILURES + 1))
    else
        # Multiple workers: split envs into chunks, run in parallel
        for chunk_idx in $(seq 0 $((N_CHUNKS - 1))); do
            CHUNK=$(get_chunk "$RESOLVED_ENVS" "$N_CHUNKS" "$chunk_idx")
            [ -z "$CHUNK" ] && continue
            TOTAL_UNITS=$((TOTAL_UNITS + 1))
            LABEL="seed${i}/chunk$((chunk_idx+1))"

            run_work_unit "$SEED" "$CHUNK" "$LABEL" &
            RUNNING=$((RUNNING + 1))
            if [ "$RUNNING" -ge "$EFFECTIVE_PARALLEL" ]; then
                wait -n || FAILURES=$((FAILURES + 1))
                RUNNING=$((RUNNING - 1))
            fi
        done

        # Wait for all chunks of this seed to finish before next seed
        while [ "$RUNNING" -gt 0 ]; do
            wait -n || FAILURES=$((FAILURES + 1))
            RUNNING=$((RUNNING - 1))
        done
    fi

    echo "=== Seed ${i}/${N_SEEDS} complete ==="
done

if [ "$FAILURES" -gt 0 ]; then
    echo "WARNING: ${FAILURES}/${TOTAL_UNITS} work units failed."
    exit 1
fi

echo "All ${TOTAL_UNITS} work units completed successfully."
