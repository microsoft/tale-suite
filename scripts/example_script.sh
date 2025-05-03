# This is an example script to show how to use a self-hosted model with vllm to run the twb
# Make sure the model you use is already registered with the llm package

MODEL=""
# MODEL=meta-llama/Llama-3.1-8B
export HF_TOKEN=''
export WANDB_API_KEY=''
export WANDB_MODE="online"
wandb login

cat <<EOL >> /root/.config/io.datasette.llm/extra-openai-models.yaml

- model_id: $MODEL
  model_name: $MODEL
  api_base: "http://0.0.0.0:8002/v1"
EOL

# Makes a log folder for vllm
mkdir -p logs 

# Clean up previous server if running
pkill -f "vllm.entrypoints.openai.api_server" || true

# Start VLLM server properly
echo "Starting VLLM server..."
nohup python -m vllm.entrypoints.openai.api_server --model ${MODEL} --port 8002 \
  --tensor-parallel-size 2 --trust-remote-code --host 0.0.0.0 \
  --chat-template "{prompt}" > logs/vllm_1.log 2>&1 &
VLLM_PID=$!
# Cleanup on script exit
trap "kill $VLLM_PID 2>/dev/null || true" EXIT

echo "Waiting for VLLM server to start..."
timeout=500 
interval=30  
elapsed=0
sleep 5
echo "Pinging VLLM server..."

# Wait loop with timeout
until curl -s -o /dev/null -w "%{http_code}" http://localhost:8002/v1/models | grep -q "200"; do
    if [ $elapsed -ge $timeout ]; then
        echo "Timeout reached! VLLM server did not start within $timeout seconds."
        exit 1
    fi
    sleep $interval
    echo "Pinging vllm server..."
    elapsed=$((elapsed + interval))
done

echo "VLLM server started successfully!"

# Test if the model is available
JSON_DATA="{
  \"model\": \"${MODEL}\",
  \"prompt\": \"You want to play a (text) game?\",
  \"max_tokens\": 10
}"

echo "Testing model with a sample prompt..."
curl -X POST "http://localhost:8002/v1/completions" \
  -H "Content-Type: application/json" \
  -d "$JSON_DATA"
echo -e "\n"

# Run the text games benchmark
set -e  # Exit immediately if a command exits with non-zero status

# Choose one method:
# Method 1: Run processes in parallel
pids=""
for i in {1..5}; do
    echo "Starting benchmark run $i..."
    python benchmark.py --agent agents/llm.py zero-shot --conversation --llm ${MODEL} \
      --context 200 --nb-steps 100 --envs JerichoEnvZork1 -ff --seed "20241106$i" &
    pids="$pids $!"
    sleep 60
done

echo "All done!"