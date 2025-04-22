# This is an example script to show how to use a self-hosted model with vllm to run the twb
model=""

cat <<EOL >> .config/io.datasette.llm/extra-openai-models.yaml

  - model_id: $model
    model_name: $model
    api_base: "http://127.0.0.1:8002/v1"
EOL

export WANDB_API_KEY=''
# Makes a log folder for vllm. This may error out if you already have a logs folder
mkdir logs 

# Run the vllm server for the meta-llama/Llama-3.1-8B-Instruct model on port 8002. Make sure you have set your HF token 
nohup bash -c 'until ! (python -m vllm.entrypoints.openai.api_server --model mistralai/Ministral-8B-Instruct-2410 --port 8002 --tensor-parallel-size 1 --trust-remote-code --host 0.0.0.0 > logs/vllm_1.log 2>&1); do sleep 120; done' &

# To make sure this doesn't run forever, we let it run for 300 seconds and check every 30 seconds
echo "Waiting for VLLM server to start..."
timeout=500 
interval=30  
elapsed=0

# Wait loop with timeout
until curl -s -o /dev/null -w "%{http_code}" http://localhost:8002/v1/models | grep -q "200"; do
    if [ $elapsed -ge $timeout ]; then
        echo "Timeout reached! VLLM server did not start within 5 minutes."
        exit 1
    fi
    sleep $interval
    echo "Pinging vllm server..."
    elapsed=$((elapsed + interval))
done

# Send a test request to the API
curl -X POST "http://localhost:8002/v1/completions" -H "Content-Type: application/json" -d '{"model": "mistralai/Ministral-8B-Instruct-2410", "prompt": "You want to play a (text) game?", "max_tokens": 10}'

# Run the text games benchmark with the model we just set up for xork1
wandb login

set -ex

pids=""

for i in {1..5}; do
    python benchmark.py --agent agents/llm.py zero-shot --conversation --llm mistralai/Ministral-8B-Instruct-2410 --envs JerichoEnvZork1 --context 100 --nb-steps 100 --conversation --seed "20241106$((i))"
    pids="$pids $!"
    sleep 60
done

wait $pids