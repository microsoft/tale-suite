# TALES Benchmark — LLM Model Inventory

> Last updated: 2026-05-12

## Models Already Evaluated

### Open-Source — ReAct Agent
| Model | Runs | Status |
|-------|------|--------|
| Qwen/Qwen3-32B | 615 | ✅ done |
| deepseek-ai/DeepSeek-R1 | 615 | ✅ done |
| deepseek-ai/DeepSeek-R1-Distill-Llama-70B | 615 | ✅ done |
| Qwen/Qwen3.6-35B-A3B | 610 | ✅ done |
| Qwen/Qwen3.5-4B | 610 | ✅ done |
| deepseek-ai/DeepSeek-V4-Flash | 610 | ✅ done |
| Qwen/Qwen3.5-0.8B | 610 | ✅ done |
| Qwen/Qwen3.5-9B | 610 | ✅ done |
| Qwen/Qwen3.5-2B | 610 | ✅ done |
| Qwen/Qwen3-30B-A3B | 610 | ✅ done |
| Qwen/Qwen3.5-27B | 610 | ✅ done |
| Qwen/Qwen3.6-27B | 610 | ✅ done |
| MiniMaxAI/MiniMax-M2.7 | 186 | ✅ done |
| deepseek-ai/DeepSeek-V4-Pro | 7 | 🔄 running |
| mistralai/Magistral-Small-2506 | 610+ | ✅ done |
| openai/gpt-oss-120b | 610+ | ✅ done (effort=1024) |
| openai/gpt-oss-120b (effort=high) | ~360 | 🔄 running (seed 3/5, image 87abc91) |
| Qwen/Qwen3-235B-A22B | 610+ | ✅ done (4 GPUs) |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | 610+ | ✅ done |

### Open-Source — Zero-Shot Only
| Model | Runs |
|-------|------|
| meta-llama/Llama-3.2-1B-Instruct | 973 |
| meta-llama/Llama-3.1-70B-Instruct | 905 |
| meta-llama/Llama-3.2-3B-Instruct | 895 |
| meta-llama/Llama-3.1-8B-Instruct | 894 |
| meta-llama/Llama-3.1-405B-Instruct | 865 |
| mistralai/Mixtral-8x22B-Instruct-v0.1 | 845 |
| microsoft/Phi-3-medium-128k-instruct | 843 |
| mistralai/Mistral-Large-Instruct-2407 | 840 |
| microsoft/Phi-3.5-mini-instruct | 825 |
| mistralai/Ministral-8B-Instruct-2410 | 825 |
| Qwen/Qwen2.5-72B-Instruct | 815 |
| mistralai/Mistral-Small-Instruct-2409 | 795 |
| microsoft/Phi-3-mini-128k-instruct | 765 |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | 645 |
| meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 | 615 |
| meta-llama/Llama-4-Scout-17B-16E-Instruct | 615 |
| mistralai/Mistral-Small-3.1-24B-Instruct-2503 | 615 |
| meta-llama/Llama-3.3-70B-Instruct | 615 |
| microsoft/Phi-3.5-MoE-instruct | 615 |
| mistralai/Mistral-Small-24B-Instruct-2501 | 615 |
| microsoft/phi-4 | 615 |
| microsoft/Phi-4-mini-instruct | 615 |
| Qwen/Qwen2.5-7B-Instruct | 615 |

### Proprietary Models
| Model | Runs | Agent |
|-------|------|-------|
| o3 | 1847 | react |
| claude-3.7-sonnet | 1234 | react, zero-shot |
| claude-opus-4.6 | 983 | react |
| claude-sonnet-4.6 | 981 | react |
| claude-haiku-4.5 | 867 | zero-shot |
| gpt-5.4-nano | 732 | react |
| gpt-5.1 | 730 | react |
| gpt-5.4-mini | 721 | react |
| gpt-4o-mini | 620 | zero-shot |
| o1 | 616 | react |
| gpt-4o | 616 | zero-shot |
| gpt-5 | 615 | react, zero-shot |
| gpt-5-mini | 615 | react, zero-shot |
| gpt-4.1 | 615 | zero-shot |
| gpt-4.1-nano | 615 | zero-shot |
| gpt-4.1-mini | 615 | zero-shot |
| gemini-2.0-flash | 615 | zero-shot |
| claude-3.5-sonnet-latest | 615 | zero-shot |
| claude-3.5-haiku | 615 | zero-shot |
| o3-mini | 615 | react |
| claude-opus-4.5 | 610 | react |
| claude-sonnet-4.5 | 610 | zero-shot |
| claude-4-sonnet | 610 | zero-shot |
| gpt-5-nano | 610 | react |
| o4-mini | 596 | react |
| gemini-2.5-pro-preview-03-25 | 594 | react |
| gpt-5.4 | 446 | react |
| gpt-5.2 | 52 | react |

---

## Candidates — Not Yet Run

### 🔴 Tier 1 — Must-Run

| # | Model | Active/Total | Type | Thinking | B200 GPUs | License | Why |
|---|-------|-------------|------|----------|-----------|---------|-----|
| 1 | deepseek-ai/DeepSeek-R1-0528 | 37B/671B | MoE | ✅ | 4 (FP8) | MIT | Best open reasoning; AIME25 87.5%; supersedes R1 |
| 2 | microsoft/Phi-4-reasoning-plus | 14B/14B | Dense | ✅ | 1 | MIT | ❌ BROKEN — degenerate infinite thinking loops, never produces actions |
| 3 | mistralai/Magistral-Small-2506 | 24B/24B | Dense | ✅ | 1 | Apache 2.0 | ✅ DONE |
| 4 | openai/gpt-oss-120b | 5.1B/117B | MoE (MXFP4) | ✅ | 1 | Apache 2.0 | ✅ DONE (both effort=1024 and effort=high) |

### 🟡 Tier 2 — High Value

| # | Model | Active/Total | Type | Thinking | B200 GPUs | License | Why |
|---|-------|-------------|------|----------|-----------|---------|-----|
| 5 | Qwen/Qwen3-235B-A22B | 22B/235B | MoE | ✅ | 4 (64 heads, TP must be 2/4/8) | Apache 2.0 | ✅ DONE (4 GPUs) |
| 6 | deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | 32B/32B | Dense | ✅ | 1 | MIT | ✅ DONE |
| 7 | nvidia/Llama-3_3-Nemotron-Super-49B-v1 | 49B/49B | Dense (NAS) | ✅ | 1 | NVIDIA OML | Best 49B; RL-tuned for reasoning+agents |
| 8 | google/gemma-4-31b-it | 30.7B/30.7B | Dense (hybrid) | ✅ | 1 | Apache 2.0 | 256K ctx; configurable thinking |
| 9 | Qwen/Qwen3.5-122B-A10B | 10B/122B | MoE (hybrid) | ✅ | 2 | Apache 2.0 | GPQA 86.6%; excellent for ScienceWorld |
| 10 | zai-org/GLM-5.1 | ~?B/~754B | MoE (DSA) | ✅ | 4-5 (FP8) | Check | HLE 31.0%, AIME26 95.3% |
| 11 | Qwen/Qwen3.5-397B-A17B | 17B/397B | MoE (hybrid) | ✅ | 5 (FP8: 3) | Apache 2.0 | Flagship Qwen3.5 |

### 🟢 Tier 3 — Coverage / Ablations

| # | Model | Active/Total | Type | Thinking | B200 GPUs | License | Why |
|---|-------|-------------|------|----------|-----------|---------|-----|
| 12 | deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 8B/8B | Dense | ✅ | 1 | MIT | Best 8B reasoning (AIME24 86.0%) |
| 13 | Qwen/Qwen3-8B | 8B/8B | Dense | ✅ | 1 | Apache 2.0 | Small thinking baseline |
| 14 | Qwen/Qwen3-14B | 14.8B/14.8B | Dense | ✅ | 1 | Apache 2.0 | 14B thinking model |
| 15 | openai/gpt-oss-20b | 3.6B/21B | MoE (MXFP4) | ✅ | 1 | Apache 2.0 | Ultra-compact OpenAI reasoning |
| 16 | nvidia/Llama-3_1-Nemotron-Ultra-253B-v1 | 253B/253B | Dense (NAS) | ✅ | 3 (FP8: 2) | NVIDIA OML | Frontier non-MoE reasoning |
| 17 | nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning | 3B/31B | MoE (Mamba2) | ✅ | 1 | NVIDIA OML | Mamba2 hybrid; fastest inference |
| 18 | poolside/Laguna-XS.2 | 3B/33B | MoE (hybrid) | ✅ | 1 | Apache 2.0 | Interleaved thinking between tool calls |
| 19 | moonshotai/Kimi-K2-Instruct | 32B/1T | MoE (MLA) | ❌ | 6 (FP8) | Modified MIT | 1T frontier; best open agentic tool-use |
| 20 | XiaomiMiMo/MiMo-V2.5-Pro | 42B/1.02T | MoE (hybrid) | ✅ | 6 (FP8) | Apache 2.0 | 1M ctx; pure RL reasoning |
| 21 | mistralai/Magistral-Medium-2506 | ~123B/~123B | Dense | ✅ | 2 | Mistral Research | 123B reasoning |
| 22 | Qwen/Qwen3.5-35B-A3B | 3B/35B | MoE (hybrid) | ✅ | 1 | Apache 2.0 | Distinct from already-run Qwen3.6-35B-A3B |
| 23 | tencent/Hy3-preview | 21B/295B | MoE | ✅ | 4 (FP8: 2) | Tencent Hy | 295B MoE reasoning |
| 24 | inclusionAI/Ling-2.6-flash | 7.4B/104B | MoE (MLA+Linear) | ✅ | 2 | Open | Fast token-efficient |
| 25 | CohereLabs/c4ai-command-a-03-2025 | 111B/111B | Dense (hybrid) | ❌ | 2 | CC-BY-NC | Enterprise RAG/agentic |

---

## Notes
- **GPU estimates** assume B200 192GB. "FP8: N" means N GPUs with FP8 quantization.
- **openai/gpt-oss-\*** may require a custom vLLM fork — verify compatibility before deploying.
- **DeepSeek-R1-0528** now supports system prompts (unlike original R1).
- **Tier 1** has 3 of 4 models fitting on a single B200 — fastest to evaluate.
- Several **zero-shot-only** models (Llama-4-Scout, Maverick, Llama-3.3-70B) could be re-run with react agent if desired.
