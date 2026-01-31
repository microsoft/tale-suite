import argparse

import numpy as np
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
from vllm import LLM, SamplingParams

import tales
from tales.agent import register
from tales.utils import (
    format_messages_to_markdown,
    is_recoverable_error,
    merge_messages,
)

SYSTEM_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the highest score."
    " Upon reading the text observation, provide a *single* short phrase to interact with the game, e.g. `get lamp` (without the backticks)."
    " When stuck, try using the `help` command to see what commands are available."
)


class VLLMAgent(tales.Agent):

    def __init__(self, *args, **kwargs):
        self.model_name = kwargs["model"]
        self.seed = kwargs["seed"]
        self.rng = np.random.RandomState(self.seed)

        # vLLM-specific parameters
        tensor_parallel_size = kwargs.get("tensor_parallel_size", 1)
        quantization = kwargs.get("quantization", None)
        dtype = kwargs.get("dtype", "auto")
        gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.9)
        max_model_len = kwargs.get("max_model_len", None)
        trust_remote_code = kwargs.get("trust_remote_code", False)

        # Initialize vLLM model
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=tensor_parallel_size,
            quantization=quantization,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            seed=self.seed,
            trust_remote_code=trust_remote_code,
        )

        # Get tokenizer for token counting
        self.tokenizer = self.llm.get_tokenizer()

        self.history = []
        self.context_limit = kwargs["context_limit"]
        if self.context_limit is not None:
            assert self.context_limit > 0, "--context-limit must be greater than 0."

        self.act_temp = kwargs["act_temp"]
        self.conversation = kwargs["conversation"]

        # Advanced sampling parameters
        self.top_p = kwargs.get("top_p", 1.0)
        self.top_k = kwargs.get("top_k", -1)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        self.presence_penalty = kwargs.get("presence_penalty", 0.0)
        self.frequency_penalty = kwargs.get("frequency_penalty", 0.0)

    @property
    def uid(self):
        return (
            f"VLLMAgent_{self.model_name.replace('/', '_')}"
            f"_s{self.seed}"
            f"_c{self.context_limit}"
            f"_t{self.act_temp}"
            f"_conv{self.conversation}"
        )

    @property
    def params(self):
        return {
            "agent_type": "vllm-zero-shot",
            "model": self.model_name,
            "seed": self.seed,
            "context_limit": self.context_limit,
            "act_temp": self.act_temp,
            "conversation": self.conversation,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }

    def _count_tokens(self, text=None, messages=None):
        """Count tokens in text or messages."""
        if text is not None:
            return len(self.tokenizer.encode(text))
        elif messages is not None:
            # Format messages using chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template'):
                formatted = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                return len(self.tokenizer.encode(formatted))
            else:
                # Fallback: concatenate message contents
                text = "\n".join([msg["content"] for msg in messages])
                return len(self.tokenizer.encode(text))
        return 0

    @retry(
        retry=retry_if_exception(is_recoverable_error),
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(100),
    )
    def _llm_call_from_messages(self, messages, *args, **kwargs):
        """Call vLLM with messages using chat template."""
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 100),
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            seed=self.seed,
        )

        # Format messages using chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            # Use the model's chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback: simple formatting
            prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            prompt += "Assistant: "

        # Generate response
        outputs = self.llm.generate([prompt], sampling_params)

        return outputs[0]

    def act(self, obs, reward, done, infos):
        messages = self.build_messages(f"{obs}\n> ")
        llm_kwargs = {
            "temperature": self.act_temp,
            "max_tokens": 100,  # Text actions are short phrases.
        }

        response = self._llm_call_from_messages(messages, **llm_kwargs)

        action = response.outputs[0].text.strip()
        self.history.append((f"{obs}\n> ", f"{action}\n"))

        # Compute usage statistics
        stats = {
            "prompt": format_messages_to_markdown(messages),
            "response": response.outputs[0].text,
            "nb_tokens_prompt": self._count_tokens(messages=messages),
            "nb_tokens_response": self._count_tokens(text=response.outputs[0].text),
        }

        stats["nb_tokens"] = stats["nb_tokens_prompt"] + stats["nb_tokens_response"]

        return action, stats

    def build_messages(self, observation):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        limit = self.context_limit or len(self.history) + 1

        for i, (obs, action) in enumerate(self.history[-limit:]):
            if len(self.history) >= limit and i == 0:
                # Add the current observation.
                obs = (
                    f"// History has been truncated to the last {limit} steps.\n...\n> "
                )

            messages.append({"role": "user", "content": obs})
            messages.append({"role": "assistant", "content": action})

        messages.append({"role": "user", "content": observation})

        # Just in case, let's avoid having multiple messages from the same role.
        messages = merge_messages(messages)

        if not self.conversation:
            # Merge all messages into a single message except for the system.
            content = "".join([msg["content"] for msg in messages[1:]])
            messages = messages[:1] + [{"role": "user", "content": content}]

        return messages


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("VLLMAgent settings")

    group.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model path or HuggingFace model ID. Default: %(default)s",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=20241001,
        help="Seed for LLM generation. Default: %(default)s",
    )
    group.add_argument(
        "--act-temp",
        type=float,
        default=0.0,
        help="Temperature for LLM when taking actions. Default: %(default)s",
    )
    group.add_argument(
        "--context-limit",
        type=int,
        help="Limit context for LLM (in conversation turns). Default: no limit.",
    )
    group.add_argument(
        "--conversation",
        required=True,
        action=argparse.BooleanOptionalAction,
        help="Enable conversation mode. Otherwise, use single prompt.",
    )

    # vLLM-specific parameters
    vllm_group = parser.add_argument_group("vLLM-specific settings")

    vllm_group.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism. Default: %(default)s",
    )
    vllm_group.add_argument(
        "--quantization",
        type=str,
        choices=["awq", "gptq", "squeezellm", "fp8"],
        help="Quantization method to use. Default: None",
    )
    vllm_group.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights. Default: %(default)s",
    )
    vllm_group.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use. Default: %(default)s",
    )
    vllm_group.add_argument(
        "--max-model-len",
        type=int,
        help="Maximum context length for the model. Default: model's default",
    )
    vllm_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading model. Default: False",
    )

    # Advanced sampling parameters
    sampling_group = parser.add_argument_group("Advanced sampling settings")

    sampling_group.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling parameter. Default: %(default)s",
    )
    sampling_group.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling parameter. -1 means disabled. Default: %(default)s",
    )
    sampling_group.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty. 1.0 means no penalty. Default: %(default)s",
    )
    sampling_group.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="Presence penalty for penalizing tokens that have appeared. Default: %(default)s",
    )
    sampling_group.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.0,
        help="Frequency penalty based on token frequency. Default: %(default)s",
    )

    return parser


register(
    name="vllm-zero-shot",
    desc=(
        "This agent uses vLLM to run a local LLM to decide which action to take in a zero-shot manner."
    ),
    klass=VLLMAgent,
    add_arguments=build_argparser,
)
