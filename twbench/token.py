import os
from typing import Optional

import tiktoken
from anthropic import NOT_GIVEN
from llm import Model

# Suppress warnings from transformers
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "True"
from transformers import AutoTokenizer


def get_token_counter(model: Optional[Model] = None):
    if model is None or model.model_id == "gpt-4o":
        return OpenAITokenCounter("gpt-4o")

    if "claude-" in model.model_id:
        return ClaudeTokenCounter(model)

    try:
        return OpenAITokenCounter(model.model_id)
    except KeyError:
        pass

    # Try to load from transformers.
    return HuggingFaceTokenCounter(model.model_id)


class TokenCounter:

    def __call__(self, *, messages=None, text=None):
        nb_tokens = 0
        if messages is not None:
            nb_tokens += sum(len(self.tokenize(msg["content"])) for msg in messages)

        if text is not None:
            nb_tokens += len(self.tokenize(text))

        return nb_tokens


class OpenAITokenCounter(TokenCounter):
    def __init__(self, model: str):
        self.model = model
        if self.model in tiktoken.model.MODEL_TO_ENCODING:
            self.tokenize = tiktoken.encoding_for_model(self.model).encode
        else:
            self.tokenize = tiktoken.encoding_for_model(self.model.split("_")[0]).encode


class HuggingFaceTokenCounter(TokenCounter):
    def __init__(self, model: str):
        self.model = model
        try:
            self.tokenize = AutoTokenizer.from_pretrained(self.model).tokenize
        except OSError:
            msg = (
                f"Tokenizer not found for model {self.model},"
                " make sure you have access to the model"
                " (e.g., HuggingFace API key is correctly set)."
            )
            raise ValueError(msg)

    def __call__(self, *, messages=None, text=None):
        nb_tokens = 0
        if messages is not None:
            nb_tokens += sum(len(self.tokenize(msg["content"])) for msg in messages)

        if text is not None:
            nb_tokens += len(self.tokenize(text))

        return nb_tokens


class ClaudeTokenCounter(TokenCounter):

    def __init__(self, model: Model):
        from anthropic import Anthropic

        self.model = model.claude_model_id
        self.client = Anthropic(api_key=model.get_key())

    def __call__(self, *, messages=None, text=None):
        messages = list(messages or [])
        if text is not None:
            messages += [{"role": "assistant", "content": text.strip()}]

        # Extract system messages, if any.
        system = NOT_GIVEN
        if messages and messages[0]["role"] == "system":
            system = messages[0]["content"]
            messages.pop(0)

        return self.client.beta.messages.count_tokens(
            model=self.model,
            messages=messages,
            system=system,
        ).input_tokens
