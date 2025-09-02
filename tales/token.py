import os
from typing import Optional

import tiktoken
from llm import Model

# Suppress warnings from transformers
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "True"
from transformers import AutoTokenizer


def get_token_counter(model: Optional[Model] = None):
    if model is None or model.model_id == "gpt-4o":
        return OpenAITokenCounter("gpt-4o")

    if "claude-" in model.model_id:
        return ClaudeTokenCounter(model)

    elif "gemini" in model.model_id or "gemma" in model.model_id:
        return GeminiTokenCounter(model)

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
        elif self.model in ("o4-mini", "o3", "gpt-5", "gpt-5-mini", "gpt-5-nano"):
            self.tokenize = tiktoken.encoding_for_model("o3-mini").encode
        elif self.model in ("gpt-4.1", "gpt-4.1-nano", "gpt-4.1-mini"):
            self.tokenize = tiktoken.encoding_for_model("gpt-4o").encode
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
        from anthropic import NOT_GIVEN

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


class GeminiTokenCounter(TokenCounter):

    def __init__(self, model: Model):
        from google import genai

        self.model = model.model_id
        self.client = genai.Client(api_key=model.get_key())

    def __call__(self, *, messages=None, text=None):
        from google.genai import types

        messages = list(messages or [])
        if text is not None:
            messages += [{"role": "assistant", "content": text.strip()}]

        system = None
        if messages and messages[0]["role"] == "system":
            system = [messages[0]["content"]]
            messages.pop(0)

        chat = self.client.chats.create(
            model=self.model,
            history=[
                types.Content(
                    role=msg["role"].replace("assistant", "model"),
                    parts=[types.Part(text=msg["content"])],
                )
                for msg in messages
            ],
            config=types.GenerateContentConfig(system_instruction=system),
        )

        return self.client.models.count_tokens(
            model=self.model,
            contents=chat.get_history(),
        ).total_tokens
