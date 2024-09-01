import os
import llm
import time
import wandb
import openai
from typing import Optional
from pydantic import validator, Field

@llm.hookimpl
def register_models(register):
    register(AzureOpenAI())

class AzureOpenAI(llm.Model):
    model_id = "azure_openai"
    deployment_id = "gpt-4o"

    client = openai.AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2024-02-01",
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    class Options(llm.Options):
        seed: Optional[int] = Field(
            description="Seed for sampling",
            default=None
        )
        temperature: Optional[float] = Field(
            description="Temperature for sampling",
            default=None
        )
        context: Optional[int] = Field(
            description="Number of previous messages to include in the context",
            default=None
        )
        stop: Optional[str] = Field(
            description="Stop token for sampling",
            default=None
        )
        top_p: Optional[float] = Field(
            description="Top p for sampling",
            default=None
        )
        max_tokens: Optional[int] = Field(
            description="Maximum number of tokens to generate",
            default=None
        )

        @validator("seed")
        def validate_seed(cls, seed):
            if seed is None:
                return None
            if not isinstance(seed, int):
                raise ValueError("seed must be an integer")
            return seed

        @validator("temperature")
        def validate_temperature(cls, temperature):
            if temperature is None:
                return None
            if not 0 <= temperature <= 1:
                raise ValueError("temperature must be between 0 and 1")
            return temperature
        
        @validator("context")
        def validate_context(cls, context):
            if context is None:
                return None
            if not 1 <= context <= 1000:
                raise ValueError("context must be between 1 and 100")
            return context
        
        @validator("stop")
        def validate_stop(cls, stop):
            if stop is None:
                return None
            if not isinstance(stop, str):
                raise ValueError("stop must be a string")
            return stop

        @validator("top_p")
        def validate_top_p(cls, top_p):
            if top_p is None:
                return None
            if not 0 <= top_p <= 1:
                raise ValueError("temperature must be between 0 and 1")
            return top_p
        
        @validator("max_tokens")
        def validate_max_tokens(cls, max_tokens):
            if max_tokens is None:
                return None
            if not 1 <= max_tokens <= 1000:
                raise ValueError("max_tokens must be between 1 and 1000")
            return max_tokens
          
    def execute(self, prompt, stream, response, conversation):
        start_time = time.time()
        messages = []
        if (conversation):
            messages= [
                {
                    "role": "system",
                    "content": prompt.system
                }
            ]
            context = prompt.options.context or 10 
            for resp in conversation.responses[-context:]:
                messages.append({ "role": "user", "content": resp.prompt.prompt})
                messages.append({ "role": "assistant", "content": resp.text()})
            messages.append({ "role": "user", "content": prompt.prompt})
        else:
            messages = [
                {
                    "role": "user",
                    "content": prompt.prompt
                }
            ]
        completion = self.client.chat.completions.create(
            model=self.deployment_id,
            messages=messages,
            max_tokens=100,
            temperature=prompt.options.temperature or 0.0,
            top_p=1,
            seed=prompt.options.seed or None,
            frequency_penalty=0,
            presence_penalty=0,
            stop="\n"
        )
        result = completion.choices[0].message.content
        
        end_time = time.time()
        token_usage = completion.usage.to_dict()

        if (result.startswith(">")):
            result = result[1:].strip()

        response.messages = messages
        response.token_usage = token_usage
        return result.strip()
    
    def text(self, prompt, **kwargs):
        result = self.execute(prompt, **kwargs)
        return result["choices"][0]["text"]