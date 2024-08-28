import os
import llm
import time
import wandb
import openai
from typing import Optional
from pydantic import validator, Field
from wandb.sdk.data_types.trace_tree import Trace

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
        seed: Optional[float] = Field(
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

        if wandb.run:
            root_span = Trace(
                name="agent_llm",
                kind="llm",
                metadata={
                    "temperature": prompt.options.temperature or 0.0,
                    "token_usage": token_usage,

                },
                start_time_ms=start_time,
                end_time_ms=end_time,
                inputs={"length": len(messages), "messages": messages},
                outputs={"response": result, "token_usage": token_usage}
            )

            root_span.log(name="openai_trace")
            
        if (result.startswith(">")):
            result = result[1:].strip()

        response.messages = messages
        response.token_usage = token_usage
        return result.strip()
    
    def text(self, prompt, **kwargs):
        result = self.execute(prompt, **kwargs)
        return result["choices"][0]["text"]