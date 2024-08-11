import os
import llm
import time
import openai
from typing import Optional
from pydantic import field_validator, Field
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
        temperature: Optional[float] = Field(
            description="Temperature for sampling",
            default=None
        )

        @field_validator("temperature")
        def validate_delay(cls, temperature):
            if temperature is None:
                return None
            if not 0 <= temperature <= 1:
                raise ValueError("temperature must be between 0 and 1")
            return temperature
        
    def execute(self, prompt, stream, response, conversation):
        start_time = time.time()
        completion = self.client.chat.completions.create(
            model=self.deployment_id,
            messages= [
            {
                "role": "user",
                "content": prompt.prompt
            }],
            max_tokens=100,
            temperature=prompt.options.temperature or 0.0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop="\n"
        )
        result = completion.choices[0].message.content
        
        end_time = time.time()
        token_usage = completion.usage.to_dict()

        root_span = Trace(
            name="agent_llm",
            kind="llm",
            metadata={
                "temperature": prompt.options.temperature or 0.0,
                "token_usage": token_usage,

            },
            start_time_ms=start_time,
            end_time_ms=end_time,
            inputs={"query": prompt.prompt},
            outputs={"response": result, "token_usage": token_usage}
        )

        root_span.log(name="openai_trace")
        
        if (result.startswith(">")):
            result = result[1:].strip()

        return result.strip()
    
    def text(self, prompt, **kwargs):
        result = self.execute(prompt, **kwargs)
        return result["choices"][0]["text"]