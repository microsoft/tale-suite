import os
import llm
import openai
from typing import Optional
from pydantic import field_validator, Field

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
        return completion.choices[0].message.content
    
    def text(self, prompt, **kwargs):
        result = self.execute(prompt, **kwargs)
        return result["choices"][0]["text"]