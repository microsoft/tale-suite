import os
import json
import llm
import time
import wandb
import requests
import urllib.request
from typing import Optional
from transformers import LlamaTokenizerFast
from pydantic import validator, Field
from wandb.sdk.data_types.trace_tree import Trace

@llm.hookimpl
def register_models(register):
    register(GCRPhi())

class GCRPhi(llm.Model):
    model_id = "gcr_phi"
    api_key = os.getenv("GCR_PHI_API_KEY")
    base_url = os.getenv("GCR_PHI_ENDPOINT")
    deployment_id = "phi-3-medium-128k-instruct-1"

    tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")

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
        
    def count_tokens(self, text):
        tokens = self.tokenizer.encode(text)
        return len(tokens)
    
    def execute(self, prompt, stream, response, conversation):
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
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        data = {
        "input_data": {
            "input_string": messages,
            "parameters": {
            "temperature": prompt.options.temperature or 0.0,
            "top_p": 0.9,
            "seed": prompt.options.seed or None,
            "do_sample": True,
            "max_new_tokens": 1000
            }
        }
        }
       
        body = str.encode(json.dumps(data))

        if not self.api_key:
            raise Exception("A key should be provided to invoke the endpoint")

        # The azureml-model-deployment header will force the request to go to a specific deployment.
        # Remove this header to have the request observe the endpoint traffic rules
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ self.api_key), 'azureml-model-deployment': self.deployment_id}

        start_time = time.time()

        req = urllib.request.Request(self.base_url, body, headers)

        try:
            result = urllib.request.urlopen(req).read()
        except urllib.error.HTTPError as error:
            print("The request failed with status code")

        result = json.loads(result.decode("utf-8"))["output"]
        end_time = time.time()

        token_usage = { "prompt_tokens": self.count_tokens(prompt.prompt), "completion_tokens": self.count_tokens(result) }
        token_usage["total_tokens"] = token_usage["prompt_tokens"] + token_usage["completion_tokens"]

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
                inputs={"query": prompt.prompt},
                outputs={"response": result, "token_usage": token_usage}
            )
            root_span.log(name="phi_trace")

        if result.startswith("<|assistant|>"):
            result = result[len("<|assistant|>"):].strip()

        response.messages = messages
        response.token_usage = token_usage
        return result.strip()
    
    def text(self, prompt, **kwargs):
        result = self.execute(prompt, **kwargs)
        return json.loads(result.decode("utf-8"))["output"]
