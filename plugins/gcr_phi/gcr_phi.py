import os
import json
import llm
import time
import requests
import urllib.request
from typing import Optional
from pydantic import field_validator, Field
from wandb.sdk.data_types.trace_tree import Trace

@llm.hookimpl
def register_models(register):
    register(GCRPhi())

class GCRPhi(llm.Model):
    model_id = "gcr_phi"
    api_key = os.getenv("GCR_PHI_API_KEY")
    base_url = os.getenv("GCR_PHI_ENDPOINT")
    deployment_id = "phi-3-medium-128k-instruct-1"

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
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        data = {
        "input_data": {
            "input_string": [
            {
                "role": "user",
                "content": prompt.prompt
            }
            ],
            "parameters": {
            "temperature": prompt.options.temperature or 0.0,
            "top_p": 0.9,
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

        token_usage = 0 #TODO: get token usage from response

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

        return result.strip()
    
    def text(self, prompt, **kwargs):
        result = self.execute(prompt, **kwargs)
        return json.loads(result.decode("utf-8"))["output"]
