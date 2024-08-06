import os
import json
import llm
import requests
import urllib.request

@llm.hookimpl
def register_models(register):
    register(GCRPhi())

class GCRPhi(llm.Model):
    model_id = "gcr_phi"
    api_key = os.getenv("GCR_PHI_API_KEY")
    base_url = os.getenv("GCR_PHI_ENDPOINT")
    deployment_id = "phi-3-medium-128k-instruct-1"

    def execute(self, prompt, stream, response, conversation):
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        prompt_str = str(prompt)
        data = {
        "input_data": {
            "input_string": [
            {
                "role": "user",
                "content": prompt_str
            }
            ],
            "parameters": {
            "temperature": 0.7,
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

        req = urllib.request.Request(self.base_url, body, headers)

        try:
            result = urllib.request.urlopen(req).read()
        except urllib.error.HTTPError as error:
            print("The request failed with status code") 
        return json.loads(result.decode("utf-8"))["output"]
    
    def text(self, prompt, **kwargs):
        result = self.execute(prompt, **kwargs)
        return json.loads(result.decode("utf-8"))["output"]
