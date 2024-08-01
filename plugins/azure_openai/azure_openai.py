import os
import llm
import requests

@llm.hookimpl
def register_models(register):
    register(AzureOpenAI())

class AzureOpenAI(llm.Model):
    model_id = "azure_openai"
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_id = "gpt-4o"

    def execute(self, prompt, stream, response, conversation):
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        prompt_str = str(prompt)
        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_str}
            ],
            "max_tokens": 100,
            # "temperature": 0,
            # "top_p": 1,
            # "frequency_penalty": 0,
            # "presence_penalty": 0,
            # "stop": "\n"
        }
        url = f"{self.base_url}openai/deployments/{self.deployment_id}/chat/completions?api-version=2024-02-01"
        response = requests.post(url, headers=headers, json=data)
        print(response)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def text(self, prompt, **kwargs):
        result = self.execute(prompt, **kwargs)
        return result["choices"][0]["text"]