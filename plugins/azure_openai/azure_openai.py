import os
import llm
import openai

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

    def execute(self, prompt, stream, response, conversation):
        completion = self.client.chat.completions.create(
            model=self.deployment_id,
            messages= [
            {
                "role": "user",
                "content": str(prompt)
            }],
            max_tokens=100,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop="\n"
        )
        return completion.choices[0].message.content
    
    def text(self, prompt, **kwargs):
        result = self.execute(prompt, **kwargs)
        return result["choices"][0]["text"]