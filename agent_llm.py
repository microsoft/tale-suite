import os
from openai import AzureOpenAI
import numpy as np
import textworld


client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-01",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)

def llm(prompt, stop=["\n"]):
    # print("prompt:" + prompt)
    completion = client.chat.completions.create(
    model="gpt-4-0125-preview",
    messages= [
    {
      "role": "user",
      "content": prompt
    }],
    max_tokens=100,
    temperature=0,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=stop
    )
    return completion.choices[0].message.content


class LLMAgent(textworld.Agent):
    def __init__(self, seed=1234):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.actions = ["north", "south", "east", "west", "up", "down",
                        "look", "inventory", "take all", "YES", "wait",
                        "take", "drop", "eat", "attack"]

    def reset(self, env):
        env.display_command_during_render = True

    def act(self, game_state, reward, done):
        action = self.rng.choice(self.actions)
        if action in ["take", "drop", "eat", "attack"]:
            if not game_state.feedback.startswith("["):
                action += " " + llm("Reply with only the next word in this step of the game starting after '" + action +"': " + game_state.feedback, stop=["\n"])[:100]

        print(f"{game_state.feedback}\n>{action}")
        return action
