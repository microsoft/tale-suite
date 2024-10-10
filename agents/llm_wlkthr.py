import logging

import llm
import numpy as np

import twbench
from twbench.utils import count_tokens

log = logging.getLogger("tw-bench")

# For the LLMWlkThrAgent, the sysprompt is initialized in the __init__ function as we need to change it once we extract the walkthrough from the env
class LLMWlkThrAgent(twbench.Agent):

    def __init__(self, *args, **kwargs):
        self.llm = kwargs["llm"]
        self.model = llm.get_model(self.llm)

        # Provide the API key, if one is needed and has been provided
        self.model.key = llm.get_key(
            kwargs.get("key"), kwargs["llm"], self.model.key_env_var
        ) or llm.get_key(None, self.model.needs_key, self.model.key_env_var)

        self.seed = kwargs.get("seed", 1234)
        self.rng = np.random.RandomState(self.seed)

        self.history = []
        self.context = kwargs.get(
            "context_limit", -0
        )  # Default: keep all conversation turns.
        self.act_temp = kwargs.get("act_temp", 0.0)
        self.conversation = (
            self.model.conversation() if kwargs.get("conversation") else None
        )

        self.sys_prompt = (
                "You are playing a text-based game and your goal is to finish it with the highest score."
                " The following is a walkthrough in the form of a list of actions to beat the game."
                " You should follow this walkthrough as closely as possible to get the maximum score"
                " You must ONLY respond with the action you wish to take with no other special tokens."
                "Walkthrough: WALKTHROUGH"
            )

    @property
    def uid(self):
        return (
            f"LLMAgent_{self.llm}"
            f"_s{self.seed}"
            f"_c{self.context}"
            f"_t{self.act_temp}"
            f"_conv{self.conversation is not None}"
            f"Walkthrough Agent"
        )
    
    def load_wlkthr(self, wlkthr):
        self.sys_prompt = self.sys_prompt.replace("WALKTHROUGH", ",".join(wlkthr))
        return True

    def act(self, obs, reward, done, infos):
        conversation = self.conversation or self.model.conversation()
        conversation.responses = conversation.responses[-self.context :]
        prompt = f"{obs}\n>" if self.conversation else self.build_prompt(obs)

        response = conversation.prompt(
            prompt=prompt,
            system=self.sys_prompt,
            temperature=self.act_temp,
            seed=self.seed,
            top_p=1,
        )

        action = response.text().strip()
        self.history.append((obs, f"> {action}\n"))

        # Compute usage statistics
        messages = conversation.responses[-1]._prompt_json["messages"]
        stats = {
            "prompt": response._prompt_json,
            "response": response.text(),
            "nb_tokens": count_tokens(messages=messages)
            + count_tokens(text=response.text()),
        }

        return action, stats

    def build_prompt(self, observation):
        messages = [{"role": "system", "content": self.sys_prompt}]

        for obs, action in self.history[-self.context :]:
            messages.append({"role": "user", "content": obs})
            messages.append({"role": "assistant", "content": action})

        messages.append({"role": "user", "content": observation})

        # Discard the system prompt.
        # Merge all messages content into a single string
        prompt = "\n".join([msg["content"] for msg in messages[1:]])
        return prompt
