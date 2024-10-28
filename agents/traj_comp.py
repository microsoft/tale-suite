import argparse

import llm
import numpy as np
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

import twbench
from twbench.agent import register
from twbench.utils import count_tokens, is_recoverable_error

SYSTEM_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the highest score."
    " Upon reading the text observation, provide a *single* short phrase to interact with the game, e.g. `get lamp` (without the backticks)."
    " When stuck, try using the `help` command to see what commands are available."
)

# SUMMARY_PROMPT = (
#     "You are playing a text-based game and your goal is to finish it with the highest score."
#     "You will be provided with a series of actions and observations, and sometimes a previous summary."
#     "Summarize the most important facts about the provided actions and observations as well as the past summary if it is provided."
#     "Your summary must be as concise as possible while capturing all relevant information."
# )

SUMMARY_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the highest score."
    "You will be provided with a series of actions and observations, and sometimes a previous summary."
    "Convert the actions, observations and rewards into a minimal description of actions, locations and objects interacted with."
    "This description should be as short and concise as possible."
)


def format_messages_to_markdown(messages):
    """Concatenate messages into a single markdown string."""
    markdown_content = ""
    for message in messages:
        role = message["role"].capitalize()
        content = message["content"]
        markdown_content += f"#### {role}\n\n```\n{content}\n```\n\n"
    return markdown_content


class TJ_LLMAgent(twbench.Agent):

    def __init__(self, *args, **kwargs):
        self.llm = kwargs["llm"]
        self.model = llm.get_model(self.llm)
        self.allows_system_prompt = self.llm not in ["o1-mini", "o1-preview"]

        # Provide the API key, if one is needed and has been provided
        self.model.key = llm.get_key(
            kwargs.get("key"), kwargs["llm"], self.model.key_env_var
        ) or llm.get_key(None, self.model.needs_key, self.model.key_env_var)

        self.seed = kwargs.get("seed", 1234)
        self.rng = np.random.RandomState(self.seed)

        self.history = []
        self.context = kwargs.get("context_limit", -0)  # i.e. keep all.
        self.act_temp = kwargs.get("act_temp", 0.0)
        self.conversation = None
        if kwargs.get("conversation"):
            self.conversation = self.model.conversation()

    @property
    def uid(self):
        return (
            f"LLMAgent_{self.llm}"
            f"_s{self.seed}"
            f"_c{self.context}"
            f"_t{self.act_temp}"
            f"_conv{self.conversation is not None}"
            f"Trajectory Compression Agent"
        )

    @property
    def params(self):
        return {
            "llm": self.llm,
            "seed": self.seed,
            "context": self.context,
            "act_temp": self.act_temp,
            "conversation": self.conversation is not None,
        }

    @retry(
        retry=retry_if_exception(is_recoverable_error),
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(100),
    )
    def _llm_call(self, conversation, *args, **kwargs):
        response = conversation.prompt(*args, **kwargs)
        response.duration_ms()  # Forces the response to be computed.
        return response

    def act(self, obs, reward, done, infos):
        conversation = self.conversation or self.model.conversation()
        conversation.responses = conversation.responses[-self.context :]
        # TODO: Add a message saying the history was truncated?

        prompt = f"{obs}\n>" if self.conversation else self.build_prompt(obs)

        if not self.allows_system_prompt:
            if len(conversation.responses) == 0:
                # Model doesn't support system prompt and this is the first message in the conversation.
                prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
            else:
                # Make sure the system prompt is added to the first message in the conversation.
                first_prompt = conversation.responses[0].prompt
                if SYSTEM_PROMPT not in first_prompt.prompt:
                    first_prompt.prompt = f"{SYSTEM_PROMPT}\n\n{first_prompt.prompt}"

        response = self._llm_call(
            conversation,
            prompt=prompt,
            system=SYSTEM_PROMPT if self.allows_system_prompt else None,
            temperature=self.act_temp,
            seed=self.seed,
            top_p=1,
            stream=False,
        )
        action = response.text().strip()
        self.history.append((obs, f"> {action}\n"))

        # Compute usage statistics
        messages = conversation.responses[-1]._prompt_json["messages"]

        stats = {
            "prompt": format_messages_to_markdown(messages),
            "response": response.text(),
            "nb_tokens": count_tokens(messages=messages)
            + count_tokens(text=response.text()),
        }

        return action, stats

    def build_prompt(self, observation):
        messages = []
        # messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # if not self.allows_system_prompt:
        #     messages[-1]["role"] = "user"
        last_act = ""
        for obs, action in self.history[-self.context :]:
            messages.append({"role": "user", "content": obs})
            messages.append({"role": "assistant", "content": action})
            last_act = action

        messages.append({"role": "user", "content": observation})

        # Discard the system prompt.
        # Merge all messages content into a single string
        # prompt = "\n".join([msg["content"] for msg in messages[1:]])
        prompt = "\n".join([msg["content"] for msg in messages])

        if len(self.history) >= self.context:
            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            # print("History:\n", self.history)
            # print("Propmt", prompt)
            response = self._llm_call(
                self.model.conversation(),
                prompt=prompt,
                system=SUMMARY_PROMPT,
                temperature=self.act_temp,
                seed=self.seed,
                top_p=1,
                stream=False,
            )
            summary = {
                "role": "user",
                "content": "History Summary:\n" + response.text().strip(),
            }
            self.history = [("History Summary:\n" + response.text().strip(), last_act)]

            # print("Summary:", response.text().strip())
            # print("Full rollout", len(prompt))
            # print("Summary: ", len(response.text().strip()))
            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

            messages.insert(0, summary)

            prompt = "\n".join([msg["content"] for msg in messages])
        return prompt


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("LLMAgent settings")

    group.add_argument(
        "--llm",
        default="gpt-4o-mini",
        help="LLM to be used for evaluation. Default: %(default)s",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=20241001,
        help="Seed for LLM (not all endpoints support this). Default: %(default)s",
    )
    group.add_argument(
        "--act-temp",
        type=float,
        default=0.0,
        help="Temperature for LLM when taking actions. Default: %(default)s",
    )
    group.add_argument(
        "--context-limit",
        type=int,
        default=10,
        help="Limit context for LLM (in conversation turns). Default: %(default)s",
    )
    group.add_argument(
        "--conversation",
        action="store_true",
        help="Enable conversation mode. Otherwise, use single prompt.",
    )

    return parser


register(
    name="traj_comp",
    desc=(
        "This agent summarizes its trajectory once the interactions exceed the context limit."
    ),
    klass=TJ_LLMAgent,
    add_arguments=build_argparser,
)
