import argparse

import twbench
from twbench.agent import register
from twbench.utils import TokenCounter


class WalkthroughAgent(twbench.Agent):
    def __init__(self, **kwargs):
        self.token_counter = TokenCounter()
        self.walkthrough = None

    @property
    def uid(self):
        return f"WalkthroughAgent"

    @property
    def params(self):
        return {}

    def reset(self, obs, info, env_name):
        # Store the walkthrough in reverse order so we can pop from it.
        if self.walkthrough is None:
            self.walkthrough = info.get("extra.walkthrough")[::-1]

    def act(self, obs, reward, done, info):
        stats = {
            "prompt": None,
            "response": None,
            "nb_tokens": self.token_counter(text=obs),
        }

        if len(self.walkthrough) == 0:
            return "QUIT", stats

        return self.walkthrough.pop(), stats


def build_argparser(parser=None):
    return parser or argparse.ArgumentParser()


register(
    name="walkthrough",
    desc=(
        "This agent will follow the walkthrough provided by the environment."
    ),
    klass=WalkthroughAgent,
    add_arguments=build_argparser,
)
