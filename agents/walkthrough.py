import argparse

import tales
from tales.agent import register
from tales.token import get_token_counter


class WalkthroughAgent(tales.Agent):
    def __init__(self, **kwargs):
        self.token_counter = get_token_counter()
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
    desc=("This agent will follow the walkthrough provided by the environment."),
    klass=WalkthroughAgent,
    add_arguments=build_argparser,
)
