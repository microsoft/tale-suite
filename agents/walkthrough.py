import argparse

import tales
from tales.agent import register
from tales.token import get_token_counter


class WalkthroughExhausted(Exception):
    """Raised when the walkthrough has no more actions but the game isn't won."""

    pass


class WalkthroughAgent(tales.Agent):
    def __init__(self, **kwargs):
        self.token_counter = get_token_counter()
        self.walkthrough = []

    @property
    def uid(self):
        return "WalkthroughAgent"

    @property
    def params(self):
        return {}

    def reset(self, obs, info, env_name):
        raw = info.get("extra.walkthrough")
        if raw is None:
            raise ValueError(f"{env_name} does not provide 'extra.walkthrough' in info")
        # Store in reverse order so we can pop from the end.
        self.walkthrough = list(raw)[::-1]

    def act(self, obs, reward, done, info):
        stats = {
            "prompt": None,
            "response": None,
            "nb_tokens": self.token_counter(text=obs),
            "nb_tokens_prompt": self.token_counter(text=obs),
            "nb_tokens_response": 0,
            "nb_tokens_thinking": 0,
        }

        if len(self.walkthrough) == 0:
            raise WalkthroughExhausted(
                "Walkthrough actions exhausted before the game was won."
            )

        return self.walkthrough.pop(), stats


def build_argparser(parser=None):
    return parser or argparse.ArgumentParser()


register(
    name="walkthrough",
    desc=("This agent will follow the walkthrough provided by the environment."),
    klass=WalkthroughAgent,
    add_arguments=build_argparser,
)
