import argparse
import sys

import tales
from tales.agent import register
from tales.token import get_token_counter
from tales.utils import format_messages_to_markdown, merge_messages

prompt_toolkit_available = False
try:
    # For command line history and autocompletion.
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import InMemoryHistory

    prompt_toolkit_available = sys.stdout.isatty()
except ImportError:
    pass


class HumanAgent(tales.Agent):

    def __init__(self, *args, **kwargs):
        self.token_counter = get_token_counter()
        self.history = []

        self._history = None
        if prompt_toolkit_available:
            self._history = InMemoryHistory()

    @property
    def uid(self):
        return f"HumanAgent"

    @property
    def params(self):
        return {
            "agent_type": "human",
        }

    def act(self, obs, reward, done, infos):
        available_commands = infos.get("admissible_commands", [])
        if prompt_toolkit_available:
            actions_completer = WordCompleter(
                available_commands, ignore_case=True, sentence=True
            )
            response = prompt(
                "\n> ",
                completer=actions_completer,
                history=self._history,
                enable_history_search=True,
            )
        else:
            if available_commands:
                print("Available actions: {}\n".format(available_commands))

            response = input("\n> ")

        messages = self.build_messages(f"{obs}\n> ")
        # response = self._llm_call_from_messages(
        #     messages,
        #     temperature=self.act_temp,
        #     max_tokens=100,  # Text actions are short phrases.
        #     seed=self.seed,
        #     stream=False,
        # )

        action = response.strip()
        self.history.append((f"{obs}\n> ", f"{action}\n"))

        # Compute usage statistics
        stats = {
            "prompt": format_messages_to_markdown(messages),
            "response": response,
            "nb_tokens": self.token_counter(messages=messages, text=response),
        }

        return action, stats

    def build_messages(self, observation):
        messages = []

        for i, (obs, action) in enumerate(self.history):
            messages.append({"role": "user", "content": obs})
            messages.append({"role": "assistant", "content": action})

        messages.append({"role": "user", "content": observation})

        # Just in case, let's avoid having multiple messages from the same role.
        messages = merge_messages(messages)

        return messages


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("HumanAgent settings")
    return parser


register(
    name="human",
    desc=("Manually decide which action to take."),
    klass=HumanAgent,
    add_arguments=build_argparser,
)
