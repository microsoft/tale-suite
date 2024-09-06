import numpy as np

import textworld


class CustomAgent(textworld.Agent):
    def __init__(self, model, *args, **kwargs):
        self.model = None
        self.seed = kwargs.get('seed', 1234)
        self.context = kwargs.get('context', 100)
        self.rng = np.random.RandomState(self.seed)
        self.conversation = None
        self.window = []
        self.actions = ["north", "south", "east", "west", "up", "down",
                        "look", "inventory", "take all", "YES", "wait",
                        "take", "drop", "eat", "attack"]

    def context_length(self):
        if self.conversation:
            return len(self.conversation.responses[-self.context:])
        else:
            return len(self.window[-self.context:])
        
    def reset(self, env):
        env.display_command_during_render = True

    def act(self, game_state, reward, done):
        action = self.rng.choice(self.actions)
        if action in ["take", "drop", "eat", "attack"]:
            words = game_state.feedback.split()  # Observed words.
            words = [w for w in words if len(w) > 3]  # Ignore most stop words.
            if len(words) > 0:
                action += " " + self.rng.choice(words)

        return str(action), None
