class Agent:

    def reset(self, obs, info, env):
        pass

    def act(self, obs, reward, done, info):
        raise NotImplementedError("Child class must implement this method.")

    @property
    def uid(self):
        """Unique identifier for this agent.

        Usually, this is a string that contains the class name and the values of the
        parameters used to initialize the agent.
        """
        # return f"{self.__class__.__name__}_" + "_".join(
        #     f"{k}:{v}" for k, v in self.kwargs.items()
        # ).strip("_")
        raise NotImplementedError("Child class must implement this property.")

    @property
    def params(self):
        """Parameters used to initialize the agent.

        Returns:
            dict: Parameters used to initialize the agent.
        """
        # return self.kwargs
        raise NotImplementedError("Child class must implement this property.")


# Registry for available agents to benchmark.
AGENTS = {}


def register(name: str, desc: str, klass: callable, add_arguments: callable) -> None:
    """ Register a new type of Agent.

    Arguments:
        name:
            Name of the agent (must be unique).
        desc:
            Bried description of how the agent works (for `benchmark.py --help`).
        klass:
            Class used to instantiate the agent.
        add_arguments:
            Function that should add the `argparse` arguments needed for this agent.
            The provided function should expect a `argparse.ArgumentParser` object.

    Example:

        >>> from tales.agent import register
        >>> from tales.agents import RandomAgent
        >>> def _add_arguments(parser):
                parser.add_argument("--seed", required=True, type=int,
                                    help="Random seed to use.")
        >>> \
        >>> register(name="random",
        >>>          desc="This agent randomly select actions.",
        >>>          klass=RandomAgent,
        >>>          add_arguments=_add_arguments)
    """
    if name in AGENTS:
        raise ValueError(f"Agent '{name}' already registered.")

    AGENTS[name] = (desc, klass, add_arguments)
