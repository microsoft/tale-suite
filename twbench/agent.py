class Agent:

    def reset(self, obs, infos):
        pass

    def act(self, obs, reward, done, infos):
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
