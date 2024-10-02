class Agent:

    def reset(self, obs, infos):
        pass

    def act(self, obs, reward, done, infos):
        raise NotImplementedError("Child class must implement this method.")
