import textworld_express as twx

TASK_NAMES = list(twx.GAME_NAMES)


def get_seeds(split, env=None):
    env = env or twx.TextWorldExpressEnv()
    if split == "train":
        return env.getValidSeedsTrain()
    elif split == "valid":
        return env.getValidSeedsDev()
    elif split == "test":
        return env.getValidSeedsTest()
    else:
        raise NotImplementedError("Only plan to support train, dev, and test splits.")
