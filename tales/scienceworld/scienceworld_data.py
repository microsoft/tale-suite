import scienceworld


def get_task_names():
    return scienceworld.ScienceWorldEnv().task_names


def get_variations(task_name, split, env=None):
    env = env or scienceworld.ScienceWorldEnv(task_name)
    if split == "train":
        return env.get_variations_train()
    elif split == "valid":
        return env.get_variations_dev()
    elif split == "test":
        return env.get_variations_test()
    else:
        raise NotImplementedError("Only plan to support train, dev, and test splits.")
