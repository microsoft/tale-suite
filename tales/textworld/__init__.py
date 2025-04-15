import gymnasium as gym

from .textworld_data import prepare_twcooking_data
from .textworld_env import TextWorldEnv, TWCookingEnv

environments = []

# TWCookingEnv
for difficulty in range(1, 10 + 1):
    env_name = f"TWCookingLevel{difficulty}"
    environments.append([env_name, "v0"])

    gym.register(
        id=f"tales/{env_name}-v0",
        entry_point="tales.textworld:TWCookingEnv",
        kwargs={"difficulty": difficulty},
    )


def download():
    prepare_twcooking_data()
