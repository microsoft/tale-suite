import gymnasium as gym

from .textworld_env import TextWorldEnv, TWCookingEnv
from .textworld_data import prepare_twcooking_data

environments = []

# TWCookingEnv
for difficulty in range(1, 10 + 1):
    env_name = f"TWCookingLevel{difficulty}"
    environments.append([env_name, "v0"])

    gym.register(
        id=f"twbench/{env_name}-v0",
        entry_point="twbench.textworld:TWCookingEnv",
        kwargs={"difficulty": difficulty},
    )

def download():
    prepare_twcooking_data()
