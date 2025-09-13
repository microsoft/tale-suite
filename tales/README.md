# Training and Testing on TALES
TALES offers both train splits and test splits, the latter of which make up the games all models in our technical report were evaluated on.

The following is an example of how to import desired environments and allow an agent to play through them.

Note that importing the relevant framework automatically registers all environments in that framework with gym.
You can individually import the frameworks if you want to only evaluate on them one at a time.
For now, we do not include a jericho train split.

```
import gymnasium as gym
from tales import *

# Training splits
train_envs = [env_spec.id for env_spec in gym.envs.registry.values() if "tales/" in env_spec.id and 'train' in env_spec.id]

# Testing splits
envs = [env_spec.id for env_spec in gym.envs.registry.values() if "tales/" in env_spec.id and 'train' not in env_spec.id]

train_env = gym.make(
    train_envs[0],
    disable_env_checker=True,
    admissible_commands=True,
)

test_env = gym.make(
    envs[0],
    disable_env_checker=True,
    admissible_commands=True,
)
```