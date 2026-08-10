import gymnasium as gym
import textworld
from textworld.envs.wrappers import Filter

from . import jericho_data


class JerichoEnv(gym.Env):

    def __init__(
        self,
        game,
        admissible_commands=False,
        game_state=False,
        *args,
        **kwargs,
    ):
        gamefile = jericho_data.get_game(game)
        self.game_state = game_state
        self.infos = textworld.EnvInfos(
            score=True,
            max_score=True,
            won=True,
            lost=True,
            feedback=True,
            moves=True,
            admissible_commands=admissible_commands,
            extras=["walkthrough"],
        )
        self.env = textworld.start(gamefile, self.infos, wrappers=[Filter])

    @property
    def native_env(self):
        env = self.env
        while hasattr(env, "_wrapped_env"):
            env = env._wrapped_env
        if not hasattr(env, "_jericho"):
            raise RuntimeError("The TextWorld environment has no Jericho backend")
        return env._jericho

    @staticmethod
    def _serialize_object(obj):
        return {
            "num": obj.num,
            "name": obj.name,
            "parent": obj.parent,
            "sibling": obj.sibling,
            "child": obj.child,
            "attributes": [
                index for index, enabled in enumerate(obj.attr) if enabled
            ],
            "properties": list(obj.properties),
        }

    def get_world_state_hash(self):
        return self.native_env.get_world_state_hash()

    def get_world_objects(self):
        return self.native_env.get_world_objects(clean=True)

    def get_player_object(self):
        player_num = self.native_env.player_obj_num
        return self.get_world_objects()[player_num]

    def get_inventory(self):
        return self.native_env.get_inventory()

    def _augment_info(self, info):
        if not self.game_state:
            return info

        world_objects = self.get_world_objects()
        player_object = world_objects[self.native_env.player_obj_num]
        info["world_state_hash"] = self.get_world_state_hash()
        info["player_object"] = self._serialize_object(player_object)
        info["inventory"] = [
            self._serialize_object(world_objects[obj.num])
            for obj in self.get_inventory()
        ]
        info["world_objects"] = [
            self._serialize_object(obj) for obj in world_objects[1:]
        ]
        return info

    def reset(self, *, seed=None, options=None):
        self.env.seed(seed)
        observation, info = self.env.reset()
        return observation, self._augment_info(info)

    def step(self, action):
        observation, score, done, info = self.env.step(action)
        return observation, score, done, self._augment_info(info)
