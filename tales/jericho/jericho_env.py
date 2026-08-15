import gymnasium as gym
import textworld
from textworld.envs.wrappers import Filter

from . import jericho_data
from .evaluation import get_winning_walkthrough
from .telemetry import CatalogTelemetry
from .walkthrough_catalog import get_walkthrough


class JerichoEnv(gym.Env):

    def __init__(
        self,
        game,
        admissible_commands=False,
        game_state=False,
        walkthrough_id=None,
        *args,
        **kwargs,
    ):
        gamefile = jericho_data.get_game(game)
        game_info = jericho_data.GAMES_INFOS[game]
        self.game = game
        self.game_selector = game_info["md5"]
        self.walkthrough = get_walkthrough(self.game_selector, walkthrough_id)
        if self.walkthrough is None:
            self.walkthrough = get_winning_walkthrough(self.game_selector)
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
        self.telemetry = CatalogTelemetry(
            self.game_selector, self.native_env.is_fully_supported
        )

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
        if self.walkthrough is not None:
            info["extra.walkthrough"] = list(self.walkthrough.actions)
            info["extra.walkthrough_metadata"] = self.walkthrough.metadata()

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
        if seed is None and self.walkthrough is not None:
            seed = self.walkthrough.seed
        super().reset(seed=seed, options=options)
        self.env.seed(seed)
        observation, info = self.env.reset()
        self.telemetry.reset()
        _, info = self.telemetry.adapt(observation, False, info)
        self.last_score = info["score"]
        return observation, self._augment_info(info)

    def step(self, action):
        observation, _score, done, info = self.env.step(action)
        self.telemetry.step(action)
        done, info = self.telemetry.adapt(observation, done, info)
        reward = info["score"] - self.last_score
        self.last_score = info["score"]
        return observation, reward, done, self._augment_info(info)

    def close(self):
        self.env.close()
