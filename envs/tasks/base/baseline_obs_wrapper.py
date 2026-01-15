from gym import Wrapper
import gym.spaces as spaces
import numpy as np
import cv2
from abc import ABC
from collections import OrderedDict

BASIC_ACTIONS = {
    "noop": dict(),
    "attack": dict(attack=np.array(1)),
    "turn_up": dict(camera=np.array([-10.0, 0.])),
    "turn_down": dict(camera=np.array([10.0, 0.])),
    "turn_left": dict(camera=np.array([0., -10.0])),
    "turn_right": dict(camera=np.array([0., 10.0])),
    "forward": dict(forward=np.array(1)),
    "back": dict(back=np.array(1)),
    "left": dict(left=np.array(1)),
    "right": dict(right=np.array(1)),
    "jump": dict(jump=np.array(1), forward=np.array(1)),
    "use": dict(use=np.array(1)),
}

NOOP_ACTION = {
    'camera': np.array([0., 0.]), 
    'smelt': 'none', 
    'craft': 'none', 
    'craft_with_table': 'none', 
    'forward': np.array(0), 
    'back': np.array(0), 
    'left': np.array(0), 
    'right': np.array(0), 
    'jump': np.array(0), 
    'sneak': np.array(0), 
    'sprint': np.array(0), 
    'use': np.array(0), 
    'attack': np.array(0), 
    'drop': 0, 
    'swap_slot': OrderedDict([('source_slot', 0), ('target_slot', 0)]), 
    'pickItem': 0, 
    'hotbar.1': 0, 
    'hotbar.2': 0, 
    'hotbar.3': 0, 
    'hotbar.4': 0, 
    'hotbar.5': 0, 
    'hotbar.6': 0, 
    'hotbar.7': 0, 
    'hotbar.8': 0, 
    'hotbar.9': 0,
}


class BaselineObsWrapper(Wrapper, ABC):
    """
    Wrapper for DreamerV3 baseline mode.
    Returns only standard observation fields without LS-Imagine specific fields.
    No heatmap, no jump, no zoomed, no intrinsic, no score.
    """
    def __init__(
        self,
        env,
        repeat=1,
        sticky_attack=0,
        sticky_jump=10,
        pitch_limit=(-70, 70),
        store_vlm_rgb: bool = False,
        vlm_rgb_key: str = "vlm_rgb",
    ):
        super().__init__(env)
        self.wrapper_name = "BaselineObsWrapper"
        self._store_vlm_rgb = bool(store_vlm_rgb)
        self._vlm_rgb_key = str(vlm_rgb_key)

        self._noop_action = NOOP_ACTION
        actions = self._insert_defaults(BASIC_ACTIONS)
        self._action_names = tuple(actions.keys())
        self._action_values = tuple(actions.values())

        # Standard DreamerV3 observation space - NO LS fields
        obs_spaces = {
            'image': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            # MPF-LSD: frozen MineCLIP text-conditioned embedding stored in replay.
            'mc_e': spaces.Box(-np.inf, np.inf, (512,), dtype=np.float32),
            'inventory': spaces.Box(-np.inf, np.inf, (41,), dtype=np.float32),
            'inventory_max': spaces.Box(-np.inf, np.inf, (41,), dtype=np.float32),
            'equipped': spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32),
            'health': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            'hunger': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            'breath': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            'obs_reward': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            'is_first': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            'is_last': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            'is_terminal': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
        }
        if self._store_vlm_rgb:
            # Raw MineDojo RGB (C,H,W) at the native 160x256 used by MineCLIP.
            obs_spaces[self._vlm_rgb_key] = spaces.Box(low=0, high=255, shape=(3, 160, 256), dtype=np.uint8)
        self.observation_space = spaces.Dict(obs_spaces)

        self.action_space = spaces.discrete.Discrete(len(BASIC_ACTIONS))
        self.action_space.discrete = True
        self._repeat = repeat
        self._sticky_attack_length = sticky_attack
        self._sticky_attack_counter = 0
        self._sticky_jump_length = sticky_jump
        self._sticky_jump_counter = 0
        self._pitch_limit = pitch_limit
        self._pitch = 0

    def reset(self):
        obs = self.env.reset()
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        obs = self._obs(obs)

        self._sticky_attack_counter = 0
        self._sticky_jump_counter = 0
        self._pitch = 0
        return obs

    def step(self, action):
        import copy
        action = copy.deepcopy(self._action_values[action])
        action = self._action(action)
        following = self._noop_action.copy()
        for key in ("attack", "forward", "back", "left", "right"):
            following[key] = action[key]
        for act in [action] + ([following] * (self._repeat - 1)):
            obs, reward, done, info = self.env.step(act)
            if "error" in info:
                done = True
                break
        obs["is_first"] = False
        obs["is_last"] = bool(done)
        obs["is_terminal"] = bool(info.get("is_terminal", info.get("real_done", done)))
        obs = self._obs(obs)

        return obs, reward, done, info

    def _obs(self, obs):
        """Process observation to standard DreamerV3 format - NO LS fields"""
        # DEBUG PRINT
        # if obs.get('is_first', False):
        #     print(f"DEBUG: Observation keys: {list(obs.keys())}")
        #     for k in ['inventory', 'inventory_max', 'equipment', 'equipped_items', 'life_stats']:
        #         if k in obs:
        #             print(f"DEBUG: key '{k}' type: {type(obs[k])}")
        #             if isinstance(obs[k], dict):
        #                 print(f"DEBUG: key '{k}' subkeys: {list(obs[k].keys())}")

        image = obs['rgb']  # 3 * H * W
        image = image.transpose(1, 2, 0).astype(np.uint8)  # H * W * 3
        image = cv2.resize(image, (64, 64))  # 64 * 64 * 3

        def get_quantity(x, shape):
            if isinstance(x, dict) and 'quantity' in x:
                val = np.array(x['quantity'], dtype=np.float32)
            elif isinstance(x, (np.ndarray, list)):
                val = np.array(x, dtype=np.float32)
            else:
                val = np.zeros(shape, dtype=np.float32)
            
            if val.size != np.prod(shape):
                # pad or truncate
                res = np.zeros(shape, dtype=np.float32)
                size = min(val.size, res.size)
                res.flat[:size] = val.flat[:size]
                return res
            return val.reshape(shape)

        # Standard DreamerV3 observations only
        processed_obs = {
            'image': image,
            # Pass through MineCLIP embedding if present (produced by ClipWrapper).
            'mc_e': np.array(obs.get('mc_e', np.zeros((512,), dtype=np.float32)), dtype=np.float32),
            'inventory': get_quantity(obs.get('inventory'), (41,)),
            'inventory_max': get_quantity(obs.get('inventory_max'), (41,)),
            'equipped': get_quantity(obs.get('equipped_items', obs.get('equipped')), (8,)),
            'health': np.array(obs.get('life_stats', {}).get('life', [20.0]), dtype=np.float32).reshape(1) if 'life_stats' in obs else np.zeros(1, dtype=np.float32),
            'hunger': np.array(obs.get('life_stats', {}).get('food', [20.0]), dtype=np.float32).reshape(1) if 'life_stats' in obs else np.zeros(1, dtype=np.float32),
            'breath': np.array(obs.get('life_stats', {}).get('air', [300.0]), dtype=np.float32).reshape(1) if 'life_stats' in obs else np.zeros(1, dtype=np.float32),
            'obs_reward': np.array(obs.get('obs_reward', [0.0]), dtype=np.float32).reshape(1),
            'is_first': obs['is_first'],
            'is_last': obs['is_last'],
            'is_terminal': obs['is_terminal'],
        }
        if self._store_vlm_rgb:
            # Keep raw CHW uint8 for training-time MineCLIP embedding recomputation.
            processed_obs[self._vlm_rgb_key] = np.array(obs.get("rgb"), dtype=np.uint8)

        # Validate all fields
        for key, value in processed_obs.items():
            if key in self.observation_space:
                if not isinstance(value, np.ndarray):
                    processed_obs[key] = np.array(value)

        return processed_obs

    def _action(self, action):
        if self._sticky_attack_length:
            if action["attack"]:
                self._sticky_attack_counter = self._sticky_attack_length
            if self._sticky_attack_counter > 0:
                action["attack"] = np.array(1)
                action["jump"] = np.array(0)
                self._sticky_attack_counter -= 1
        if self._sticky_jump_length:
            if action["jump"]:
                self._sticky_jump_counter = self._sticky_jump_length
            if self._sticky_jump_counter > 0:
                action["jump"] = np.array(1)
                action["forward"] = np.array(1)
                self._sticky_jump_counter -= 1
        if self._pitch_limit and action["camera"][0]:
            lo, hi = self._pitch_limit
            if not (lo <= self._pitch + action["camera"][0] <= hi):
                action["camera"] = (0, action["camera"][1])
            self._pitch += action["camera"][0]
        return action

    def _insert_defaults(self, actions):
        actions = {name: action.copy() for name, action in actions.items()}
        for key, default in self._noop_action.items():
            for action in actions.values():
                if key not in action:
                    action[key] = default
        return actions

