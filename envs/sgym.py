import gymnasium as gym
import numpy as np

class SafetyGymnasium(gym.Env):
    metadata = {}

    def __init__(self, name, action_repeat=1, size=(64, 64), camera='fixednear', seed=0):
        import safety_gymnasium
        
        self._action_repeat = action_repeat
        self._size = size
        self._name = name
        
        # Initialize safety gymnasium environment
        self._env = safety_gymnasium.make(name, render_mode='rgb_array', width=size[0], height=size[1], camera_name=camera)

    @property
    def observation_space(self):
        spaces = {}
        if isinstance(self._env.observation_space, gym.spaces.Dict):
            for key, value in self._env.observation_space.spaces.items():
                if len(value.shape) == 0:
                    shape = (1,)
                else:
                    shape = value.shape
                spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        else:
            spaces['state'] = gym.spaces.Box(-np.inf, np.inf, self._env.observation_space.shape, dtype=np.float32)
            
        # We replace visual observation with 'image' explicitly
        spaces['image'] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return self._env.action_space

    def render(self, *args, **kwargs):
        # We assume render_mode='rgb_array' was passed on init
        return self._env.render()

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        cost_sum = 0
        for _ in range(self._action_repeat):
            obs, r, cost, term, trunc, info = self._env.step(action)
            reward += r or 0
            cost_sum += cost or 0
            if term or trunc:
                break
        
        # Pipelining visual output into generic 'image'
        image = self.render()
        
        new_obs = {}
        if isinstance(obs, dict):
            for key, val in obs.items():
                new_obs[key] = np.array([val], dtype=np.float32) if np.isscalar(val) else np.array(val, dtype=np.float32)
        else:
            new_obs['state'] = np.array([obs], dtype=np.float32) if np.isscalar(obs) else np.array(obs, dtype=np.float32)
            
        new_obs['image'] = image
        new_obs['is_terminal'] = term
        new_obs['is_first'] = False
        new_obs['is_last'] = term or trunc
        
        # Pass cost info in info so agent can log it if needed
        info["cost"] = cost_sum
        done = term or trunc
        if "discount" not in info:
            info["discount"] = np.array(1.0 - float(term)).astype(np.float32)
            
        return new_obs, float(reward), done, info

    def reset(self, **kwargs):
        obs, info = self._env.reset(seed=kwargs.get('seed', None))
        image = self.render()
        
        new_obs = {}
        if isinstance(obs, dict):
            for key, val in obs.items():
                new_obs[key] = np.array([val], dtype=np.float32) if np.isscalar(val) else np.array(val, dtype=np.float32)
        else:
            new_obs['state'] = np.array([obs], dtype=np.float32) if np.isscalar(obs) else np.array(obs, dtype=np.float32)
            
        new_obs['image'] = image
        new_obs['is_terminal'] = False
        new_obs['is_first'] = True
        new_obs['is_last'] = False
        return new_obs
