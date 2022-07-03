import gym

class GoRightEnv(gym.Env):

    def __init__(self, states=100, limit=100, make_box=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = gym.spaces.Discrete(2)

        if make_box:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        else:
            self.observation_space = gym.spaces.Discrete(states)

        self.state = 0
        self.end_state = states - 1
        self.step_count = 0
        self.limit = limit

        self.max_visited = 0

    def step(self, action):
        if action == 0:
            self.state = max(self.state - 1, 0)
        else:
            self.state = min(self.state + 1, self.end_state)

        self.step_count += 1
        done = self.state == self.end_state

        # reward = 100 if done else -1
        # reward = 100 if done else action

        if done:
            reward = 100
        else:
            reward = -1

        if self.state >= self.max_visited:
            reward += 2
            self.max_visited = self.state

        if self.step_count > self.limit:
            return self.state, 0, done, True, {}

        return self.state, reward, done, False, {}

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.state = 0
        self.step_count = 0
        self.max_visited = 0
        return self.state, {}

    def render(self, mode="human", **kwargs):
        return self.state

