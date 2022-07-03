import random

import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


class PipeKind:
    INLET = 0
    PIPE = 1
    OUTLET = 2
    LEAK = 3
    VALVE_OPEN = 4
    VALVE_CLOSED = 5
    WATER = 6
    LEN_TYPES = 7  # with empty for no reason

    @staticmethod
    def kind_repr(kind):
        reprs = [
            "INLET",
            "PIPE",
            "OUTLET",
            "LEAK",
            "V_OPEN",
            "V_CLOSE",
            "WATER",
        ]

        return reprs[kind]


def neighbours_gen(x, y, xlen, ylen):
    if x - 1 >= 0:
        yield x - 1, y
    if x + 1 < xlen:
        yield x + 1, y
    if y - 1 >= 0:
        yield x, y - 1
    if y + 1 < ylen:
        yield x, y + 1


class HydraulicValvesEnv:
    def __init__(self, n=16):
        self.n = n
        self.grid = HydraulicValvesEnv.generate_grid(n, inlets_n=3, outlets_n=5)
        self.inlets = np.transpose(np.nonzero(self.grid[PipeKind.INLET]))
        self.outlets_count = np.sum(np.nonzero(self.grid[PipeKind.OUTLET]))
        self.is_done = False

    def step(self, action):
        assert np.shape(action)[0:2] == np.shape(self.grid)[0:2]
        action = np.rint(action)
        action_indicies = np.transpose(np.nonzero(action))

        score = 0

        # Reset grid watering
        self.grid[:, :, PipeKind.WATER] = 0

        # Flip valves
        for x, y in action_indicies:
            pipe_kind = self.grid[x, y]
            if (
                not pipe_kind[PipeKind.VALVE_CLOSED]
                and not pipe_kind[PipeKind.VALVE_OPEN]
            ):
                score -= 1
                continue

            if pipe_kind[PipeKind.VALVE_CLOSED]:
                self.grid[x, y, PipeKind.VALVE_CLOSED] = 0
                self.grid[x, y, PipeKind.VALVE_OPEN] = 1
            else:
                self.grid[x, y, PipeKind.VALVE_OPEN] = 0
                self.grid[x, y, PipeKind.VALVE_CLOSED] = 1

        # Flood fill pipes
        stack = [tuple(x) for x in [*self.inlets]]
        while stack:
            x, y = stack.pop()
            tile_kind = self.grid[x, y]

            self.grid[x, y, PipeKind.WATER] = 1

            if tile_kind[PipeKind.PIPE] == 0 and tile_kind[PipeKind.VALVE_OPEN] == 0:
                continue

            for n in neighbours_gen(x, y, self.n, self.n):
                nx, ny = n
                ntile_kind = self.grid[nx, ny]

                if ntile_kind[PipeKind.WATER] == 1:
                    continue

                if n in stack:
                    continue

                if (
                    1
                    in ntile_kind[
                        [
                            PipeKind.PIPE,
                            PipeKind.VALVE_OPEN,
                            PipeKind.LEAK,
                            PipeKind.OUTLET,
                        ]
                    ]
                ):
                    stack.append(n)

        flow_score = np.sum(self.grid[PipeKind.WATER]) * 10
        leak_score = np.sum(self.grid[PipeKind.WATER] * self.grid[PipeKind.LEAK]) * -100
        outlets_reached = np.sum(self.grid[PipeKind.WATER] * self.grid[PipeKind.OUTLET])
        goal_score = (outlets_reached / self.outlets_count) * 100

        score += flow_score + leak_score + goal_score

        if outlets_reached >= self.outlets_count:
            self.is_done = True

        return self.grid, score

    def done(self):
        return self.is_done

    def reset(self):
        x = HydraulicValvesEnv(self.n)
        self.grid = x.grid
        self.inlets = x.inlets

    @staticmethod
    def generate_grid(n, inlets_n=20, outlets_n=20):
        assert n > 5

        edges = [(0, x) for x in range(n)]
        edges += [(x, 0) for x in range(1, n)]
        edges += [(x, n - 1) for x in range(n - 1)]
        edges += [(n - 1, x) for x in range(n)]

        inner = []
        for x in range(1, n - 1):
            for y in range(1, n - 1):
                inner.append((x, y))

        random.shuffle(edges)
        random.shuffle(inner)

        inlets = edges[0:inlets_n]
        outlets = edges[inlets_n : inlets_n + outlets_n]

        pipes = []
        pipes_leaks = []
        pipes_valves = []

        def random_pipe_path(inlet, outlet):
            obstacles = np.random.uniform(low=0.0, high=1.0, size=(n, n))
            obstacles = (obstacles > 0.05).astype(int)
            grid = Grid(matrix=obstacles)
            path, runs = AStarFinder().find_path(
                grid.node(*inlet), grid.node(*outlet), grid
            )
            return path

        for inlet, outlet in zip(inlets, outlets):
            path = random_pipe_path(inlet, outlet)

            pipes.extend([x for x in path if (x != inlet and x != outlet)])

            for tile in path:
                if random.random() < 0.005:
                    pipes_leaks.append(tile)
                if random.random() < 0.1:
                    pipes_valves.append(tile)

        grid = np.zeros((n, n, PipeKind.LEN_TYPES))

        for x, y in pipes:
            grid[x, y, PipeKind.PIPE] = 1

        for x, y in inlets:
            grid[x, y, PipeKind.INLET] = 1

        for x, y in outlets:
            grid[x, y, PipeKind.OUTLET] = 1

        for x, y in pipes_leaks:
            grid[x, y, PipeKind.LEAK] = 1

        for x, y in pipes_valves:
            grid[x, y, PipeKind.VALVE_CLOSED] = 1

        grid[:, :, PipeKind.PIPE] = (
            grid[:, :, PipeKind.PIPE] + grid[:, :, PipeKind.LEAK]
        )
        grid[:, :, PipeKind.PIPE] = np.clip(grid[:, :, PipeKind.PIPE], 0.0, 1.0)

        return grid

    @staticmethod
    def sanity_test(grid):
        pass

    def render(self):
        pic = np.zeros((*np.shape(self.grid)[0:2], 3))

        # Inlet = Green
        pic[self.grid[:, :, PipeKind.INLET] == 1] = (0, 1, 0)  # RGB

        # Outlet = Yellow
        pic[self.grid[:, :, PipeKind.OUTLET] == 1] = (1, 1, 0)

        # Pipe = White
        pic[self.grid[:, :, PipeKind.PIPE] == 1] = (1, 1, 1)

        # Valve - closed - redish
        pic[self.grid[:, :, PipeKind.VALVE_CLOSED] == 1] = (0.8, 0.5, 0.5)

        # Valve - opened - greenish
        pic[self.grid[:, :, PipeKind.VALVE_OPEN] == 1] = (0.5, 0.8, 0.5)

        # Water - blueish
        pic[self.grid[:, :, PipeKind.WATER] == 1] -= (0.5, 0.5, 0)

        # Cracked pipe - purple
        pic[self.grid[:, :, PipeKind.LEAK] == 1] = (0.6, 0, 0.6)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.imshow(pic)
        plt.show()


if __name__ == "__main__":
    from csidrl.agents import RandomAgent

    random_agent = RandomAgent(32)
    env = HydraulicValvesEnv(32)

    for g in range(10):
        print(f"Game {g}")
        state = env.reset()
        score = 0
        print("At reset")
        env.render()
        for t in range(100):
            print(f"Turn {t}", end=" ")
            action = random_agent.action(state)
            print(f"Action mean: {np.mean(action)}")
            state, reward = env.step(action)

            score += reward
            print(score)

            if env.done():
                break

        print(f"Game done with score {score}")
        env.render()
