import sys
from collections import namedtuple
from typing import Optional, Union
import colorama

TurnTuple = namedtuple(
    "TurnTuple", "state action next_state reward terminated trunc inf"
)

eval_returns = []
returns = []

EVAL_ALL = object()
EVAL_NO = object()




def RL_train(
    agent,
    env=None,
    episodes=1,
    evaluation: Union[object, int] = EVAL_NO,
    debug=False,
    env_generator=None,
    seed=42,
    callback=None,
    allow_update=False,
    verbose=False,
    reset_f=lambda x: x.reset(),
):
    agent_update = getattr(agent, "update", None)

    training_results = []
    step_index = 0

    for episode in range(episodes):

        env = env if env else env_generator()

        #print(f"Using seed {seed} ... ", end="")
        observation, info = reset_f(env)  #seed=seed)
        seed += 1

        episode_return = 0

        if evaluation == EVAL_NO:
            episode_evaluation = False
        elif evaluation == EVAL_ALL:
            episode_evaluation = True
        else:
            episode_evaluation = episode % evaluation == 0

        done_trunc = False

        while not done_trunc:
            prev_state_ascii = None
            if debug:
                prev_state_ascii = env.render()

            action = agent.action(observation, evaluation=episode_evaluation)
            next_state, reward, terminated, trunc, info = env.step(action)

            if debug:
                print(f"== STEP {step_index} ==")
                print("State:")
                print(prev_state_ascii)
                print(f"Agent decision: {action}")
                print(f"Agent decision Q val: {agent.what_would_agent_do(observation)}")
                print("Next state:")
                print(env.render())

            if not episode_evaluation:
                agent.observe(
                    observation, action, next_state, reward, terminated or trunc
                )

            if callback:
                callback(
                    TurnTuple(
                        observation, action, next_state, reward, terminated, trunc, info
                    )
                )

            if agent_update and allow_update:
                agent.update(step_index)

            episode_return += reward
            observation = next_state

            done_trunc = terminated or trunc

            step_index += 1

        if verbose:
            print(f"Episode {episode}, return: {episode_return}")

        training_results.append(episode_return)

    return training_results


def RL_eval(
    agent,
    env=None,
    episodes=1,
    env_generator=None,
    seed=42,
    callback=None,):

    return RL_train(agent, env, episodes, evaluation=EVAL_ALL, debug=True, env_generator=env_generator, seed=seed, callback=callback, allow_update=False)



def train_episode(
    env,
    agent,
    evalution=False,
    seed=None,
    render=False,
    env_generator=None,
    callback=None,
    eval_check: Optional[int] = 100,
    debug=False,
):
    if env_generator:
        env = env_generator()

    observation, info = env.reset(seed=seed)

    total_return = 0
    done = False
    i = 0

    has_update = getattr(agent, "update", None)

    while not done:
        if debug:
            prev_state_ascii = env.render(last_action=None)

        action = agent.action(observation, evaluation=evalution)
        next_state, reward, terminated, trunc, info = env.step(action)

        if debug:
            print(f"== STEP {i} ==")
            print("State:")
            print(prev_state_ascii)
            print()
            print(f"Agent decision: {action}")
            print(f"Agent decision Q val: {agent.what_would_agent_do(observation)}")
            print()
            print("Next state:")
            print(env.render())
            print()

        if not evalution:
            agent.observe(observation, action, next_state, reward, terminated or trunc)

        if callback:
            callback(
                TurnTuple(
                    observation, action, next_state, reward, terminated, trunc, info
                )
            )

        if has_update:
            agent.update(i)

        total_return += reward
        observation = next_state

        done = terminated or trunc

        i += 1

        if eval_check is not None and not evalution and i % eval_check:
            train_episode(env, agent, evalution=True, eval_check=None)

    if not evalution:
        returns.append(total_return)
    else:
        eval_returns.append(total_return)

    return total_return


def train_epoch(episode_n, *args, **kwargs):
    total_returns = []
    for i in range(episode_n):
        total_returns.append(train_episode(*args, **kwargs))
        print(f"Epoch score {i}: {total_returns[-1]}")

    return train_episode


def find_good_seed(
    env, agent, better_than, max_attempts, env_generator=None, start_seed=0
):
    for i in range(max_attempts):
        print(f"Attempt {i}/{max_attempts}: ", end="")
        seed = start_seed + i
        total_return = train_episode(
            env, agent, evalution=True, seed=seed, env_generator=env_generator
        )
        print(total_return)

        if total_return > better_than:
            return seed

    return None


def check_seed_consistency(env, agent, max_attempts, env_generator=None, start_seed=0):
    callbacks1 = []
    callbacks2 = []

    def append_and_check(a, x):
        a.append(x)

        check_len = min(len(callbacks1), len(callbacks2))

        assert callbacks1[0:check_len] == callbacks2[0:check_len], f"Failed at {x}"

    for i in range(max_attempts):
        print(f"Attempt {i}/{max_attempts}: ", end="")

        seed = start_seed + i
        train_episode(
            env,
            agent,
            evalution=True,
            seed=seed,
            env_generator=env_generator,
            callback=lambda x: append_and_check(callbacks1, x),
        )
        train_episode(
            env,
            agent,
            evalution=True,
            seed=seed,
            env_generator=env_generator,
            callback=lambda x: append_and_check(callbacks2, x),
        )

        callbacks1.clear()
        callbacks2.clear()

    return None


def console(actions):
    has_save = "save" in actions
    safe_to_quit = True

    if not has_save:
        print(
            "WARN: Could not find a 'save' action, auto saving disabled",
            file=sys.stderr,
        )

    try:
        while True:

            print(colorama.Style.BRIGHT +
                  f"{'* ' if not safe_to_quit else ''}Please select one of {list(actions.keys())}: " +
                  colorama.Style.RESET_ALL, end="")
            print(colorama.Fore.GREEN, end="")
            command = input()
            print(colorama.Fore.RESET, end="")

            action, n = command.split(":") if ":" in command else (command, 1)

            n = int(n)
            safe_to_quit = action == "save"

            for i in range(n):
                if action not in actions:
                    print("No such action")
                    break

                print(f"({i}/{n}) Running {action} .... ", end="")
                ret = actions[action]()
                print(ret)

    except (KeyboardInterrupt, EOFError):
        print("[KeyboardInterrupt/EOFError] Quitting...", file=sys.stderr)

        if not safe_to_quit and has_save:
            print("Running autosave...", file=sys.stderr)
            actions["save"]()


def render_progress():
    import pandas as pd
    import plotly.express as px
    import numpy as np

    data = eval_returns
    bucket = (np.arange(0, len(data)) / 100).astype(int)

    df = pd.DataFrame(data={"bucket": bucket, "data": data})

    # df = px.data.tips()
    fig = px.box(df, x="bucket", y="data", points="suspectedoutliers")
    fig.show()
