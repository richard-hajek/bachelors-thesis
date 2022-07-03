# Combinatorial Solvers in Deep Reinforcement Learning ( CSiDRL )

## Quickstart

Install `mamba` and `conda`

```bash
bash .scripts/build.sh
```

## Testing

To run a parameter search, run

```bash
conda activate csidrl
python3 src/csidrl/main_rl_rotate_sb3.py
```

On prompting, enter "grid"

```
Using cuda device
Wrapping the env in a DummyVecEnv.
Please select one of ['ipy', 'eval', 'train', 'grid', 'dgrid', 'show', 'save', 'load']: grid
(0/1) Running grid .... Using cuda device
Wrapping the env in a DummyVecEnv.
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 34       |
|    ep_rew_mean      | -34      |
|    exploration rate | 0.987    |
| time/               |          |
|    episodes         | 4        |
|    fps              | 1413     |
|    time_elapsed     | 0        |
|    total timesteps  | 136      |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 35.6     |
|    ep_rew_mean      | -35.6    |
|    exploration rate | 0.973    |
| time/               |          |
|    episodes         | 8        |
|    fps              | 2144     |
|    time_elapsed     | 0        |
|    total timesteps  | 285      |
----------------------------------
....
```