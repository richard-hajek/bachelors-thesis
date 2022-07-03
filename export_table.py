import pandas as pd

def extract_const(df, cols):
    most_used_df = pd.DataFrame(columns=["Parameter", "Constant Value"])
    for col in cols:
        most_used_value = df[col].value_counts().idxmax()
        new_row = pd.DataFrame(
            {"Parameter": [col], "Constant Value": [most_used_value]}
        )
        most_used_df = pd.concat([most_used_df, new_row], ignore_index=True)
    return most_used_df


CONSTS = [
    "iterations",
    "rl_iterations",
    "learning_starts",
    "target_update_interval",
    "train_freq",
    "total_timesteps",
    "buffer_size",
    "tau",
    "gamma",
    "policy"
]

RENAMES = {
    "iterations": "I",
    "rl_iterations": "RLI",
    "total_timesteps": "N",
    "buffer_size": "BS",
    "learning_starts": "LS",
    "target_update_interval": "TUI",
    "gamma": "$\\gamma$",
    "tau": "$\\tau$",
    "train_freq": "TF",
    "exploration_fraction": "EF",
    "exploration_initial_eps": "EIE",
    "exploration_final_eps": "EFE",
    "learning_rate": "LR",
    "batch_size": "Batch S",
    "policy_kwargs.net_arch": "Arch",
    "time_tooks": "T",
    "returns_mean": "$\\bar{R}$",
    "returns_std": "$\\sigma_R$",
    "policy_kwargs.features_extractor_class": "FE",
}

INTS = [
    "time_tooks", "returns_mean", "returns_std"
]

EXPLANATIONS = {
    "Total iterations": "I",
    "Check iterations": "RLI",
    "Total steps per iteration": "N",
    "Replay buffer size": "BS",
    "Deep learning - started at step": "LS",
    "Deep learning - target update interval": "TUI",
    "Discount factor": "$\\gamma$",
    "Soft update coefficient": "$\\tau$",
    "Train frequency": "TF",
    "Exploration factor - ratio of training where interpolated": "EF",
    "Exploration factor - initial": "EIE",
    "Exploration factor - final": "EFE",
    "Deep learning - learning rate": "LR",
    "Deep learning - batch size": "Batch S",
    "Architecture of the policy": "Arch",
    "Training time in seconds": "T",
    "Total return mean": "$\\bar{R}$",
    "Total return standard deviation": "$\\sigma_R$",
    "Neural network backend": "policy"
}
EXPLANATIONS = {v: k for k, v in EXPLANATIONS.items()}

def wrangle_data(path, where=None):
    df = pd.read_json(path, lines=True)

    if where:
        df = where(df)

    keyargs_normalized = pd.json_normalize(df[0])

    returns_mean = df[1].apply(lambda x: x[0][0])
    returns_mean.name = "returns_mean"
    returns_std = df[1].apply(lambda x: x[0][1])
    returns_std.name = "returns_std"
    time_tooks = df[1].apply(lambda x: x[1])
    time_tooks.name = "time_tooks"

    table = pd.concat(
        [keyargs_normalized, returns_mean, returns_std, time_tooks], axis=1
    )

    for i in INTS:
        table[i] = table[i].astype(int)

    consts_table = extract_const(table, CONSTS)

    for k, v in RENAMES.items():
        consts_table = consts_table.replace(k, v)

    for k, v in EXPLANATIONS.items():
        consts_table = consts_table.replace(k, v)

    consts_table = consts_table.rename(columns=RENAMES)

    table = table.drop(columns=CONSTS)
    table = table.rename(columns=RENAMES)

    return consts_table, table


if __name__ == "__main__":
    df_c, df = wrangle_data("log.jsonl")
    legend = pd.DataFrame.from_dict(EXPLANATIONS, orient="index")

    print(df.to_latex(float_format="%.1g"))
    print(df_c.to_latex(float_format="%.1g"))

    print("\n\n\n SATNet extractor\n")
    df_c, df = wrangle_data("log_sat.jsonl")

    print(df.to_latex(float_format="%.1g"))
    print(df_c.to_latex(float_format="%.1g"))
