from collections import defaultdict

import numpy as np
import pandas as pd
from plotly import express as px
from sklearn.manifold import TSNE


def show(results, x_col, y_col, color=lambda x: x["estimate"] < 100, ignore_desc=None, weights=None):

    if ignore_desc is None:
        ignore_desc = []

    if weights is None:
        weights = defaultdict(lambda: 1)
    else:
        weights = defaultdict(lambda: 1, weights)

    data = to_dataframe(results)
    df = pd.DataFrame(data=data)
    ignore_desc.extend(get_columns_with_same_value(df))
    df['description'] = df.apply(lambda row: '\n'.join([f'{col}={val}' for col, val in row.items() if col not in ignore_desc]), axis=1)
    df["color"] = color(df)

    show_data(df, weights, x_col, y_col)
    show_correlation(df)


def get_columns_with_same_value(df):
    selected_columns = []
    for column in df.columns:
        if df[column].nunique() == 1:
            selected_columns.append(column)
    return selected_columns


def show_legacy(results):

    y_col = "mae"
    x_col = "lr"

    data = to_dataframe(results)
    df = pd.DataFrame(data=data)
    df['description'] = df.apply(lambda row: ', '.join([f'{col}={val}' for col, val in row.items()]), axis=1)

    df["mae"] = abs(df["estimate"] - 52)**2
    df["good"] = df["mae"] < 10

    show_data(df, x_col, y_col)
    show_correlation(df)


def metric(weights, x, y):
    x *= weights
    y *= weights
    return np.linalg.norm(x - y)


def show_data(df, weights_dict, x_col, y_col):
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = remove_nonstd(numeric_df)
    normalized = normalize(numeric_df)

    weights = []
    for c in normalized.columns:
        weights.append(weights_dict[c])

    wmetric = lambda x, y: metric(weights, x, y)

    tsne = TSNE(n_components=2, random_state=42, metric=wmetric)
    data_tsne = tsne.fit_transform(normalized)
    df_tsne = pd.DataFrame(data=data_tsne, columns=["c1", "c2"])
    df_combined = pd.concat([df, df_tsne], axis=1)

    fig = px.scatter(df_combined, x=x_col, y=y_col, hover_name="description", color="color")
    fig.update_layout(title='Results', xaxis_title=x_col, yaxis_title=y_col)
    fig.update_traces(textposition='top center')
    fig.show()


def show_correlation(df):
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = remove_nonstd(numeric_df)
    normalized = normalize(numeric_df)
    corr_df = normalized.corr()
    fig = px.imshow(corr_df)
    fig.update_layout(
        title='Correlation Matrix',
        xaxis=dict(title='Variables'),
        yaxis=dict(title='Variables'),
    )
    fig.show()


def to_dataframe(results):
    data = defaultdict(list)
    for params, result in results.items():
        for param, val in params:
            try:
                val = float(val)
            except (TypeError, ValueError):
                data[param].append(str(val))
                continue

            data[param].append(val)

        score, time = result
        std = None

        try:
            score, std = score
        except:
            pass

        data["time"].append(time)
        data["std"].append(std)
        data["estimate"].append(score)

    return data


def normalize(df):
    return (df-df.mean())/df.std()


def remove_nonstd(df):
    std_dev = df.std()
    zero_std_dev_columns = std_dev[std_dev == 0].index
    return df.drop(zero_std_dev_columns, axis=1)