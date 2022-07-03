import itertools
from typing import List

import diskcache


def dict_to_tuple_pairs(d: dict) -> List:
    result = []

    for k, v_list in d.items():
        subresult = []
        for v in v_list:
            subresult.append((k, v))
        result.append(subresult)

    return result


def tuple_pairs_to_dict(pairs) -> dict:
    result = {}

    for k, v in pairs:
        result[k] = v

    return result


def it_count(iterator):
    return sum(1 for _ in iterator)


def gridsearch(estimator, options, estimation_callback=None):
    results = {}

    options = dict_to_tuple_pairs(options)
    total_scans = it_count(itertools.product(*options))

    for i, option in enumerate(itertools.product(*options)):
        option_kwargs = tuple_pairs_to_dict(option)
        result = estimator(**option_kwargs)

        if estimation_callback:
            estimation_callback(option_kwargs, result, i, total_scans)

        print(f"Scan {i}/{total_scans}")

    return results


cache = diskcache.Cache(directory=".cache")
