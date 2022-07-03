import itertools
from datetime import datetime
import time

import torch.nn
from torch.utils.tensorboard import SummaryWriter

from csidrl.deep.meownets import MeowModule
from csidrl.deep import meownets
from csidrl.optimisation import dict_to_tuple_pairs, tuple_pairs_to_dict, cache
from csidrl.visualisation import show_legacy


@cache.memoize(ignore={"dataset", "device", "nn_module"})
def test_args(dataset, epochs, **kwargs):
    neural_network = MeowModule(
        input_shape=(2,),
        output_shape=(1,),
        loss=torch.nn.MSELoss,
        **kwargs,
    )

    neural_network.sanity_check()

    args_repr = ""
    for k, v in kwargs.items():
        if str(k) in ["cache_var", "cache_nn_module"]:
            continue
        args_repr += f"{k}:{v},"

    timestamp = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    writer = SummaryWriter(f"runs/mdt_{args_repr}_{timestamp}")

    print(f"Testing: {args_repr}")
    start = time.time()
    neural_network.train(epochs, dataset, writer, verbose=True)
    end = time.time()

    writer.close()

    test_x = torch.tensor((52, 52), dtype=torch.float)
    test_y = neural_network(test_x).item()

    print(f"Tested output: {test_y}")

    return abs(test_y - 52), end - start


def gridsearch(dataset, options):
    results = {}

    total_scans = len(list(itertools.product(*options)))

    for i, option in enumerate(itertools.product(*options)):
        option_kwargs = tuple_pairs_to_dict(option)
        result = test_args(dataset, exp_id=1, cache_var=3, cache_nn_module=str(option_kwargs["nn_module"]), **option_kwargs)

        if type(result) != list:
            result = [result]

        results[option] = result

        print(f"Scan {i}/{total_scans}")

    return results


if __name__ == "__main__":
    options = {
        "batch_size": [1, 2, 4, 8, 16, 32],
        "epochs": [100, 200, 400, 800, 16000],
        "lr": [0.1, 0.0001, 0.0000001, 0.0000000001],
        "optimizer": [torch.optim.Adam, torch.optim.RMSprop],
    }

    options_debug = {
        "batch_size": [1, 2, 8, 32, 64, 128, 256, 4],
        "epochs": [200],
        "lr": [0.0001, 0.000001, 1e-10],
        "optimizer": [torch.optim.Adam],
        "device": ["cuda"],
        "nn_module": [meownets.DoubleLinear, meownets.LinearRelu, meownets.SingleLinear]
    }

    tuples = dict_to_tuple_pairs(options_debug)

    dataset = []

    for i in range(1000):
        dataset.append(((i / 10, i / 10), i / 10))

    result = gridsearch(dataset, tuples)

    for row in result.items():
        print(row)

    show_legacy(result)
