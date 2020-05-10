import itertools
from collections import OrderedDict


def iter_hparams(hparams):
    ordered = OrderedDict(hparams)
    keys = list(ordered.keys())
    values = list(ordered.values())

    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


if __name__ == '__main__':
    for hp in iter_hparams({
        'a': [1, 2, 3],
        'b': [4, 5, 6]
    }):
        print(hp)
