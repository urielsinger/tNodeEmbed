from math import ceil

import models
import loader

from config import params
from metrics import get_metrics


def run(**kwargs):
    # load graph
    graph_nx, dump_folder = loader.load_dataset(kwargs['dataset'])

    # initialize tNodeEmbed
    task = kwargs['task']
    test_size = kwargs['test_size']
    tnodeembed = models.tNodeEmbed(graph_nx, task=task, dump_folder=dump_folder, test_size=test_size, **kwargs['n2vargs'])

    # load dataset
    X, y = tnodeembed.get_dataset(train_skip=kwargs['train_skip'])

    # fit
    keras_args = kwargs['keras_args']
    batch_size = keras_args.pop('batch_size', 32)
    steps_per_epoch = ceil(len(X['train']) / batch_size)
    # tNodeEmbed
    generator = loader.dataset_generator(X['train'], y['train'], tnodeembed.graph_nx, tnodeembed.train_time_steps, batch_size=batch_size)
    tnodeembed.fit_generator(generator, steps_per_epoch, **keras_args)
    # node2vec
    static_model = models.StaticModel(task=task)
    generator = loader.dataset_generator(X['train'], y['train'], tnodeembed.graph_nx, [max(tnodeembed.train_time_steps)], batch_size=batch_size)
    static_model.fit_generator(generator, steps_per_epoch, **keras_args)


    # predict
    steps = ceil(len(X['test']) / batch_size)
    generator = loader.dataset_generator(X['test'], y['test'], tnodeembed.graph_nx, tnodeembed.train_time_steps, batch_size=batch_size, shuffle=False)
    tnodeembed_metrics = get_metrics(y['test'], tnodeembed.predict_generator(generator, steps))
    generator = loader.dataset_generator(X['test'], y['test'], tnodeembed.graph_nx, [max(tnodeembed.train_time_steps)], batch_size=batch_size, shuffle=False)
    node2vec_metrics = get_metrics(y['test'], static_model.predict_generator(generator, steps))

    print(f'tnodeembed: {tnodeembed_metrics}')
    print(f'node2vec: {node2vec_metrics}')


if __name__ == '__main__':
    run(**params)
