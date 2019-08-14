import threading
import numpy as np
from math import ceil

from utils import nodes2embeddings
from utils.graph_utils import get_graph_T

class threadsafe_iter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def dataset_generator(X, y, graph_nx, train_time_steps, batch_size, shuffle=True):
    '''
    graph samples embedding generator
    Args:
        X: np.array - of samples, where each sample is a list of node names
        y: np.array - the labels of all samples
        graph_nx: networkx - the graph that holds all the node embeddings
        train_time_steps: list - of all train time steps to be used
        batch_size: int - size of batch
        shuffle: bool - if True then shuffle samples, else False.
    Returns:
        yields samples to be trained on
    '''

    node_embeddings = list(get_graph_T(graph_nx, max_time=max(train_time_steps)).nodes(data=True))[0][1]
    dimensions = max([node_embeddings[train_time_step].shape[0] for train_time_step in train_time_steps if
                      train_time_step in node_embeddings])
    node2embedding = {}
    for i, node in enumerate(graph_nx.nodes):
        node2embedding[node] = nodes2embeddings(node, graph_nx, train_time_steps, dimensions=dimensions)

    steps_per_epoch = ceil(len(X) / batch_size)
    while True:
        if shuffle:
            random_indices = np.arange(len(X))
            np.random.shuffle(random_indices)
            X = X[random_indices]
            y = y[random_indices]
        for step in range(steps_per_epoch):
            cur_X = X[step * batch_size:(step + 1) * batch_size]
            cur_X = nodes2embeddings(cur_X, graph_nx, train_time_steps, dimensions=dimensions, node2embedding=node2embedding)

            cur_y = y[step * batch_size:(step + 1) * batch_size]

            yield cur_X, cur_y
