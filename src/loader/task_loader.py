import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from utils.graph_utils import random_false_edges, get_graph_T, multigraph2graph
from utils.consts import TLP, NC

def load_task(graph_nx, task, train_skip=1, pivot_time=None, test_size=None):
    '''
    This function is responsible of creating the dataset of the wanted task from the given graph_nx.
    Args:
        graph_nx: networkx - the graph representing the dataset.
                             They are necessary attributes per task:
                             temporal link prediction - must have 'time' attribute for each edge
                             node classification - must have 'label' attribute for each node.
        task: str - name of the task. One of the following:
                        1. temporal_link_prediction
                        2. node_classification
        train_skip: float - ratio of the data we take for train. For example, if we have N possible
                             samples in the given graph_nx, then we take only int(N/train_skip) samples for train.
                             This is highly important in large graphs.
        pivot_time: int - pivot time in the temporal graph. This is relevant only in the case of
                          task=='temporal_link_prediction'.
        test_size: int - size ratio of the test set. This is relevant only in the case of
                         task=='node_classification'.

    Returns:
        X: np.array - the matrix where each entry is a sample with the embeddings.
        y: np.array - the array where y[i] is the label of X[i] for the given task.
    '''
    if task == TLP:
        if pivot_time is None:
            raise Exception('cant create the temporal link prediction task without pivot_time')
        return load_temporal_link_prediction_task(graph_nx, train_skip, pivot_time)
    elif task == NC:
        return load_node_classification_task(graph_nx, train_skip, test_size)
    else:
        raise ('task not available')

def load_link_prediction_task(graph_nx, train_skip, get_true=True, get_false=True, num_false=None):
    '''
    This function is responsible of creating the link_prediction task from the graph.
    Args:
        graph_nx: networkx - the graph we're creating the dataset from. this graph needs to include 'embedding_attrs'
                             attributes for each edge.
        train_skip: float - ratio of the data we take for train. For example, if we have N possible
                             samples in the given graph_nx, then we take only int(N/train_skip) samples for train.
                             This is highly important in large graphs.
        get_true: bool - if True then return true samples, else skip. By default True.
        get_false: bool - if True then return false samples, else skip. By defaule True.
        num_false: int - number of false samples to generate. If None, then |false_samples| will be equal to
                         |true_samples|.

    Returns:
        X: np.array - the matrix where each entry is a sample with the embeddings.
        y: np.array - the array where y[i] is the label of X[i] for the given task.
    '''
    X, y = [], []

    if get_true:
        # true edges
        true_edges = list(graph_nx.edges(data=False))[::train_skip]
        X += true_edges
        y += [1] * len(true_edges)

    if get_false:
        # false edges
        num_false = num_false if num_false is not None else len(y)
        false_edges = list(random_false_edges(graph_nx, num_false))
        X += false_edges
        y += [0] * len(false_edges)

    X = np.array(X)
    y = np.array([y]).T

    return X, y

def load_temporal_link_prediction_task(graph_nx, train_skip, pivot_time):
    '''
    This function is responsible of creating the temporal_link_prediction task from the graph.
    Args:
        graph_nx: networkx - the graph we're creating the dataset from. this graph needs to include 'time' attribute
                             for each edge.
        train_skip: float - ratio of the data we take for train. For example, if we have N possible
                             samples in the given graph_nx, then we take only int(N/train_skip) samples for train.
                             This is highly important in large graphs.
        pivot_time: int - pivot time in the temporal graph.

    Returns:
        X: np.array - the matrix where each entry is a sample with the embeddings.
        y: np.array - the array where y[i] is the label of X[i] for the given task.
    '''
    X, y = {}, {}
    train_graph_nx = multigraph2graph(get_graph_T(graph_nx, max_time=pivot_time))
    X['train'], y['train'] = load_link_prediction_task(train_graph_nx, train_skip)

    # test true edges
    test_graph_nx = get_graph_T(graph_nx, min_time=pivot_time)
    test_graph_nx.remove_nodes_from([node for node in test_graph_nx if node not in train_graph_nx])
    test_graph_nx = multigraph2graph(test_graph_nx)
    X_test_true, y_test_true = load_link_prediction_task(test_graph_nx, 1, get_false=False)
    if len(X_test_true) == 0:
        raise Exception('no samples for the test set in \'temporal link prediction\'. Think of changing the pivot time step.')

    # test false edges
    test_graph_nx = graph_nx.copy()
    test_graph_nx.remove_nodes_from([node for node in test_graph_nx if node not in train_graph_nx])
    test_graph_nx = multigraph2graph(test_graph_nx)
    X_test_false, y_test_false = load_link_prediction_task(test_graph_nx, 1, get_true=False, num_false=len(y_test_true))

    X['test'] = np.concatenate([X_test_true, X_test_false])
    y['test'] = np.concatenate([y_test_true, y_test_false])

    return X, y

def load_node_classification_task(graph_nx, train_skip, test_size):
    '''
    This function is responsible of creating the node_classification task from the graph.
    Args:
        graph_nx: networkx - the graph we're creating the dataset from. this graph needs to include 'label' attribute
                             for each node.
        train_skip: float - ratio of the data we take for train. For example, if we have N possible
                             samples in the given graph_nx, then we take only int(N/train_skip) samples for train.
                             This is highly important in large graphs.
        test_size: int - size ratio of the test set. This is relevant only in the case of
                         task=='node_classification'.
    Returns:
        X: np.array - the matrix where each entry is a sample with the embeddings.
        y: np.array - the array where y[i] is the label of X[i] for the given task.
    '''
    X, y = [], []

    for u, attr in list(graph_nx.nodes(data=True))[::train_skip]:
        X.append(u)
        y.append(attr['label'])

    X = np.array(X)
    y = np.array([y]).T

    y = OneHotEncoder().fit_transform(y).toarray()

    # split to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X = {'train': X_train, 'test': X_test}
    y = {'train': y_train, 'test': y_test}

    return X, y