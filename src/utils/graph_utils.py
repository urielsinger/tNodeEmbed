import numpy as np
import pandas as pd
import networkx as nx

import warnings
from tqdm import tqdm


def random_false_edges(graph_nx, num):
    '''
    Sample false edges from a graph
    Args:
        graph_nx: networkx - the given graph
        num: int - number of false edges to be sampled

    Returns:
        false_edges: list - of all false edges
    '''
    E, N = len(graph_nx.edges()), len(graph_nx.nodes())
    max_num_false_edges = N * (N - 1) - E if graph_nx.is_directed() else N * (N - 1) / 2 - E
    if max_num_false_edges <= 0:
        warnings.warn('not enough false edges for sampling, switching to half of possible samples')
        num = max_num_false_edges / 2
    try:
        all_false_edges = np.array(list(nx.non_edges(graph_nx)))
        false_edges = all_false_edges[np.random.choice(len(all_false_edges), num, replace=False)]
    except:

        false_edges = []
        nodes = graph_nx.nodes()
        with tqdm(total=num, desc='False Edges', unit='false_edge') as pbar:
            while len(false_edges) < num:
                random_edge = sorted(np.random.choice(nodes, 2, replace=False))
                if random_edge[1] not in graph_nx[random_edge[0]] and random_edge not in false_edges:
                    false_edges.append(random_edge)
                    pbar.update(1)

    return false_edges


def get_distance(source, target, graph_nx):
    '''
    distance between 2 nodes
    Args:
        source: str - name of the source node
        target: str - name of the target node
        graph_nx: networkx - the given garph

    Returns:
        the distance between the nodes and 'inf' if one doesnt exist or there is no path between them
    '''
    if source not in graph_nx or target not in graph_nx:
        return np.inf
    try:
        distance = nx.shortest_path(graph_nx, source, target)
    except:
        return np.inf
    return len(distance) - 1


def get_graph_T(graph_nx, min_time=-np.inf, max_time=np.inf, return_df=False):
    '''
    Given a graph with a time attribute for each edge, return the subgraph with only edges between an interval.
    Args:
        graph_nx: networkx - the given graph
        min_time: int - the minimum time step that is wanted. Default value -np.inf
        max_time: int - the maximum time step that is wanted. Default value np.inf
        return_df: bool - if True, return a DataFrame of the edges and attributes,
                          else, a networkx object

    Returns:
        sub_graph_nx: networkx - subgraph with only edges between min_time and max_time
    '''
    relevant_edges = []
    attr_keys = []

    if len(graph_nx.nodes()) == 0:
        return graph_nx

    for u, v, attr in graph_nx.edges(data=True):
        if min_time < attr['time'] and attr['time'] <= max_time:
            relevant_edges.append((u, v, *attr.values()))

            if attr_keys != [] and attr_keys != attr.keys():
                raise Exception('attribute keys in \'get_graph_T\' are different')
            attr_keys = attr.keys()

    graph_df = pd.DataFrame(relevant_edges, columns=['from', 'to', *attr_keys])

    if return_df:
        node2label = nx.get_node_attributes(graph_nx, 'label')
        if len(node2label) > 0:
            graph_df['from_class'] = graph_df['from'].map(lambda node: node2label[node])
            graph_df['to_class'] = graph_df['to'].map(lambda node: node2label[node])
        return graph_df
    else:
        sub_graph_nx = nx.from_pandas_edgelist(graph_df, 'from', 'to', list(attr_keys), create_using=type(graph_nx)())

        # add node attributes
        for node, attr in graph_nx.nodes(data=True):
            if node not in sub_graph_nx:
                continue
            sub_graph_nx.nodes[node].update(attr)

        return sub_graph_nx


def get_graph_times(graph_nx):
    '''
    Return all times in the graph edges attributes
    Args:
        graph_nx: networkx - the given graph

    Returns:
        list - ordered list of all times in the graph
    '''
    return np.sort(np.unique(list(nx.get_edge_attributes(graph_nx, 'time').values())))


def get_node_attribute_matix(graph_nx, attribute, nbunch=None):
    '''
    Given a graph where nodes attributes or vectors, return the matrix of all nodes vectors
    Args:
        graph_nx: networkx - the given graph
        attribute: str - the name of the attribute
        nbunch: list - of nodes we want their vectors. If None then take all nodes in graph

    Returns:
        np.array - matrix with all nodes vectors
    '''
    if nbunch is None:
        nbunch = graph_nx.nodes()
    return np.array([graph_nx.nodes[node][attribute] for node in nbunch])


def get_pivot_time(graph_nx, wanted_ratio=0.2, min_ratio=0.1):
    '''
    Given a graph with 'time' attribute for each edge, calculate the pivot time that gives
    a wanted ratio to the train and test edges
    Args:
        graph_nx: networkx - Graph
        wanted_ratio: float - number between 0 and 1 representing |test|/(|train|+|test|)
        min_ratio: float - number between 0 and 1 representing the minimum value of the expected ratio

    Returns:
        pivot_time: int - the time step that creates such deviation
    '''
    times = get_graph_times(graph_nx)

    if wanted_ratio == 0:
        return times[-1]

    time2dist_from_ratio = {}
    for time in times[int(len(times) / 3):]:
        train_graph_nx = multigraph2graph(get_graph_T(graph_nx, max_time=time))
        num_edges_train = len(train_graph_nx.edges())

        test_graph_nx = get_graph_T(graph_nx, min_time=time)
        test_graph_nx.remove_nodes_from([node for node in test_graph_nx if node not in train_graph_nx])
        test_graph_nx = multigraph2graph(test_graph_nx)
        num_edges_test = len(test_graph_nx.edges())

        current_ratio = num_edges_test / (num_edges_train + num_edges_test)

        if current_ratio <= min_ratio:
            continue

        time2dist_from_ratio[time] = np.abs(wanted_ratio - current_ratio)

    pivot_time = min(time2dist_from_ratio, key=time2dist_from_ratio.get)

    print(f'pivot time {pivot_time}, is close to the wanted ratio by {round(time2dist_from_ratio[pivot_time], 3)}')

    return pivot_time


def multigraph2graph(multi_graph_nx):
    '''
    convert a multi_graph into a graph, where a multi edge becomes a singe weighted edge
    Args:
        multi_graph_nx: networkx - the given multi_graph

    Returns:
        networkx graph
    '''
    if type(multi_graph_nx) == nx.Graph or type(multi_graph_nx) == nx.DiGraph:
        return multi_graph_nx
    graph_nx = nx.DiGraph() if multi_graph_nx.is_directed() else nx.Graph()

    if len(multi_graph_nx.nodes()) == 0:
        return graph_nx

    # add edges + attributes
    for u, v, data in multi_graph_nx.edges(data=True):
        data['weight'] = data['weight'] if 'weight' in data else 1.0

        if graph_nx.has_edge(u, v):
            graph_nx[u][v]['weight'] += data['weight']
        else:
            graph_nx.add_edge(u, v, **data)

    # add node attributes
    for node, attr in multi_graph_nx.nodes(data=True):
        if node not in graph_nx:
            continue
        graph_nx.nodes[node].update(attr)

    return graph_nx


def nodes2embeddings(X, graph_nx, train_time_steps, dimensions, node2embedding=None):
    '''
    Given a np.array from any dimension where the final entry is a nodes name, change the nodes name into the
    nodes embeddings.
    Args:
        X: np.array - from any dimension holding node names
        graph_nx: networkx - the given graph
        train_time_steps: list - of time steps we want their embeddings
        dimensions: int - if embedding doesnt exist, in what size of zero-array to pad with
        node2embedding: dict - from a singe node to it's embedding. If None then it calculated during run time.

    Returns:
        np.array holding insted of node names - their embeddings
    '''
    if isinstance(X, dict):
        for k in X:
            X[k] = nodes2embeddings(X[k], graph_nx, train_time_steps, dimensions)
        return X
    elif not isinstance(X, np.ndarray):
        if node2embedding is not None:
            return node2embedding[X]
        embeddings = []
        for train_time_step in train_time_steps:
            embedding = graph_nx.node[X][train_time_step] if train_time_step in graph_nx.node[X] else np.zeros(dimensions)
            embeddings.append(embedding)
        return np.array(embeddings)
    else:
        embeddings = []
        for x in X:
            embedding = nodes2embeddings(x, graph_nx, train_time_steps, dimensions)
            embeddings.append(embedding)
        return np.array(embeddings)
