import os
from os.path import join

import networkx as nx
import numpy as np
from tqdm import tqdm
from node2vec import Node2Vec
from scipy.linalg import orthogonal_procrustes

from keras.layers import Input, LSTM, Dense, Activation, Concatenate, Lambda
from keras.models import Model

import loader
from models.task_model import TaskModel
from utils.graph_utils import get_graph_T, get_pivot_time, get_graph_times, multigraph2graph
from utils.general_utils import load_object, save_object
from utils.consts import TLP, NC

class tNodeEmbed(TaskModel):
    def __init__(self, graph_nx, task, dump_folder, test_size=0, align=True, **n2vargs):
        '''
        tNodeEmbed init
        Args:
            graph_nx: networkx - holding the temporal graph
            task: str - name of the task. either 'temporal_link_prediction' or 'node_classification'
            dump_folder: string - link to a dump folder of the graph dataset, in order for future runs to run faster
            test_size: folat - the wanted size of test_size
            align: bool - True if alignment is wanted, else False
            **n2vargs: dict - node2vec arguments
        '''
        super(tNodeEmbed, self).__init__(task=task)

        self.dump_folder = dump_folder
        self.test_size = test_size
        self.align = align
        self.n2vargs = n2vargs

        self.graph_nx = graph_nx
        times = get_graph_times(self.graph_nx)

        if self.task == TLP:
            self.pivot_time = self.calculate_pivot_time()
        else:
            self.pivot_time = times[-1]

        self.train_time_steps = times[times <= self.pivot_time]

        # initialize
        self.graph_nx = self.initialize()

    def get_dataset(self, train_skip=1):
        '''
        This function is responsible of creating the dataset of the wanted task from the given graph_nx. Wraps the
        function 'task_loader.load_task' for caching.
        Args:
            train_skip: float - ratio of the data we take for train. For example, if we have N possible
                                 samples in the given graph_nx, then we take only int(N/train_skip) samples for train.
                                 This is highly important in large graphs.
        Returns:
            X: dict - with keys 'train', 'test'. each value is a np.array of the dataset, where each entery is a sample
                      with the embeddings.
            y: dict - with keys 'train', 'test', each value is a np.array of the dataset, where y[key][i] is the label
                      of X[key][i] for the given task.
        '''
        # load task data
        task_data_path = join(self.dump_folder, f'{self.task}_dataset_{self.pivot_time}.data')
        if os.path.exists(task_data_path):
            X, y = load_object(task_data_path)
        else:
            X, y = loader.load_task(self.graph_nx, self.task, train_skip=1, pivot_time=self.pivot_time,
                                    test_size=self.test_size)
            save_object((X, y), task_data_path)

        X = {'train': X['train'][::train_skip], 'test': X['test']}
        y = {'train': y['train'][::train_skip], 'test': y['test']}

        return X, y

    @staticmethod
    def _get_model(task, input_shape, latent_dim=128, num_classes=1):
        '''
        Given the task, return the desired architecture of training
        Args:
            task: string - of the tasks name, either 'temporal_link_prediction' or 'node_classification'
            input_shape: tuple - shape of a singe sample
            latent_dim: int - the size of the LSTM latent space
            num_classes: int - number of classes. Relevant only if task=='node_classification'
        Returns:
            keras model of tNodeEmbed
        '''
        if task == TLP:
            inputs = Input(shape=input_shape)

            lmda_lyr1 = Lambda(lambda x: x[:, 0, :, :], output_shape=input_shape[1:])(inputs)
            lmda_lyr2 = Lambda(lambda x: x[:, 1, :, :], output_shape=input_shape[1:])(inputs)

            lstm_lyr = LSTM(latent_dim, return_sequences=False, activation='relu')
            lstm_lyr1 = lstm_lyr(lmda_lyr1)
            lstm_lyr2 = lstm_lyr(lmda_lyr2)
            concat_lyr = Concatenate(axis=-1)([lstm_lyr1, lstm_lyr2])

            fc_lyr1 = Dense(latent_dim, activation='relu')(concat_lyr)
            fc_lyr2 = Dense(1)(fc_lyr1)
            soft_lyr = Activation('sigmoid')(fc_lyr2)

            model = Model(inputs, soft_lyr)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        elif task == NC:
            inputs = Input(shape=input_shape)

            lstm_lyr = LSTM(latent_dim)(inputs)

            fc_lyr = Dense(num_classes)(lstm_lyr)
            soft_lyr = Activation('softmax')(fc_lyr)

            model = Model(inputs, soft_lyr)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            raise Exception('unknown task for _get_model')
        return model

    def calculate_pivot_time(self):
        '''
        Calculate the pivot time that is needed in order to create a 'time_split_ratio' between train edges and
        test edges
        Returns:
            time step representing the pivot time step
        '''
        ratio2pivot = {}
        ratio2pivot_path = join(self.dump_folder, 'ratio2pivot.dict')
        if os.path.exists(ratio2pivot_path):
            ratio2pivot = load_object(ratio2pivot_path)
            if self.test_size in ratio2pivot:
                return ratio2pivot[self.test_size]
        pivot_time = get_pivot_time(self.graph_nx, self.test_size)
        ratio2pivot[self.test_size] = pivot_time
        save_object(ratio2pivot, ratio2pivot_path)
        return pivot_time

    def initialize(self):
        '''
        initialize the model by calculating embeddings per each time step and aligning them all together
        Returns:
            initialized netwrokx graph
        '''
        init_path = join(self.dump_folder, 'init.emb')
        if os.path.exists(init_path):
            graph_nx = nx.read_gpickle(init_path)
        else:
            # initialize embeddings
            graph_nx = tNodeEmbed._initialize_embeddings(self.graph_nx, self.train_time_steps, **self.n2vargs)
            nx.write_gpickle(graph_nx, init_path)

        if self.align:
            # initialize alignment
            graph_nx = tNodeEmbed._align_embeddings(graph_nx, self.train_time_steps)

        return graph_nx

    @staticmethod
    def _initialize_embeddings(graph_nx, times=None, **n2vargs):
        '''
        Given a graph, learn embeddings for all nodes in all time step. Attribute 'time' for each edge mast appear
        Args:
            graph_nx: networkx - the given graph
            times: list - of the times we want to work with, if None then calculates for all times in the graph
            **n2vargs: node2vec arguments

        Returns:
            graph_nx: networkx - but with attributes for each node for each time step of its embedding
        '''
        if times is None:
            times = get_graph_times(graph_nx)

        for time in tqdm(times, desc='Time Steps', unit='time_step'):
            cur_graph_nx = get_graph_T(graph_nx, max_time=time)
            cur_graph_nx = multigraph2graph(cur_graph_nx)

            node2vec = Node2Vec(graph=cur_graph_nx, quiet=True, **n2vargs)
            ntv_model = node2vec.fit()
            ntv = {node: ntv_model[str(node)] for node in cur_graph_nx.nodes()}
            nx.set_node_attributes(graph_nx, ntv, time)

        return graph_nx

    @staticmethod
    def _align_embeddings(graph_nx, times=None):
        '''
        Given a graph that went through 'initialize_embeddings', align all time step embeddings to one another
        Args:
            graph_nx: networkx - the given graph
            times: list - of the times we want to work with, if None then calculates for all times in the graph

        Returns:
            graph_nx: networkx - with aligned embeddings in its attributes for each node
        '''
        if times is None:
            times = get_graph_times(graph_nx)

        node2Q_t_1 = nx.get_node_attributes(graph_nx, times[0])
        Q_t_1 = np.array([node2Q_t_1[node] for node in node2Q_t_1])
        for time in times[1:]:
            node2Q_t = nx.get_node_attributes(graph_nx, time)
            Q_t = np.array([node2Q_t[node] for node in node2Q_t_1])
            R_t, _ = orthogonal_procrustes(Q_t, Q_t_1)
            Q_t = np.array([node2Q_t[node] for node in node2Q_t])
            R_tQ_t = np.dot(Q_t, R_t)
            node2R_tQ_t = {node: vec for node, vec in zip(node2Q_t, R_tQ_t)}
            nx.set_node_attributes(graph_nx, node2R_tQ_t, time)
            node2Q_t_1 = node2R_tQ_t
            Q_t_1 = R_tQ_t

        return graph_nx
