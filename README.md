# tNodeEmbed

This repository provides a reference implementation of *tNodeEmbed* as described in the paper:<br>
> Node Embedding over Temporal Graphs.<br>
> Uriel Singer, Ido Guy and Kira Radinsky.<br>
> International Joint Conferences on Artificial Intelligence, 2019.<br>
> https://www.ijcai.org/proceedings/2019/0640.pdf<Insert paper link>

The *tNodeEmbed* algorithm learns temporal representations for nodes in any (un)directed, (un)weighted temporal graph.
For Further explanation of tNodeEmbed please visit it's blog in [Medium](https://medium.com/@urielsinger/tnodeembed-node-embedding-over-temporal-graphs-b7bcbf59938f). 

### Requirements
 - python>=3.6
 - networkx
 - numpy
 - tqdm
 - pandas
 - keras
 - matplotlib
 - node2vec
 - sklearn

### Basic Usage

Start by creating a networkx graph where each edge has a 'time' attribute. Given a DataFrame with 'source','target' and 'time' columns, you can execute the following:
```python
graph_nx = loader.dataset_loader.df2graph(graph_df, source, target, time, create_using=nx.Graph())
```

Continue by initializing a *tNodeEmbed* model by writing:<br/>
```python
tnodeembed = models.tNodeEmbed(graph_nx, task=task, dump_folder=dump_folder)
```
Where task can be either 'temporal_link_prediction' or 'node_classification'. The dump_folder is for future runnning times.



Then create your task dataset by writing:<br/>
```python
X, y = tnodeembed.get_dataset()
```
Where X and y are dictionaries with keys 'train' and 'test'.



Training time!
```python
X['train'] = graph_utils.nodes2embeddings(X['train'], tnodeembed.graph_nx, tnodeembed.train_time_steps)
tnodeembed.fit(X['train'] ,y['train'])
```
Or by using a generator: 
```python
steps_per_epoch = ceil(len(X['train']) / batch_size)
generator = loader.dataset_generator(X['train'], y['train'], tnodeembed.graph_nx, tnodeembed.train_time_steps, batch_size=batch_size)
tnodeembed.fit_generator(generator, steps_per_epoch)
```
    
And prediction:
```python
X['test'] = graph_utils.nodes2embeddings(X['test'], tnodeembed.graph_nx, tnodeembed.train_time_steps)
tnodeembed.predict(X['test'])
```
Or by using a generator:
```python
steps = ceil(len(X['test']) / batch_size)
generator = loader.dataset_generator(X['test'], y['test'], tnodeembed.graph_nx, tnodeembed.train_time_steps, batch_size=batch_size, shuffle=False)
tnodeembed.predict_generator(generator, steps)
```

A full flow example and comparission to node2vec can be found in [``main.py``](src/main.py)

### Citing
If you find *tNodeEmbed* useful for your research, please consider citing the following paper:

	@inproceedings{ijcai2019-640,
	  title     = {Node Embedding over Temporal Graphs},
	  author    = {Singer, Uriel and Guy, Ido and Radinsky, Kira},
	  booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on
		       Artificial Intelligence, {IJCAI-19}},
	  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
	  pages     = {4605--4612},
	  year      = {2019},
	  month     = {7},
	  doi       = {10.24963/ijcai.2019/640},
	  url       = {https://doi.org/10.24963/ijcai.2019/640},
	}



### Miscellaneous

Please send any questions you might have about the code and/or the algorithm to <urielsinger@cs.technion.ac.il>.

Note: This is only a beta version of the tNodeEmbed algorithm. There are other amendments that need to be made before this work can be relied upon.
