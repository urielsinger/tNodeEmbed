import platform
from os.path import join
from datetime import datetime
from utils.consts import TLP, NC

experiment_name = ''
time_now = datetime.now().strftime('%y%m%d_%H%M%S')

params = {
'results_path': join(r"../results", f"{platform.node()}_{time_now}_{experiment_name}"),
'dataset': 'PPI',
'task': TLP,#TLP, NC
'test_size': 0.2,
'train_skip': 100,  # down sample the training set
'n2vargs': {'workers': 4},
'keras_args':
    {'batch_size': 64,
     'nb_epoch': 10,
     'nb_worker': 4}
}
