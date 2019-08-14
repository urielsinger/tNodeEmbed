from keras.layers import Input, Dense, Activation, Concatenate, Lambda
from keras.models import Model

from models.task_model import TaskModel
from utils.consts import TLP, NC

class StaticModel(TaskModel):
    def __init__(self, task):
        '''
        Static model init
        Args:
            task: str - name of the task. either 'temporal_link_prediction' or 'node_classification'
        '''
        super(StaticModel, self).__init__(task=task)

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
            keras model of the static node2vec
        '''
        if task == TLP:
            inputs = Input(shape=input_shape)

            lmda_lyr1 = Lambda(lambda x: x[:, 0, -1, :], output_shape=input_shape[2:])(inputs)
            lmda_lyr2 = Lambda(lambda x: x[:, 1, -1, :], output_shape=input_shape[2:])(inputs)
            concat_lyr = Concatenate(axis=-1)([lmda_lyr1, lmda_lyr2])

            fc_lyr1 = Dense(latent_dim, activation='relu')(concat_lyr)
            fc_lyr2 = Dense(1)(fc_lyr1)
            soft_lyr = Activation('sigmoid')(fc_lyr2)

            model = Model(inputs, soft_lyr)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        elif task == NC:
            inputs = Input(shape=input_shape)
            lmbd_lyr = Lambda(lambda x: x[:, -1, :], output_shape=input_shape[1:])(inputs)
            fc_lyr = Dense(num_classes)(lmbd_lyr)
            soft_lyr = Activation('softmax')(fc_lyr)

            model = Model(inputs, soft_lyr)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            raise Exception('unknown task for _get_model')
        return model