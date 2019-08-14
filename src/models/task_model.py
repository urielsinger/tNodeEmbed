from abc import ABC, abstractmethod

class TaskModel(ABC):
    def __init__(self, task):
        '''
        Model init
        Args:
            task: str - name of the task. either 'temporal_link_prediction' or 'node_classification'
        '''
        self.task = task

    def fit(self, X, y, **keras_args):
        '''
        Model fitting
        Args:
            X: np.array - input samples
            X: np.array - output labels
            keras_args: arguments to be used in the keras fitting
        Returns:
        '''
        self.model = self._get_model(self.task, X.shape[1:], latent_dim=X.shape[-1], num_classes=y.shape[1])
        self.model.fit(X, y, **keras_args)

    def fit_generator(self, generator, steps_per_epoch, **keras_args):
        '''
        Model fitting using a generator
        Args:
            generator: iter - generating batches of samples
            steps_per_epoch: int - number of steps in an epoch
            keras_args: arguments to be used in the keras fitting
        Returns:
        '''
        X_sample, y_sample = next(generator)
        self.model = self._get_model(self.task, X_sample.shape[1:], latent_dim=X_sample.shape[-1], num_classes=y_sample.shape[1])

        self.model.fit_generator(generator, steps_per_epoch, **keras_args)

    def predict(self, X):
        '''
        Model prediction
        Args:
            X: np.array - input samples
        Returns:
            prediction of the static model
        '''
        return self.model.predict(X)

    def predict_generator(self, generator, steps):
        '''
        Model prediction using a generator
        Args:
            generator: iter - generating batches of samples
            steps: int - number of steps to call the generator
        Returns:
            prediction of the static model
        '''
        return self.model.predict_generator(generator, steps)

    @staticmethod
    @abstractmethod
    def _get_model(task, input_shape, latent_dim=128, num_classes=1):
        '''
        Given the task, return the desired architecture of training
        Args:
            task: string - of the tasks name, either 'temporal_link_prediction' or 'node_classification'
            input_shape: tuple - shape of a singe sample
            latent_dim: int - the size of the LSTM latent space
            num_classes: int - number of classes. Relevant only if task=='node_classification'
        Returns:
            keras model
        '''
        pass