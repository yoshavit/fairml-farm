from abc import ABC, abstractmethod, abstractproperty

class BaseClassifier(ABC):
    """Abstract class for binary classifiers.
    """
    @abstractproperty
    def name(self):
        pass
    @abstractmethod
    def build(self, hparams={}):
        """ Builds model components to the point where prediction/training can
        be done.
        Useful for routines requiring inherited/overwritten methods.
        """
        raise NotImplementedError
    @abstractmethod
    def load_model(self, filedir):
        """ Loads the model parameters from files in filedir
        """
        raise NotImplementedError
    @abstractmethod
    def save_model(self, logdir):
        """ Saves the model parameters in a file in logdir
        Returns the filename for the stored parameters
        """
        raise NotImplementedError
    @abstractmethod
    def predict(X):
        """
        Arguments:
            X: an n x d matrix of datapoints
        Returns:
            Yhat: an n x 1 vector of predicted classifications, each from 0 to 1
        """
        raise NotImplementedError
    @abstractmethod
    def compute_embedding(X):
        """
        Arguments:
            X: an n x d matrix of datapoints
        Returns:
            Z: an n x r lower-dimensional embedding of each of the datapoints
        """
        return X
