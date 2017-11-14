import tensorflow as tf
import pandas as pd
import urllib.request
from os import path

def get_adult_dataset(data_format="numpy", datadir=None, normalize=False):
    """ Download and preprocess the "Adult" dataset
    Args:
        data_format - the format in which the data is returned, with the labels
            always as follows: ["data", "label", "protected"]
            "numpy" - returns a dict of numpy arrays with the above keys
            "tfdataset" - returns a Tensorflow Dataset object, where each item
                is a dict with the above keys
        data_dir - if not None, try looking in this dir for the dataset before
            redownloading, and save data to this directory after downloading.
            The files in question would be named:
            datadir + "/adult_" + [train,validation] + ".txt"
        normalize - if True, training data are normalized to be 0 mean and unit
            variance, and test data have the same transformations (computed on
            training set) applied to them.

    Returns:
        train_dataset: an object containing the training data (of format
            specified by tfdataset)
        validation_dataset: same as train_dataset, but with validation samples
        data_names: a list of the names of the data columns

    """
    assert data_format in ["numpy", "tfdataset"]
    # Download the 'Adult' dataset from the UCI dataset archive
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.'
    traindata_url = url + 'data'
    valdata_url = url + 'test'
    if datadir is not None:
        trainfile = path.join(datadir, 'adult_train.txt')
        valfile = path.join(datadir, 'adult_val.txt')
        if not(path.exists(trainfile) and path.exists(valfile)):
            trainfile, _ = urllib.request.urlretrieve(traindata_url, trainfile)
            valfile, _ = urllib.request.urlretrieve(valdata_url, valfile)
    else:
        trainfile = urllib.request.urlopen(traindata_url)
        valfile = urllib.request.urlopen(valdata_url)

    names = ["age","workclass","fnlwgt","education","education-num",
             "marital-status","occupation","relationship","race","sex",
             "capital-gain","capital-loss","hours-per-week",
             "native-country","income>50k"]
    continuousnames = ["age", "fnlwgt", "education-num", "capital-gain",
                       "capital-loss", "hours-per-week"]
    targetname = "income>50k"
    protectedname = "sex"
    categoricalnames = set(names) - set([targetname, protectedname] + continuousnames)
    # Loading and processing the data.
    train_df = pd.read_csv(trainfile, names=names, index_col=False, comment='|')
    val_df = pd.read_csv(valfile, names=names, index_col=False, comment='|')
    trainset_size = train_df.shape[0]
    # 'fnlwgt' is a scalar relating to the demographic importance of the individual, so we discard it
    full_df = pd.concat([train_df, val_df]).drop(columns=["fnlwgt"])
    # construct dummy variables for the categorical vars
    full_df = pd.get_dummies(full_df, columns=categoricalnames)
    datanames = full_df.drop(columns=[targetname, protectedname]).columns
    # split the data back into train and test
    train_df, val_df = full_df.iloc[:trainset_size, :], full_df.iloc[trainset_size:, :]
    if normalize:
        train_data = train_df.drop(columns=[targetname, protectedname]).values
        train_data_mean = train_data.mean(axis=0)
        train_data_stdev = train_data.std(axis=0)
    datasets = []
    for df in [train_df, val_df]:
        data = df.drop(columns=[targetname, protectedname]).values
        if normalize:
            data = (data - train_data_mean)/train_data_stdev
        dataset = {"data": data,
                   "label": df[targetname].factorize()[0], # discretize and grab labels
                   "protected": df[protectedname].factorize()[0]}
        if data_format == "tfdataset":
            dataset = tf.data.Dataset.from_tensor_slices(dataset)
        datasets += [dataset]
    train_dataset, validation_dataset = datasets
    return train_dataset, validation_dataset, datanames

if __name__ == '__main__':
    get_adult_dataset()
