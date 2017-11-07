import tensorflow as tf
import pandas as pd
import urllib

def get_adult_dataset():
    """ Download and preprocess the "Adult" dataset

    Returns:
        train_dataset: Tensorflow Dataset, with the fields
            ["data", "label", "protected"]
        validation_dataset: same as train_dataset, but with validation samples
        data_names: a list of the names of the data columns

    """
    # Download the 'Adult' dataset from the UCI dataset archive
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.'
    traindata_url = url + 'data'
    valdata_url = url + 'test'
    trainfile = urllib.request.urlopen(traindata_url)
    valfile = urllib.request.urlopen(valdata_url)

    names = ["age","workclass","fnlwgt","education","education-num",
             "marital-status","occupation","relationship","race","sex",
             "capital-gain","capital-loss", "hours-per-week",
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
    datasets = []
    for df in [train_df, val_df]:
        dataset = tf.data.Dataset.from_tensor_slices(
            {"data": df.drop(columns=[targetname, protectedname]).values,
             "label": df[targetname].factorize()[0], # discretize and grab labels
             "protected": df[protectedname].factorize()[0]})
        datasets += [dataset]
    train_dataset, validation_dataset = datasets
    return train_dataset, validation_dataset, datanames

if __name__ == '__main__':
    get_adult_dataset()
