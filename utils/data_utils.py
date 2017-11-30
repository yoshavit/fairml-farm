import pandas as pd
import urllib.request
from os import path

def fetch_adult_dataset(datadir=None):
    """ Download and preprocess the "Adult" dataset into a dataframe
    Args:
        datadir - if not None, try looking in this dir for the dataset before
            redownloading, and save data to this directory after downloading.
            The files in question would be named:
            datadir + "/adult_" + [train,validation] + ".txt"
    Returns:
        training_dataset: a dataframe containing the training data, with dummy
            variables as appropriate
        validation_dataset: same as train_dataset, but with validation samples
    """
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
    # split the data back into train and test
    training_dataset = full_df.iloc[:trainset_size, :]
    validation_dataset = full_df.iloc[trainset_size:, :]
    return training_dataset, validation_dataset

def adult_dataset(datadir=None,
                  normalize=False,
                  removable_columns=['sex', 'income>50k'],
                  protectedfn=lambda s: "Male" in s['sex'],
                  labelfn=lambda s: ">50K" in s['income>50k']
                 ):
    """Fetches and processes the Adult dataset into data, protected, and label
    groupings.
    Args:
        datadir - the directory to which the data had been previously
            downloaded, or will be downloaded.
        normalize - if True, training data are normalized to be 0 mean and unit
            variance, and test data have the same transformations (computed on
            training set) applied to them. Default False.
        removable_columns - columns from the original dataset to be removed
            during execution. Default ['sex', 'income>50k']
        protectedfn -  function, computed on a row in the adult dataset, for
            extracting a protected attribute from the dataset.
            Default: lambda s: "Male" in s['sex'],
        labelfn -  function, computed on a row in the adult dataset, for
            extracting a label from the dataset.
            Default: lambda s: ">50K" in s['income>50k']
            (WARNING: due to formatting of the original data, categories for
            'income>50k' are different for the train and validation sets)
    Returns:
        train_dataset - a dictionary of three numpy arrays, containing fields:
            "data" - the data to be trained on
            "protected" - the attribute to be protected
            "label" - the label to be predicted
        validation_dataset - same as train_dataset, but for the validation
            examples
    """
    train_df, val_df = fetch_adult_dataset(datadir)
    train_data = train_df.drop(columns=removable_columns).values
    if normalize:
        train_data_mean = train_data.mean(axis=0)
        train_data_stdev = train_data.std(axis=0)
        train_data = (train_data - train_data_mean)/train_data_stdev
    train_dataset = {"data": train_data,
                     "label": train_df.apply(labelfn, axis=1).as_matrix(),
                     "protected": train_df.apply(protectedfn,
                                                 axis=1).as_matrix()
                    }
    val_data = val_df.drop(columns=removable_columns).values
    if normalize:
        val_data = (val_data - train_data_mean)/train_data_stdev
    validation_dataset = {"data": val_data,
                          "label": val_df.apply(labelfn, axis=1).as_matrix(),
                          "protected": val_df.apply(protectedfn,
                                                    axis=1).as_matrix()
                         }
    return train_dataset, validation_dataset
