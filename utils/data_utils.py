import os
import urllib.request
import pandas as pd

# ===== ADD NEW DATASET NAMES HERE =====
dataset_names = ["adult", "german"]
# ======================================

def get_dataset(dataset_name,
                base_datadir=None,
                normalize=False,
                removable_columns=None,
                protectedfn=None,
                labelfn=None
               ):
    """Fetches and processes the Adult dataset into data, protected, and label
    groupings.
    Args:
        dataset_name - name of the dataset to be fetched (must be one of
            data_utils.dataset_names)
        datadir - the directory to which the data had been previously
            downloaded, or will be downloaded.
        normalize - if True, training data are normalized to be 0 mean and unit
            variance, and test data have the same transformations (computed on
            training set) applied to them. Default False.
        removable_columns - columns from the original dataset to be removed
            during execution.
        protectedfn -  function, computed on a row in the adult dataset, for
            extracting a protected attribute from the dataset.
        labelfn -  function, computed on a row in the adult dataset, for
            extracting a label from the dataset.
    Returns:
        train_dataset - a dictionary of three numpy arrays, containing fields:
            "data" - the data to be trained on
            "protected" - the attribute to be protected
            "label" - the label to be predicted
        validation_dataset - same as train_dataset, but for the validation
            examples
    """

    assert dataset_name in dataset_names
    datadir = os.path.join(base_datadir, dataset_name)
    os.makedirs(datadir, exist_ok=True)
    if dataset_name == 'adult':
        train_df, val_df = fetch_adult_dataset(datadir)
        if not removable_columns: removable_columns = ['sex', 'income>50k']
        if not protectedfn: protectedfn=lambda s: s['sex']
        if not labelfn: labelfn=lambda s: s['income>50k']
    elif dataset_name == "german":
        train_df, val_df = fetch_german_dataset(datadir)
        if not removable_columns: removable_columns = ["risk_good"]
        if not protectedfn: protectedfn=lambda s: s['age']>25
        if not labelfn: labelfn=lambda s: s['risk_good']
    elif dataset_name == 'heloc':
        # FICO hasn't yet given enough info on their 'HELOC' credit score
        # dataset for this to be sufficiently supported yet.
        # To learn more, check https://community.fico.com/community/xml/pages/overview
        raise NotImplementedError
    else:
        raise NotImplementedError("{} dataset support not "
                                  "implemented".format(dataset_name))
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
        trainfile = os.path.join(datadir, 'adult_train.txt')
        valfile = os.path.join(datadir, 'adult_val.txt')
        if not(os.path.exists(trainfile) and os.path.exists(valfile)):
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
    categoricalnames = set(names) - set(continuousnames)
    # Loading and processing the data.
    train_df = pd.read_csv(trainfile, names=names, index_col=False, comment='|')
    val_df = pd.read_csv(valfile, names=names, index_col=False, comment='|')
    # remove EOL periods from val file
    val_df["income>50k"] = val_df["income>50k"].apply(lambda s: s[:-1])
    # 'fnlwgt' is a scalar relating to the demographic importance of the individual, so we discard it
    full_df = pd.concat([train_df, val_df]).drop(columns=["fnlwgt"])
    # construct dummy variables for the categorical vars
    full_df = pd.get_dummies(full_df, columns=categoricalnames)
    # remove redundant encodings, simplify var names
    full_df["income>50k"] = full_df["income>50k_ >50K"]
    full_df["sex"] = full_df["sex_ Male"]
    full_df = full_df.drop(
        columns=[c for c in full_df.columns if "sex_" in c or "50k_" in c])
    # split the data back into train and test
    trainset_size = train_df.shape[0]
    training_dataset = full_df.iloc[:trainset_size, :]
    validation_dataset = full_df.iloc[trainset_size:, :]
    return training_dataset, validation_dataset

def fetch_german_dataset(datadir=None):
    # The original "German" dataset from the UCI repository (see
    # https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data) )
    # is poorly-formatted and confusing to use. Instead, we use a preprocessed
    # version provided in this kaggle competition discussion:
    # https://www.kaggle.com/uciml/german-credit/discussion/26658
    url = "https://kaggle2.blob.core.windows.net/forum-message-attachments/237294/7771/german_credit_data.csv"
    if datadir is not None:
        fpath = os.path.join(datadir, "german_credit_dataset.csv")
        if not os.path.exists(fpath):
            fpath, _ = urllib.request.urlretrieve(url, fpath)
    else:
        fpath = urllib.request.urlopen(url)

    names = ["age", "sex", "job", "housing", "saving_accounts",
             "checking_account", "credit_amount", "duration", "purpose", "risk",
             ]
    continuous_names = ["age", "credit_amount", "duration"]
    categoricalnames = set(names) - set(continuous_names)

    df = pd.read_csv(fpath, names=names, skiprows=[0])
    df = pd.get_dummies(df, columns=categoricalnames)
    df = df.drop(columns=["risk_bad", "sex_female"]) # remove redundant encodings
    # split the data into train and test (in a consistent way)
    train_split = 0.85
    trainset_size = int(df.shape[0]*train_split)
    training_dataset = df.iloc[:trainset_size, :]
    validation_dataset = df.iloc[trainset_size:, :]
    return training_dataset, validation_dataset


