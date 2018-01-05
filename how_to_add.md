# How to Add Content
## How to Add a Fairness Algorithm
First, create a new file under `algos/` in which to write a class
that trains and implements your algorithm. 

Your algorithm should inherit from the abstract class `algos.base.BaseClassifier`,
 and implement all of its methods.

If the algorithm does not compute an embedding, add a dummy method like
```python
class A:
    def compute_embeddings(self, X):
        return X 
```
If you'd like, consider logging training data using
 (tensorboard)[https://github.com/tensorflow/tensorboard].

Finally, in `algos/__init__.py`, import your class and add it to `classifier_types`.

 That's it!
## How to Add a Dataset
Dataset fetching and cleaning is handled in `utils/data_utils.py`. To add a new 
dataset, you will need to change this file in 3 places:
* Add the dataset 'name' to `dataset_names` (no spaces or special chars)
* Add a method `fetch_{dataset_name}_dataset()` to load the data into a pandas dataframe,
converting all columns into either numeric variables, or one-hot encodings of categorical
variables.
* In the `get_dataset()` method, add an `elif` clause to get the dataset, and define
the default "target" and "protected" attribute extractors (must both be boolean)
