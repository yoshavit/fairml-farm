import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition.pca import PCA

def plot_embeddings(embeddings, labels, protecteds,
                    plot3d=False,
                    subsample=False,
                    label_names=None,
                    protected_names=None):
    if protected_names is None:
        protected_names = ["A0", "A1"]
    if label_names is None:
        label_names = ["L0", "L1"]
    n = embeddings.shape[0]
    if not subsample:
        subsample = n
    inds = np.random.permutation(n)[:subsample]
    pca = PCA(n_components= 3 if plot3d else 2)
    labels = labels.astype(bool)[inds]
    protecteds = protecteds.astype(bool)[inds]
    pca.fit(embeddings)
    embs = pca.transform(embeddings)[inds, :]
    fig = plt.figure()
    if plot3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    for l in [False, True]:# labels
        for p in [False, True]: # protecteds
            idxs = np.logical_and(labels == l, protecteds == p)
            embs_slice = embs[idxs, :]
            data_vectors = [embs_slice[:, 0], embs_slice[:, 1]]
            if plot3d: data_vectors.append(embs_slice[:,2])
            color = "b" if p else "r"
            marker = "o" if l else "x"
            name = "{} {}".format(protected_names[p],
                                  label_names[l])
            ax.scatter(*data_vectors,
                        edgecolors=color,
                        marker=marker,
                        facecolors=[color, 'none'][l], # only leave circles unfilled
                        label=name)
    ax.legend(fontsize="small")
    plt.show()

