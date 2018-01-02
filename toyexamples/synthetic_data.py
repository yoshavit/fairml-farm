import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

"""
This file contains an interface for defining and drawing from simple, synthetic
datasets that are useful as toy examples when testing fairness algorithms.
"""

class SquareBlock:
    def __init__(self, group, origin, probpositive=1, sizex=1, sizey=1):
        assert group in [0,1]
        self.group = group
        self.origin = origin
        self.sizex = sizex
        self.sizey = sizey
        self.probpositive = probpositive # the likelihood a draw from this
        # block returns a positive (1) outcome

        # TODO: allow for gradated probabilities within a block

    def draw_n(self, n):
        """
        Return n draws from the given block.
        Blocks are of the format (x, y, class_id, outcome_id)
        """
        draws = {"data": np.stack([self.origin[0] + np.random.random(n)*self.sizex,
                                   self.origin[1] + np.random.random(n)*self.sizey]).T,
                 "protected": np.repeat(self.group, n),
                 "label": np.random.binomial(1, self.probpositive, size=n)}
        return draws

class ToyWorld:
    def __init__(self):
        self.totalprob = 0
        self.blocks = []
        self.blockprobs = []
        self.axs = []

    def add_block(self, block, blockprob):
        self.totalprob += blockprob
        if self.totalprob > 1:
            import warnings
            warnings.warn("Total probability greater than 1"
                          "; normalizing values to be equal to 1")
        self.blocks.append(block)
        self.blockprobs.append(blockprob)

    def draw_n(self, n):
        draws_per_block = np.random.multinomial(
            n, pvals=np.array(self.blockprobs)/self.totalprob)
        draw_lists = [b.draw_n(n) for b, n in zip(self.blocks, draws_per_block)]
        draws = {k:np.concatenate([draw_list[k] for draw_list in draw_lists],
                                  axis=0)
                 for k in draw_lists[0].keys()}
        return draws

    def dataset(self, n=1000):
        train_dataset, validation_dataset = self.draw_n(n), self.draw_n(n)
        return train_dataset, validation_dataset

    def plot_points(self, n=100, fig=None):
        draws = self.draw_n(n)
        xs = draws["data"][:, 0]
        ys = draws["data"][:, 1]
        groups = draws["protected"]
        outcomes = draws["label"]
        if fig is None: fig = plt.figure()
        self.axs = fig.subplots(1, 3, sharex=True, sharey=True)
        for group in [0, 1]:
            x = xs[groups == group]
            y = ys[groups == group]
            o = outcomes[groups == group]
            color = ['blue', '#5e2f0e'][group]
            for ax in [self.axs[group], self.axs[2]]:
                ax.scatter(x, y, s=9, c=color)
                ax.scatter(x, y, s=1, c=np.where(o, 'Chartreuse', 'red'))
        fig.set_size_inches(9, 3, forward=True)
        # fig.tight_layout()

    def plot_line(self, a, b, c, label=None, **kwargs):
        """Coordinates of a line written as ax + by + c = 0
        """
        assert self.axs is not [], "Must plot points first to generate axes"
        for ax in self.axs:
            if b != 0:
                xs = ax.get_xlim()
                ys = [-a/b*x - c/b for x in xs]
            else:
                ys = ax.get_ylim()
                xs = [-c/a]*2
            line = Line2D(xs, ys, label=label, **kwargs)
            ax.add_line(line)

    def plot_contour(self, decision_function, lines=3, delta=0.025, color=None):
        default_ax = self.axs[0] # assumes all axes have same coord system
        x = np.arange(*default_ax.get_xlim(), delta)
        y = np.arange(*default_ax.get_ylim(), delta)
        X, Y = np.meshgrid(x, y)
        Z = decision_function(X, Y)
        for ax in self.axs:
            CS = ax.contour(X, Y, Z, lines, colors=color)
            ax.clabel(CS, fontsize=10, inline=1)


if __name__ == '__main__':
    b1 = SquareBlock(0, [0,0], probpositive=0.8)
    b2 = SquareBlock(1, [1,1], probpositive=0.5)
    w = ToyWorld()
    w.add_block(b1, 7)
    w.add_block(b2, 3)
    w.plot_points(n=500)
    w.plot_line(2, 0, -1, label="$\gamma = 1$")
    plt.show()


