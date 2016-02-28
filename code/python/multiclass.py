#!/usr/bin/env python
# -*- coding: utf-8 -*-


""" Multi-Class Classification """


import numpy as np
from sklearn import cross_validation, metrics
from matplotlib import pylab as plt
import itertools


__author__ = "Mikhail Karasikov"
__copyright__ = ""
__email__ = "karasikov@phystech.edu"


def cross_val_score(estimator, X, y, cv, fig_name=None, show_plot=True, output=True):
    """Performs sklearn cross_val_score and returns mean confusion matrix.
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like
        The data to fit. Can be, for example a list, or an array at least 2d.
    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.
    Returns
    -------
    confusion_matrix : 2d-array of float
    """

    scores = []
    labels = list(set(y))

    def scoring(estimator, X, y):
        confusion_matrix = metrics.confusion_matrix(y, estimator.predict(X), labels)
        scores.append(confusion_matrix)
        print(end=".", flush=True)
        return np.diag(confusion_matrix).sum() / confusion_matrix.sum()

    cross_validation.cross_val_score(estimator, X, y, scoring=scoring, cv=cv)

    confusion_mean = np.mean(scores, 0)

    if not output:
        return confusion_mean

    print("\nMean accuracy:", np.diag(confusion_mean).sum() / confusion_mean.sum())
    print("Confusion matrix:\n", (confusion_mean /
                                  confusion_mean.sum(1).reshape(-1, 1)).round(2))
    multiclass_recall_plot(confusion_mean, fig_name, labels, show_plot)
    return confusion_mean


def multiclass_recall_plot(confusion_matrix, fig_name=None, labels=None, show_plot=True):
    if labels is None:
        labels = np.arange(confusion_matrix.shape[0]) + 1
    width = 0.8

    num_objects = confusion_matrix.sum(1)
    sensitivity = np.diag(confusion_matrix) / num_objects

    x = np.arange(len(labels))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    rects = ax.bar(x, num_objects, width=width, color='#0000FF')
    rects2 = ax.bar(x, sensitivity * num_objects, width=width, color='#00FF00')

    plt.yticks(size=20)
    plt.xticks(x + width/2, labels, size=20)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        # labelbottom='off', # labels along the bottom edge are off
    )

    plt.xlabel('Class labels', size=20)
    plt.ylabel('Objects number', size=20)
    plt.title('Mean Accuracy: ${:.4f}$'.format(np.diag(confusion_matrix).sum() /
                                          num_objects.sum()),
              size=22)
    plt.ylim([0, max(num_objects) * 1.07])

    #for spine in ax.spines:
        #spine.set_linewidth(0.5)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(left='off')

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', serif='Times New Roman')
    plt.rc('pgf', texsystem='pdflatex')

    max_height = max([rect.get_height() for rect in rects])

    for bar in itertools.chain(rects, rects2):
        bar.set_linewidth(0.5)

    for i, rect in enumerate(rects):
        plt.text(rect.get_x() + rect.get_width()/2., rect.get_height(),
                 '${:.1f}\%$'.format(sensitivity[i] * 100),
                 ha='center', va='bottom', size=min(20, 25 * 6 / len(rects)))

    plt.tight_layout()
    if fig_name is not None:
        plt.savefig(fig_name)
    if not show_plot:
        plt.close(fig)
    plt.show()


def plot_grid_search_scores(grid_search_cv, vmin=None, vmax=None,
                            fig_name=None, show_plot=True):
    """Draw heatmap of the validation accuracy as a function of x and y

    Keyword arguments:
        grid_search ... sklearn.grid_search.GridSearchCV object, fitted estimator
    """

    try:
        y_name, x_name = grid_search_cv.grid_scores_[0][0].keys()
    except:
        raise Exception('Number of parameters must be 2')

    x_range = grid_search_cv.param_grid[x_name]
    y_range = grid_search_cv.param_grid[y_name]
    scores = [x[1] for x in grid_search_cv.grid_scores_]
    scores = np.array(scores).reshape(len(y_range), len(x_range))

    fig = plt.figure(figsize=(8, 6))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', serif='Times New Roman')
    plt.rc('pgf', texsystem='pdflatex')

    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    plt.xlabel(x_name, size=20)
    plt.ylabel(y_name, size=20)
    plt.colorbar()
    plt.xticks(np.arange(len(x_range)), x_range.round(5), rotation=45, size=10)
    plt.yticks(np.arange(len(y_range)), y_range.round(5), size=10)
    plt.title('Validation Accuracy', size=22)
    if fig_name is not None:
        plt.savefig(fig_name)
    if not show_plot:
        plt.close(fig)
    plt.show()
