#!/usr/bin/env python
""" collections of utility functions """

import sys
import os
import csv
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve

class Instance(object):
    """
        Instance class represents set of raw data collected per data instance
    """
    def __init__(self, dir):
        self.audio = self._load_audio(dir)
        self.touch = self._load_touch_data(dir)
        self.info = self._load_instance_info(dir)

    def _load_audio(self, dir):
        """ load audio data
            param dir: a path to an instance directory
            return: audio data
        """
        rate, wav = scipy.io.wavfile.read(os.path.join(dir, "audio.wav"))
        return wav
        
    def _load_touch_data(self, dir):
        """ load touch data
            param dir: a path to an instance directory
            return : a dictionary contains touch data
        """
        with open(os.path.join(dir, "touch.csv"), "rU") as f:
            reader = csv.DictReader(f)
            for touch in reader:
                for key in  touch.keys():
                    touch[key] = float(touch[key])
                break
        return touch
    
    def _load_instance_info(self, dir):
        """ load instance info from a directory path
            param dir: a path to an instance directory
            return: a dictionary contains basic instance information
        """
        info = {}
        user_dirnames = os.path.basename(os.path.dirname(dir)).split("-")
        info["surface"] = user_dirnames[0]
        info["user"] = user_dirnames[1]
        instance_dirnames = os.path.basename(dir).split("-")
        info["timestamp"] = instance_dirnames[0]
        # set None to classlabel if it's test data
        info["classlabel"] = instance_dirnames[1] if len(instance_dirnames) == 2 else None
        return info


def load_instances(dir):
    """ function for loading raw data instances
        param dir: a path to a data directory (i.e. task_data/train or task_data/test)
        return: a list of data instance objects
    """
    instances = []
    for root, dirs, files in os.walk(os.path.join(dir)):
        for filename in files:
            if filename == "audio.wav":
                instances.append(Instance(root))
    return instances


def load_labels(instances):
    """ load class labels
        param instances: a list of data instance objects
        return: class labels mapped to a number (0=pad, 1=knuckle)
    """
    y = np.array([{"pad": 0, "knuckle": 1}[instance.info["classlabel"]] for instance in instances], dtype=int)
    return y


def load_timestamps(instances):
    """ load timestamps
        param instances: a list of data instance objects
    """
    timestamps = [instance.info["timestamp"] for instance in instances]
    return timestamps


def convert_to_classlabels(y):
    """ convert to classlabels
        param y: mapped class labels
        return: class labels
    """
    classlabels = [["pad", "knuckle"][y[i]] for i in range(len(y))]
    return classlabels


def write_results(timestamps, classlabels, output):
    """ write classification results to an output file
        param timestamps: a list of timestamps
        param classlabels: a list of predicted class labels
        return : None
    """
    if len(timestamps) != len(classlabels):
        raise Exception("The number of timestamps and classlabels doesn't match.")
    with open(output, "wb") as f:
        f.write("timestamp,label\n")
        for timestamp, classlabel in zip(timestamps, classlabels):
            f.write(timestamp + "," + classlabel + "\n")

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def main(argv):
    raise Exception("This script isn't meant to be run.")


if __name__ == '__main__':
    exit(main(sys.argv[1:]))
