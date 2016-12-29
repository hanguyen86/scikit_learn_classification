"""
Author : Hoang Anh Nguyen
Email  : n.h.a.1986@gmail.com
Licence: BSD
"""

import sys
import os
import numpy as np
from time import time
from utility import load_instances, load_labels, load_timestamps, convert_to_classlabels, write_results, plot_learning_curve
from scipy.stats import randint as sp_randint
from sklearn.svm import NuSVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

#------------------------------------------------------------
#--------------------- Classfier Classes --------------------
#------------------------------------------------------------
class Classifier:
    
    def __init__(self, X, y):
        self.y = y
        self.selectFeature(X)        
    
    # Public methods
    def train(self, performTuning=False, showLearningCurve=False):
        """ train the model w.r.t input X and ground-truth label y
            return: optimal model
        """
        print('======= Training model ========')
        
        # define cross validation method: Shuffle & Split
        if performTuning == False:
            self.model.fit(self.X, self.y)
            
            # show learning curve
            if showLearningCurve == True:
                title = 'Learning Curves'
                cv = ShuffleSplit(self.X.shape[0],
                                  n_iter=1,
                                  test_size=0.2,
                                  random_state=0)
                plot_learning_curve(self.model,
                                    title,
                                    self.X,
                                    self.y,
                                    cv=cv)
                plt.show()
        else:
            if not hasattr(self, 'param_rand'):
                print('This method does not support hyper-parameter tuning!')
                self.model.fit(self.X, self.y)
                return
            
            print('Start hyper-parameter tuning:')
            
            # we use random search grid for exploring hyperparameters space +
            # shuffle data input at each iteration
            n_iter_search = 20
            cv = ShuffleSplit(self.X.shape[0],
                              n_iter=n_iter_search,
                              test_size=0.2,
                              random_state=0)
            random_search = RandomizedSearchCV(n_jobs=-1,
                                               estimator=self.model,
                                               cv=cv,
                                               param_distributions=self.param_rand,
                                               n_iter=n_iter_search)
            
            start = time()
            random_search.fit(self.X, self.y)
            print("RandomizedSearchCV took %.2f seconds for %d candidates"
                  " parameter settings." % ((time() - start), n_iter_search))
            self._report(random_search.cv_results_)

            # use the best estimator
            self.model = random_search.best_estimator_
            
        print("Training accuracy: %.4f" % self.model.score(self.X,
                                                           self.y))
        
    def predict(self, X):
        """ predict label on the given input
            return: predicted label of each input
        """
        return self.model.predict(X)
    
    ## Helpers
    def selectFeature(self, X):
        """ Compute feature importances by using ExtraTreesRegressor
            return: less dimensional features
        """
        clf = ExtraTreesRegressor(n_jobs=-1)
        clf.fit(X, self.y)
        self.features = SelectFromModel(estimator=clf,
                                        threshold='0.24*mean',
                                        prefit=True)
        self.X = self.features.transform(X)
        print('New feature size: %.d' % self.X.shape[1])
    
    def _report(self, results, n_top=3):
        """ Display top 3 best models from random search for hyperparameters
            return: None
        """
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.4f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

##------------------------------------------
# Class 1: Nu SVM
# Similar to SVM but uses a parameter to control the number of support vectors
class SVM(Classifier):
    
    def __init__(self, X, y):
        Classifier.__init__(self, X, y)
        
        # All hyperparameters were fine-tuned from random search
        self.model = NuSVC(nu=0.4,
                           kernel='rbf',
                           degree=2,
                           verbose=True)
        
        # For Nu SVM, we focus on tuning following 3 hyperparameters:
        #   1. nu: an upper bound on the fraction of training errors and
        #   a lower bound of the fraction of support vectors.
        #   2. kernel: specifies the kernel type to be used in the algorithm
        #   3. degree: degree of the polynomial kernel function ('poly')
        self.param_rand = {
            "nu": [0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95],
            "kernel": ['poly', 'rbf', 'sigmoid'],
            "degree": sp_randint(2, 10)
        }

##------------------------------------------
# Class 2: Multi-layer Perceptron classifier
# This model optimizes the log-loss function using LBFGS or stochastic 
# gradient descent.
class NeuralNetwork(Classifier):
    
    def __init__(self, X, y):
        Classifier.__init__(self, X, y)
        
        # All hyperparameters were fine-tuned from random search
        self.model = MLPClassifier(solver='adam',
                                   learning_rate='adaptive',
                                   alpha=0.005,
                                   hidden_layer_sizes=(116),
                                   random_state=1,
                                   learning_rate_init=1e-4,
                                   verbose=True)
        
        # For Neural Network, we focus on tuning following 3 hyperparameters:
        #    1. alpha: L2 penalty (regularization term)
        #    2. learning_rate_init: initial learning rate used
        #    3. hidden_layer_sizes: size of hidden layer
        self.param_rand = {
            "alpha": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 1e-2, 1e-1],
            "learning_rate_init": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, \
                                   1e-1, 1, 10],
            "hidden_layer_sizes": sp_randint(4, 200)
        }
    
##------------------------------------------
# Class 3: Extremely Randomized Trees
# This class implements a meta estimator that fits a number of randomized 
# decision trees (a.k.a. extra-trees) on various sub-samples of the dataset 
# and use averaging to improve the predictive accuracy and control over-fitting.
class ExtraTrees(Classifier):
    
    def __init__(self, X, y):
        Classifier.__init__(self, X, y)
        
        # All hyperparameters were fine-tuned from random search
        self.model = ExtraTreesClassifier(n_jobs=-1,
                                          max_features=18,
                                          min_samples_split=3,
                                          min_samples_leaf=1,
                                          n_estimators=127,
                                          verbose=True)
        
        # For Extra Trees, we focus on tuning following 4
        # hyperparameters:
        #   1. max_features: the number of features to consider when looking
        #   for the best split.
        #   2. min_samples_split: the minimum number of samples required to 
        #   split an internal node.
        #   3. min_samples_leaf: the minimum number of samples required to be
        #   at a leaf node.
        #   4. n_estimators: the number of trees in the forest.
        self.param_rand = {
            "max_features": sp_randint(3, 40),
            "min_samples_split": sp_randint(2, 20),
            "min_samples_leaf": sp_randint(1, 20),
            "n_estimators": sp_randint(50, 150)
        }        

##------------------------------------------
# Class 4: Gaussian Naive Bayes (GaussianNB)
# GaussianNB implements the Gaussian Naive Bayes algorithm for classification
class GaussianNaiveBayes(Classifier):
    
    def __init__(self, X, y):
        Classifier.__init__(self, X, y)
        
        self.model = GaussianNB()

#------------------------------------------------------------
#---------------------- Main function -----------------------
#------------------------------------------------------------
def generate_features(instances):
    """ generate features
        param instances: a list of Instance class objects
        return: a feature matrix
    """
    # feature set: [wave_data x y major minor pressure orientation]
    X = np.array([np.append(instance.audio.astype(float), \
                            [instance.touch['x'], \
                             instance.touch['y'], \
                             instance.touch['major'], \
                             instance.touch['minor'], \
                             instance.touch['pressure'], \
                             instance.touch['orientation']]) \
                  for instance in instances])
    return X

def calcul_precision(y_predicted, y_label):
    """ compare predicted label with ground_truth
        return: precision from 0->1
    """
    return np.average(y_predicted==y_label)

def main(argv):
    
    # preprocessing: perform normalization and centralization on input data  
    max_abs_scaler = StandardScaler()

    print('======= Preparing training dataset ========')
    train_instances = load_instances("data/train")
    X_data = max_abs_scaler.fit_transform(generate_features(train_instances))
    y_data = load_labels(train_instances)
    
    # training & hyperparameter tuning will take 80% data, testing takes 20%
    X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                        y_data,
                                                        test_size=0.2,
                                                        random_state=0)
    
    print('Training samples : %.d' % X_train.shape[0])
    print('Feature size     : %.d' % X_train.shape[1])
    print('Testing samples  : %.d' % X_test.shape[0])
    
    # Pick one of the 4 following classifiers:
#    classifier = GaussianNaiveBayes(X_data, y_data)
#    classifier = SVM(X_train, y_train)
#    classifier = NeuralNetwork(X_train, y_train)
    classifier = ExtraTrees(X_train, y_train)

    # train with fine-tuned hyperparameters with/without
    # performing hyperparameter tuning (much longer to finish).
    classifier.train(performTuning=False,
                     showLearningCurve=False)
    
    print('======= Validation on 20% training dataset ========')
    print('Testing score: %.4f' % \
          classifier.model.score(classifier.features.transform(X_test),
                                 y_test))

    print('======= Classifying real test dataset ========')
    # Now prepare to run on real test dataset
    test_instances = load_instances("data/test")
    X_real = max_abs_scaler.transform(generate_features(test_instances))
    timestamps = load_timestamps(test_instances)
    
    # predict
    y_pred = np.asarray(classifier.predict(classifier.features.transform(X_real)))
    classlabels = np.asarray(convert_to_classlabels(y_pred))
    
    # save results
    write_results(timestamps, classlabels, "./fingersense-test-labels.csv")
    print('Result saved to file fingersense-test-labels.csv')

if __name__ == '__main__':
    exit(main(sys.argv[1:]))
