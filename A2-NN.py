# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 22:13:32 2024

@author: User
"""
import numpy as np 
import pandas as pd
import mlrose_hiive
import time
import matplotlib.pyplot as plt
import dataframe_image
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, make_scorer, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbpipeline
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNC
import timeit
from functools import partial
import time

SEED = 1337
os.chdir(r"D:\Georgia Tech\CS7641 Machine Learning\A2")


pd.set_option('display.max_columns', None)

# Source: https://www.kaggle.com/datasets/l3llff/banana
df = pd.read_csv('banana_quality.csv')
df.describe()

df.loc[df['Quality'] == 'Good', 'Quality'] = 1
df.loc[df['Quality'] == 'Bad', 'Quality'] = 0
df['Quality'] = df['Quality'].astype('int')

x_var = ['Size', 'Weight', 'Sweetness', 'Softness', 'HarvestTime', 'Ripeness',
       'Acidity']

y_var = ['Quality']

rand = np.random.RandomState(seed=1111)
x_train, x_test, y_train, y_test = train_test_split(np.array(df.loc[:,x_var]),np.array(df.loc[:,y_var]),
                                                    test_size=0.80,random_state=SEED)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

rand = np.random.RandomState(seed=1337)
nnc = MLPClassifier(activation='logistic', random_state=rand)
strat_kfold = StratifiedKFold(n_splits=4,shuffle=True, random_state=rand)

hidden_layers_list = [1,5,10,15,20]
lr_list = [0.001, 0.01, 0.1]
nnc_params = {'learning_rate_init': lr_list, 'hidden_layer_sizes': hidden_layers_list, }

grid_search_nnc = GridSearchCV(estimator=nnc, param_grid=nnc_params, scoring='accuracy', cv=strat_kfold, n_jobs=-1, return_train_score=True)
grid_search_nnc.fit(x_train, y_train)

cv_score = grid_search_nnc.best_score_
test_score = grid_search_nnc.score(x_test, y_test)
print('Grid Search NNC')
print(grid_search_nnc.best_params_)
#{'classifier__hidden_layer_sizes': 15, 'classifier__learning_rate_init': 0.01}
print('CV Accuracy: %.5f' % (cv_score*100))
print('Test Accuracy: %.5f' % (test_score*100))
print('CV Error Rate: %.5f' % ((1-cv_score)*100))
print('Test Error Rate: %.5f' % ((1-test_score)*100))

ITER_LIST = [2**x for x in range(15)]
MAX_ATTEMPTS_LIST = [10, 100, 200, 300, 400, 500]

scorer = partial(accuracy_score)

sa_grid_search = {
    "learning_rate_init"        : [0.001, 0.01, 0.03, 0.05, 0.1],
    "hidden_layer_sizes"        : [[5,5],[10,10],[15,15],[20,20],[25,25]],
    "max_attempts"              : MAX_ATTEMPTS_LIST,
    "activation"                : [mlrose_hiive.neural.activation.sigmoid],
    "is_classifier"             : [True],
    "schedule"                  : [
        mlrose_hiive.GeomDecay(init_temp=1.0, decay=0.99, min_temp=0.001),
        mlrose_hiive.GeomDecay(init_temp=5.0, decay=0.99, min_temp=0.001),
        mlrose_hiive.GeomDecay(init_temp=10.0, decay=0.99, min_temp=0.001),
        mlrose_hiive.GeomDecay(init_temp=50.0, decay=0.99, min_temp=0.001),
        mlrose_hiive.GeomDecay(init_temp=100.0, decay=0.99, min_temp=0.001),]
}

sa_runner = mlrose_hiive.NNGSRunner(
    x_train = x_train_scaled, y_train = y_train, x_test=x_test_scaled, y_test=y_test,
    experiment_name             = "sa",
    output_directory            = "runner/",
    algorithm                   = mlrose_hiive.algorithms.sa.simulated_annealing,
    grid_search_parameters      = sa_grid_search,
    grid_search_scorer_method   = scorer,
    iteration_list              = ITER_LIST,
    bias                        = True,
    early_stopping              = True,
    clip_max                    = 1,
    generate_curves             = True,
    seed                        = SEED,
    n_jobs                      = -1,
    cv                          = 4
    )

print('Starting SA Runner...')
start_time = time.time()
sa_stats, sa_curves, sa_cv_results, sa_best_est = sa_runner.run()
end_time = time.time()
print(sa_best_est.best_params_)
print('SA Runner completed in %d (s).' % int(end_time-start_time))


rhc_grid_search = {
    "learning_rate_init"    : [0.001, 0.01, 0.03, 0.05, 0.1],
    "hidden_layer_sizes"    : [[5,5],[10,10],[15,15],[20,20],[25,25]],
    "max_attempts"          : MAX_ATTEMPTS_LIST,
    'restarts'              : [0], # restarts seems to be bugged, taking an extremely long time to run
    "activation"            : [mlrose_hiive.neural.activation.sigmoid],
    "is_classifier"         : [True],
}

rhc_runner = mlrose_hiive.NNGSRunner(
    x_train = x_train_scaled, y_train = y_train, x_test=x_test_scaled, y_test=y_test,
    experiment_name             = "rhc",
    output_directory            = "runner/",
    algorithm                   = mlrose_hiive.algorithms.random_hill_climb,
    grid_search_parameters      = rhc_grid_search,
    grid_search_scorer_method   = scorer,
    iteration_list              = ITER_LIST,
    bias                        = True,
    early_stopping              = True,
    clip_max                    = 1,
    generate_curves             = True,
    seed                        = SEED,
    n_jobs                      = -1,
    cv                          = 4
    )

print('Starting RHC Runner...')
start_time = time.time()
rhc_stats, rhc_curves, rhc_cv_results, rhc_best_est = rhc_runner.run()
end_time = time.time()
print(rhc_best_est.best_params_)
print('RHC Runner completed in %d (s).' % int(end_time-start_time))

'''
"learning_rate_init"    : [0.001, 0.01, 0.03, 0.05, 0.1],
"hidden_layer_sizes"    : [[5,5],[10,10],[15,15],[20,20],[25,25]],
"max_attempts"          : MAX_ATTEMPTS_LIST,
"activation"            : [mlrose_hiive.neural.activation.sigmoid],
"is_classifier"         : [True],
"mutation_prob"         : [0.1, 0.2, 0.3, 0.4, 0.5],
"pop_size"              : [100, 200, 300, 400, 500]
'''

ga_grid_search = {

    "learning_rate_init"    : [0.001, 0.01, 0.03],
    "hidden_layer_sizes"    : [[5,5]],
    "max_attempts"          : [10],
    "activation"            : [mlrose_hiive.neural.activation.sigmoid],
    "is_classifier"         : [True],
    "mutation_prob"         : [0.1, 0.2, 0.3],
    "pop_size"              : [100, 250, 500]   
}

ga_runner = mlrose_hiive.NNGSRunner(
    x_train = x_train_scaled, y_train = y_train, x_test=x_test_scaled, y_test=y_test,
    experiment_name             = "ga",
    output_directory            = "runner/",
    algorithm                   = mlrose_hiive.algorithms.genetic_alg,
    grid_search_parameters      = ga_grid_search,
    grid_search_scorer_method   = scorer,
    iteration_list              = ITER_LIST,
    bias                        = True,
    early_stopping              = True,
    clip_max                    = 1,
    generate_curves             = True,
    seed                        = SEED,
    n_jobs                      = -1,
    cv                          = 4
    )

print('Starting GA Runner...')
start_time = time.time()
ga_stats, ga_curves, ga_cv_results, ga_best_est = ga_runner.run()
end_time = time.time()
print(ga_best_est.best_params_)
print('GA Runner completed in %d (s).' % int(end_time-start_time))


# Fit
print('Fitting SA')
y_pred = sa_best_est.predict(x_test)
print( classification_report(y_test, y_pred))

print('Fitting RHC')
y_pred = rhc_best_est.predict(x_test)
print( classification_report(y_test, y_pred))

print('Fitting GA')
y_pred = ga_best_est.predict(x_test)
print( classification_report(y_test, y_pred))






