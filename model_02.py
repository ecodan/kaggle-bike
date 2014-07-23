__author__ = 'dan'


import os
import math
import pandas as pd
import numpy as np
import sklearn as sl
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn.metrics import roc_auc_score
import bike_common as bc


def main(in_dir, out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    in_file = in_dir + '/train.csv'

    print "reading datafile: " + str(in_file)
    df = pd.read_table(in_file, sep=',', header=0, parse_dates=['datetime'])
    df['dow'] = df['datetime'].apply(lambda x: x.weekday())

    X, enc, scalar = bc.prep_data(df)

    y = df[['count']].values

    X_train, X_test, y_train, y_test = sl.cross_validation.train_test_split(X, y)

    # clf = linear_model.LinearRegression()
    # clf = sl.tree.DecisionTreeRegressor()
    # clf = ensemble.GradientBoostingClassifier()
    # clf = ensemble.GradientBoostingRegressor()
    clf = ensemble.RandomForestRegressor()
    param_grid = {'n_estimators': [5,10,20,50], 'max_features': (None,0.75,0.50,0.25)}

    rmsle_scorer = sl.metrics.make_scorer(bc.score_func, greater_is_better=False)

    # train the rental model
    print ('training')
    srch = sl.grid_search.GridSearchCV(clf, param_grid, rmsle_scorer)
    srch.fit(X_train, y_train.ravel())
    clf = srch.best_estimator_
    print('clf stats: best_score=%f best_params=%s' % (srch.best_score_, srch.best_params_) )

    zc = clf.predict(X_test)
    zc[zc<0] = 0
    print ('clf Xtrain RMSLE: ' + str(bc.score_func(y_test, zc)))

    # full prediction + addition
    z = clf.predict(X)
    print ('clf Xt RMSLE: ' + str(bc.score_func(y, z)))


    ##############
    # run model against test dataset

    test_file = in_dir + '/test.csv'
    print "reading test datafile: " + str(test_file)
    df_test = pd.read_table(test_file, sep=',',header=0, parse_dates=['datetime'])
    df_test['dow'] = df_test['datetime'].apply(lambda x: x.weekday())

    Xtest, enc, scalar = bc.prep_data(df_test, enc, scalar)

    # full prediction + addition
    zc = clf.predict(Xtest)
    zc[zc<0] = 0

    df_test['count'] = zc

    df_test[['datetime','count']].to_csv(out_dir + '/submission.csv', sep=',', header=True, index=False)



if __name__=='__main__':

    args = {
        'in_dir':  '/Users/dan/dev/datasci/kaggle/bikeshare',
        'out_dir':  '/Users/dan/dev/datasci/kaggle/bikeshare/out'
    }
    model = main(**args)
