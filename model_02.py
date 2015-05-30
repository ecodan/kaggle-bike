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
    # output prepped file for visual review
    # X.to_csv(out_dir + '/X.csv', sep=',', header=True, index=False)

    y = df[['count']].values

    Xcorr = pd.concat([X, df[['count']]], axis=1)
    Ycorr = Xcorr.corr()
    # output prepped file for visual review
    Ycorr.to_csv(out_dir + '/Ycorr.csv', sep=',', header=True, index=False)

    # strip X down to minimal columns based on pearson coef
    # cols = [0,1,2,3,7,9,12,14,17,18,20,21,32,33,34,35,36,37,38,40,48,49,50,51,55]
    # Xmini = X[cols]
    # print("shrunk X from " + str(X.columns) + " to " + str(Xmini.columns))
    # X = Xmini

    X_train, X_test, y_train, y_test = sl.cross_validation.train_test_split(X, y)

    # clf = linear_model.LinearRegression()
    # clf = sl.tree.DecisionTreeRegressor()
    # clf = ensemble.GradientBoostingClassifier()
    clf = ensemble.GradientBoostingRegressor(verbose=1)
    param_grid = {'loss':('ls','lad','huber','quantile'), 'n_estimators':[50,100,500], 'max_features':(None,'auto'), 'max_depth':(10,20,40)}
    # clf = ensemble.RandomForestRegressor()
    # param_grid = {'n_estimators': [5,10,20,50,100,500,1000], 'max_features': (None,0.75,0.50,0.25)}

    rmsle_scorer = sl.metrics.make_scorer(bc.score_func, greater_is_better=False)

    # train the rental model
    print ('training')
    srch = sl.grid_search.GridSearchCV(clf, param_grid, scoring=rmsle_scorer, verbose=1)
    srch.fit(X_train, y_train.ravel())
    clf = srch.best_estimator_
    print('clf stats: best_score=%f best_params=%s' % (srch.best_score_, srch.best_params_) )

    # clf = ensemble.RandomForestRegressor(n_estimators = 500, verbose = 1, n_jobs=10)
    # clf = linear_model.LinearRegression()
    # clf = ensemble.GradientBoostingRegressor(n_estimators=500, verbose=1, loss='huber', max_depth=24)

    clf.fit(X_train, y_train.ravel())

    zc = clf.predict(X_test)
    print("number rows < 0 = " + str(len(zc[zc<0])))
    zc[zc<0] = 0
    print ('clf CV RMSLE: ' + str(bc.score_func(y_test, zc)))

    # full prediction + addition
    # z = clf.predict(X)
    # print ('clf Xt RMSLE: ' + str(bc.score_func(y, z)))


    ##############
    # run model against test dataset

    test_file = in_dir + '/test.csv'
    print "reading test datafile: " + str(test_file)
    df_test = pd.read_table(test_file, sep=',',header=0, parse_dates=['datetime'])
    df_test['dow'] = df_test['datetime'].apply(lambda x: x.weekday())

    Xtest, enc, scalar = bc.prep_data(df_test, enc, scalar)

    # Xtest = Xtest[cols]

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
