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
    #names=['datetime','season','holiday','workingday','weather','temp','atemp','humidity','windspeed','casual','registered','count']
    df = pd.read_table(in_file, sep=',', header=0, parse_dates=['datetime'])
    df['dow'] = df['datetime'].apply(lambda x: x.weekday())

    # prepare the feature data
    X, enc, scalar = bc.prep_data(df)
    X.to_csv(out_dir + '/scratch.csv', sep=',', header=True, index=False)

    # extract the labels for the casual and reserved bike sets
    yc = df[['casual']].values
    yr = df[['registered']].values

    # create training and test data for casual and reserved data
    Xc_train, Xc_test, yc_train, yc_test = sl.cross_validation.train_test_split(X, yc)
    Xr_train, Xr_test, yr_train, yr_test = sl.cross_validation.train_test_split(X, yr)


    # try out a bunch of different models

    # RMLSE score=0.993142954278 (vs. test 15.57659)
    # clfc = linear_model.LinearRegression()
    # clfr = linear_model.LinearRegression()
    # param_grid = {}

    # RMLSE score=0.251551972408 (vs. test 0.59778)
    # with DOW RMLSE score=0.252605615098 (vs. .59090)
    # clfc = sl.tree.DecisionTreeRegressor()
    # clfr = sl.tree.DecisionTreeRegressor()
    # param_grid = {'max_features':(None,'auto','sqrt')}

    # RMLSE score=0.226339353755 (vs. test 0.57022
    # clfc = sl.tree.ExtraTreeRegressor()
    # clfr = sl.tree.ExtraTreeRegressor()
    # param_grid = {'max_features':(None,'auto','sqrt')}

    # RMLSE score=1.31458932334
    # clfc = sl.svm.SVR()
    # clfr = sl.svm.SVR()
    # param_grid = {'C':[.01,.1,1,10], 'epsilon':[.01,.1,1], 'kernel': ('rbf','linear','poly','sigmoid')}

    # RMLSE score=0.983135646219
    # clfc = ensemble.GradientBoostingRegressor()
    # clfr = ensemble.GradientBoostingRegressor()
    # param_grid = {'loss':('ls','lad','huber','quantile'), 'n_estimators':[50,100,200,500], 'max_features':(None,'auto','sqrt')}

    # RMLSE score=0.316815997324 (vs. test 0.54892)
    # with params RMLSE score=0.313294668232 (vs. test 0.57309)
    # with DOW RMLSE score=0.307161680679 (vs. test 0.55184)
    clfc = ensemble.RandomForestRegressor()
    clfr = ensemble.RandomForestRegressor()
    param_grid = {'n_estimators': [5,10,20,50], 'max_features': (None,0.75,0.50,0.25)}

    # turn my RMSLE method into a scoring func for use in GridSearch
    rmsle_scorer = sl.metrics.make_scorer(bc.score_func, greater_is_better=False)

    # train the casual rental model
    print ('training 1')
    srch = sl.grid_search.GridSearchCV(clfc, param_grid, rmsle_scorer)
    srch.fit(Xc_train, yc_train.ravel())
    clfc = srch.best_estimator_
    print('clfc stats: best_score=%f best_params=%s' % (srch.best_score_, srch.best_params_) )

    # re-fit with entire dataset
    clfc.fit(X, yc.ravel())

    # score this model for fun
    zc = clfc.predict(Xc_test)
    zc[zc<0] = 0
    print ('clf Xc rmsle: ' + str(bc.score_func(yc_test.ravel(), zc.ravel())))

    # train the reserved rental model
    print ('training 2')
    srch = sl.grid_search.GridSearchCV(clfr, param_grid, rmsle_scorer)
    srch.fit(Xr_train, yr_train.ravel())
    clfr = srch.best_estimator_
    print('clfr stats: best_score=%f best_params=%s' % (srch.best_score_, srch.best_params_) )

    # re-fit with entire dataset
    clfr.fit(X, yr.ravel())

    # score this model for fun
    zr = clfr.predict(Xr_test)
    zr[zr<0] = 0
    print ('clf Xr rmsle: ' + str(bc.score_func(yr_test, zr)))

    # now add the two predictions and score the combined model
    zc = clfc.predict(X)
    zc[zc<0] = 0
    zr = clfr.predict(X)
    zr[zr<0] = 0

    zt = zc + zr
    yt = df[['count']].values
    score = bc.score_func(yt, zt)
    print('RMLSE score=' + str(score))

    df['casual_pred'] = zc
    df['registered_pred'] = zr
    df['count_pred'] = zt
    df.to_csv(out_dir + '/train_xval_preds.csv', sep=',', header=True, index=False)


    ###########################
    # apply trained models to test set
    test_file = in_dir + '/test.csv'
    print "reading test datafile: " + str(test_file)
    df_test = pd.read_table(test_file, sep=',',header=0, parse_dates=['datetime'])
    df_test['dow'] = df_test['datetime'].apply(lambda x: x.weekday())

    Xtest, enc, scalar = bc.prep_data(df_test, enc, scalar)

    zc = clfc.predict(Xtest)
    zc[zc<0] = 0
    zr = clfr.predict(Xtest)
    zr[zr<0] = 0

    zt = zc + zr
    df_test['count'] = zt

    df_test[['datetime','count']].to_csv(out_dir + '/submission.csv', sep=',', header=True, index=False)



if __name__=='__main__':

    args = {
        'in_dir':  '/Users/dan/dev/datasci/kaggle/bikeshare',
        'out_dir':  '/Users/dan/dev/datasci/kaggle/bikeshare/out'
    }
    model = main(**args)
