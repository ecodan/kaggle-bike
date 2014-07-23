__author__ = 'dan'

import math
import numpy as np
import pandas as pd
import sklearn as sl


# implements the contest root mean squared log error algorithm for use standalone or in sklearn components
def score_func(y, y_pred, **kwargs):
    y = y.ravel()
    y_pred = y_pred.ravel()
    res = math.sqrt( np.sum( np.square(np.log(y_pred+1) - np.log(y+1)) ) / len(y) )
    return res


# features that will be used as-is
Xcont_feat = np.array(['temp','atemp','humidity','windspeed'])

# features to be broken out into categorical features
Xcat_feat = np.array(['season','weather','yr','mo','hr', 'dow'])

# features already set up as categorical features
Xcat2_feat = np.array(['holiday','workingday'])


# extract date elements, encode categories to features and then scale
def prep_data(df, enc=None, scalar=None):

    # extract date/time
    df['yr'] = df['datetime'].apply(lambda x: x.year)
    df['mo'] = df['datetime'].apply(lambda x: x.month)
    df['day'] = df['datetime'].apply(lambda x: x.day)
    df['hr'] = df['datetime'].apply(lambda x: x.hour)

    # convert categories to features
    if enc == None:
        enc = sl.preprocessing.OneHotEncoder(n_values=[   5,    5, 2013,   13,   24,   8])
        enc.fit(df[Xcat_feat])

    Xcat = enc.transform(df[Xcat_feat])

    # remove a bunch of useless year columns (0-2010)
    Xcat = np.delete(Xcat.toarray(), np.s_[10:2020], 1)

    Xcat2 = df[Xcat2_feat]

    Xcont = df[Xcont_feat].values

    if scalar == None:
        scaler = sl.preprocessing.StandardScaler().fit(Xcont)

    Xcont = scaler.transform(Xcont)

    # build the whole feature matrix
    X = np.concatenate( ( Xcont, Xcat2, Xcat ), axis=1 )

    # re-add column names for pretty outputting of intermediate outputs
    Xcat_ph = np.zeros(Xcat.shape[1])
    cols = np.concatenate((Xcont_feat, Xcat2_feat, Xcat_ph), axis=0)
    X = pd.DataFrame(X, columns=cols)

    return X, enc, scalar


# runs tests
if __name__=='__main__':

   pred = np.ones(10)
   act = np.ones(10)
   print('t1 equal: ', score_func(act, pred))

   act = act * 2
   print('t2 +1: ', score_func(act, pred))

   act = act * 10
   print('t3 +19: ', score_func(act, pred))
