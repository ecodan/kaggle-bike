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

    for x in range(0,len(Xcat_feat)):
        print("col " + str(df[Xcat_feat].columns[x]) + ": " + str(df[Xcat_feat].iloc[:,x].nunique()))

    # convert categories to features
    if enc == None:
        enc = sl.preprocessing.OneHotEncoder(n_values=[   5,    5, 2013,   13,   24,   8], sparse=False)
        enc.fit(df[Xcat_feat])

    Xcat = enc.transform(df[Xcat_feat])

    # remove a bunch of useless year columns (0-2010)
    Xcat = np.delete(Xcat, np.s_[10:2020], 1)

    Xcat2 = df[Xcat2_feat]

    Xcont = df[Xcont_feat].values

    if scalar == None:
        scaler = sl.preprocessing.StandardScaler().fit(Xcont)

    #Xcont = scaler.transform(Xcont)

    # build the whole feature matrix
    X = np.concatenate( ( Xcont, Xcat2, Xcat ), axis=1 )

    # re-add column names for pretty outputting of intermediate outputs
    Xcat_ph = np.zeros(Xcat.shape[1])
    cols = np.concatenate((Xcont_feat, Xcat2_feat, Xcat_ph), axis=0)
    X = pd.DataFrame(X, columns=cols)

    new_headers = np.array(['Season 0','Season 1','Season 2','Season 3','Season 4','Weather 0','Weather 1','Weather 2','Weather 3','Weather 4','2011','2012','2013','Mo 1','Mo 2','Mo 3','Mo 4','Mo 5','Mo 6','Mo 7','Mo 8','Mo 9','Mo 10','Mo 11','Mo 12','Mo 13','Hr 1','Hr 2','Hr 3','Hr 4','Hr 5','Hr 6','Hr 7','Hr 8','Hr 9','Hr 10','Hr 11','Hr 12','Hr 13','Hr 14','Hr 15','Hr 16','Hr 17','Hr 18','Hr 19','Hr 20','Hr 21','Hr 22','Hr 23','Hr 24','DOW  1','DOW  2','DOW  3','DOW  4','DOW  5','DOW  6','DOW  7','DOW 8'])
    print(str(X.columns[0:5].values))
    print(str(new_headers))
    X.columns = np.concatenate((X.columns[0:6].values, new_headers), axis=0)
    print(str(X.columns))

    return X, enc, scalar


# runs tests
if __name__=='__main__':

   pred = np.ones(10)
   act = np.ones(10)
   print('t1 equal: ', score_func(act, pred))

   act = act * 1.1
   print('t2a +.1: ', score_func(act, pred))

   act = act * 2
   print('t2b +1: ', score_func(act, pred))

   act = act * 10
   print('t3 +19: ', score_func(act, pred))

   print('99/9 ', score_func(np.array([9]),np.array([99])))
   print('2x99/9 ', score_func(np.array([9,9]),np.array([99,99])))