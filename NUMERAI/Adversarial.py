

#!/usr/bin/env python

"train a classifier to distinguish between train and test"
"save train examples in order of similarity to test (ascending)"

import numpy as np
import pandas as pd

from sklearn import cross_validation as CV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, PolynomialFeatures
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score as accuracy
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA

from time import ctime

#

train_file = '../input/numerai_training_data.csv'
test_file = '../input/numerai_tournament_data.csv'
output_file = '../input/train_sorted.csv'

print('https://github.com/zygmuntz/adversarial-validation/blob/master/numerai/sort_train.py')
print('http://fastml.com/adversarial-validation-part-two/')
print ("loading...")

train = pd.read_csv( train_file )
test = pd.read_csv( test_file )

train.drop('id',axis=1,inplace=True)#Added
test.drop( 'id', axis = 1, inplace = True )


test['target'] = 0		# dummy for preserving column order when concatenating
train['is_test'] = 0
test['is_test'] = 1

orig_train = train.copy()
assert( np.all( orig_train.columns == test.columns ))

train = pd.concat(( orig_train, test ))
train.reset_index( inplace = True, drop = True )

do_PCA=False
if do_PCA:
	print ('Doing PCA analysis')
	output_file = '../input/PCA_train_sorted.csv'
	x = train.drop( ['era','data_type','is_test', 'target','is_test' ], axis = 1 )
	pca=PCA(n_components=2)
	pca.fit(x)
	x=pca.transform(x)
	y = train.is_test
	x=pd.DataFrame(x)
	print ('Finished PCA analysis')
else:	
	x = train.drop( ['era','data_type','is_test', 'target' ], axis = 1 )
	y = train.is_test
#'id','era','data_type','target'
#

print ("cross-validating...")

n_estimators = 100
clf = RF( n_estimators = n_estimators, n_jobs = -1 )
#clf = AdaBoostClassifier(n_estimators=n_estimators,learning_rate=1.0)# Works terribly when compared to RF
#clf = AdaBoostClassifier(n_estimators=n_estimators,learning_rate=0.001)

predictions = np.zeros( y.shape )

cv = CV.StratifiedKFold( y, n_folds = 5, shuffle = True, random_state = 5678 )
tryCV=True
if tryCV:
	for f, ( train_i, test_i ) in enumerate( cv ):

		#print "# fold {}, {}".format( f + 1, ctime())

		x_train = x.iloc[train_i]
		x_test = x.iloc[test_i]
		y_train = y.iloc[train_i]
		y_test = y.iloc[test_i]
	
		clf.fit( x_train, y_train )	

		p = clf.predict_proba( x_test )[:,1]
	
		auc = AUC( y_test, p )
		#print "# AUC: {:.2%}\n".format( auc )	
	
		predictions[ test_i ] = p

		# fold 1
		# AUC: 87.00%


	train['p'] = predictions
	i = predictions.argsort()
	train_sorted = train.iloc[i]
else:
	alg = AdaBoostClassifier(n_estimators=nTrees,learning_rate=lr)#Fold accuracy actually decreased at1E-5
	alg.fit(trainX[features],trainY['target'])


"""
print "predictions distribution for test"
train_sorted.loc[ train_sorted.is_test == 1, 'p' ].hist()
p_test_mean = train_sorted.loc[ train_sorted.is_test == 1, 'p' ].mean()
p_test_std = train_sorted.loc[ train_sorted.is_test == 1, 'p' ].std()
print "# mean: {}, std: {}".format( p_test_mean, p_test_std )
# mean: 0.404749669062, std: 0.109116404564
"""

train_sorted = train_sorted.loc[ train_sorted.is_test == 0 ]
assert( train_sorted.target.sum() == orig_train.target.sum())

"""
print "predictions distribution for train"
p_train_mean = train_sorted.p.mean()
p_train_std = train_sorted.p.std()
print "# mean: {}, std: {}".format( p_train_mean, p_train_std )
# mean: 0.293768613822, std: 0.113601453932
"""

train_sorted.drop( 'is_test', axis = 1, inplace = True )
train_sorted.to_csv( output_file, index = False )
