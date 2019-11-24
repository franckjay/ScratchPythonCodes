

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

print ("Normally a good idea to SCALE the data prior to PCA, but in this case it does not seem to be an issue.")
print ("loading...")

train = pd.read_csv( train_file )
test = pd.read_csv( test_file )
nFeat=15
do_PCA=True
if do_PCA:
	print ('Doing PCA analysis')
	x = train.drop( ['era','data_type','id', 'target'], axis = 1 )
	test_x=test.drop(['era','data_type','id', 'target'],axis=1)
	pca=PCA(n_components=nFeat)
	pca.fit(x)
	x=pca.transform(x)
	test_x=pca.transform(test_x)
	x=pd.DataFrame(x)
	test_x=pd.DataFrame(test_x)
	x['target']=train['target']
	#test_x['target']=test['target']
	print ('Finished PCA analysis')
	
#print x.describe()
#print test_x.head()
	
output_file = '../input/PCA_train_'+str(nFeat)+'.csv'
output_file_test = '../input/PCA_test_'+str(nFeat)+'.csv'	
x.to_csv( output_file, index = False )
test_x.to_csv(output_file_test,index=False)


print (pca.explained_variance_)
print (np.sum(pca.explained_variance_))#n=2 : 0.15, while n=3: 0.183 of all variance... pretty dang low! If n=15, only gets up to 19%
print (pca.explained_variance_ratio_)

y=x['target']
x=x.drop('target',axis=1)
print(x[0].describe())
alg=AdaBoostClassifier(n_estimators=100)
alg.fit(x,y)
predictions=alg.predict(test_x)
print (predictions)
