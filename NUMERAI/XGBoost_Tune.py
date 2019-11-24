import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib.pylab as plt
####################

def modelfit(alg, dtrain, predictors,dtrainY,target,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrainY[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        print (cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrainY[target],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrainY[target].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrainY[target], dtrain_predprob))

print ('Taken from: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/')
print ('Loading data')
tag=''	
df=pd.read_csv('../input/numerai_training_data.csv')
trainX=df.drop(['id','era','data_type','target'],axis=1)
trainY=df[['target']]
df=''
features=trainX.columns
target='target'
	
print('Starting XGBOOST training tuning!')
 
#Choose all predictors except target & IDcols
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 scale_pos_weight=1,
 random_state=13)
#modelfit(xgb1, trainX, features,trainY,target)

print ("Parameter Test 4")
param_test4 = { 'learning_rate':[0.0001,0.001,0.01,0.1,0.3,1.0] }
gsearch4 = GridSearchCV(estimator = XGBClassifier(n_estimators=78, max_depth=4,
 min_child_weight=6, gamma=0.2, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', scale_pos_weight=1,random_state=27), 
 param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
#gsearch4.fit(trainX[features],trainY[target])
#print ("Best Params: ",gsearch4.best_params_)#Best Params:  0.1
#print ("Best Score: ",gsearch4.best_score_)#Best Score: 0.512

depths=[2,4,6,8]
childs=[2,4,6]
print ("Parameter Test 1")
param_test1 = { 'max_depth':depths, 'min_child_weight':childs}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=78, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', scale_pos_weight=1, random_state=27), 
 param_grid = param_test1, scoring='roc_auc',iid=False, cv=5)
gsearch1.fit(trainX[features],trainY[target] )
print (gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
print ("Best Params: ",gsearch1.best_params_)#{'min_child_weight': 6, 'max_depth': 4}
print ("Best Score: ",gsearch1.best_score_)# 0.5129046047357047

print ("Parameter Test 2")
param_test2b = { 'min_child_weight':[6,8,10,12] }
gsearch2b = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=78, max_depth=4,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', scale_pos_weight=1,random_state=27), 
 param_grid = param_test2b, scoring='roc_auc',iid=False, cv=5)
#gsearch2b.fit(trainX[features] , trainY[target] )
#print (gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)
#print ("Best Params: ",gsearch2b.best_params_)#Best Params:  {'min_child_weight': 6}
#print ("Best Score: ",gsearch2b.best_score_)#Best Score:  0.5129046047357047


print ("Parameter Test 3")
param_test3 = {  'gamma':[i/10.0 for i in range(0,5)] }
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=78, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', scale_pos_weight=1,random_state=27), 
 param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
#gsearch3.fit(trainX[features],trainY[target])
#print (gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)
#print ("Best Params: ",gsearch3.best_params_)#Best Params:  {'gamma': 0.2}
#print ("Best Score: ",gsearch3.best_score_)#Best Score:  0.5129715704006925




print ("Parameter Test 5")
param_test5 = { 'subsample':[0.5,0.6,0.7,0.8,0.9,1.0],'colsample_bytree':[0.5,0.6,0.7,0.8,0.9,1.0] }
gsearch5 = GridSearchCV(estimator = XGBClassifier(n_estimators=78, max_depth=4,
 min_child_weight=6, gamma=0.2, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', scale_pos_weight=1,random_state=27), 
 param_grid = param_test5, scoring='roc_auc',iid=False, cv=5)
#gsearch5.fit(trainX[features],trainY[target])
#print ("Best Params: ",gsearch5.best_params_)#Best Params:  0.1
#print ("Best Score: ",gsearch5.best_score_)#Best Score: 0.512
xgb2 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=78,
 max_depth=4,
 min_child_weight=6,
 gamma=0.2,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 random_state=27)
#modelfit(xgb2,trainX, features,trainY,target)
