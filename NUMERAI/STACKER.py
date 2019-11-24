import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from xgboost.sklearn import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense,Dropout
import matplotlib.pyplot as plt


####################
def Keras(A,B,C,A_y,h_size,active,drop1,drop2,optim,nEpochs):
    RANDOM_SEED,batch_size=11,1000

    N,M=A.shape
    num_labels = len(np.unique(A_y))
    all_y= np.eye(num_labels)[A_y]
    print (np.shape(all_y),all_y[0:2,:])
    A,B,C=A.as_matrix(),B.as_matrix(),C.as_matrix()
    A,B,C=A.astype('float32'),B.astype('float32'),C.astype('float32')

    
    model = Sequential()
    model.add(Dense(h_size, input_dim=M, init='uniform', activation=active))
    model.add(Dropout(drop1, noise_shape=None, seed=RANDOM_SEED))
    model.add(Dense(h_size, init='uniform', activation=active))
    model.add(Dropout(drop2, noise_shape=None, seed=RANDOM_SEED))
    model.add(Dense(2, init='uniform', activation='sigmoid'))   
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])

    print('Training')
    model.fit(A, all_y, epochs=nEpochs, batch_size=batch_size,  verbose=2)
    print('Done Training')
    pred_B = model.predict(B)
    pred_C = model.predict(C)
    #print ("Keras1",pred_C)
    print ("Mean Keras: ",np.mean(pred_C)," Median: ",np.median(pred_C))
    print ("Min Value: ",np.min(pred_C)," Max: ",np.max(pred_C)," STD: ", np.std(pred_C))
    return pred_B,pred_C

def Keras2(A,B,C,A_y,h_size,active,drop1,drop2,optim,nEpochs):
    RANDOM_SEED,batch_size=11,1000

    N,M=A.shape
    num_labels = len(np.unique(A_y))
    all_y= np.eye(num_labels)[A_y]
    print (np.shape(all_y),all_y[0:2,:])
    A,B,C=A.as_matrix(),B.as_matrix(),C.as_matrix()
    A,B,C=A.astype('float32'),B.astype('float32'),C.astype('float32')

    
    model = Sequential()
    model.add(Dense(h_size, input_dim=M, init='uniform', activation=active))
    model.add(Dropout(drop1, noise_shape=None, seed=RANDOM_SEED))
    model.add(Dense(h_size, init='uniform', activation=active))
    model.add(Dropout(drop2, noise_shape=None, seed=RANDOM_SEED))
    model.add(Dense(h_size, init='uniform', activation=active))
    model.add(Dense(2, init='uniform', activation='sigmoid'))   
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])

    print('Training')
    model.fit(A, all_y, epochs=nEpochs, batch_size=batch_size,  verbose=2)
    print('Done Training')
    pred_B = model.predict(B)
    pred_C = model.predict(C)
    #print ("Keras2",pred_C)
    return pred_B,pred_C

def Keras3(A,B,C,A_y,h_size,active,drop1,drop2,optim,nEpochs):
    RANDOM_SEED,batch_size=11,1000

    N,M=A.shape
    num_labels = len(np.unique(A_y))
    all_y= np.eye(num_labels)[A_y]
    print (np.shape(all_y),all_y[0:2,:])
    A,B,C=A.as_matrix(),B.as_matrix(),C.as_matrix()
    A,B,C=A.astype('float32'),B.astype('float32'),C.astype('float32')

    
    model = Sequential()
    model.add(Dense(h_size, input_dim=M, init='uniform', activation=active))
    model.add(Dropout(drop1, noise_shape=None, seed=RANDOM_SEED))
    model.add(Dense(h_size, init='uniform', activation=active))
    model.add(Dense(h_size, init='uniform', activation=active))
    model.add(Dense(h_size, init='uniform', activation=active))    
    model.add(Dense(2, init='uniform', activation='sigmoid'))   
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])

    print('Training')
    model.fit(A, all_y, epochs=nEpochs, batch_size=batch_size,  verbose=2)
    print('Done Training')
    pred_B = model.predict(B)
    pred_C = model.predict(C)
    #print ("Keras3",pred_C)
    return pred_B,pred_C

def nKeras(A,B,C,A_y,h_size,active,drop1,drop2,optim,nEpochs,nLayers):
    RANDOM_SEED,batch_size=11,1000

    N,M=A.shape
    num_labels = len(np.unique(A_y))
    all_y= np.eye(num_labels)[A_y]
    print (np.shape(all_y),all_y[0:2,:])
    A,B,C=A.as_matrix(),B.as_matrix(),C.as_matrix()
    A,B,C=A.astype('float32'),B.astype('float32'),C.astype('float32')

    model = Sequential()
    model.add(Dense(h_size, input_dim=M, init='uniform', activation=active))
    model.add(Dropout(drop1, noise_shape=None, seed=RANDOM_SEED))
    for i in range(nLayers-1):
        model.add(Dense(h_size, input_dim=M, init='uniform', activation=active))
    model.add(Dense(2, init='uniform', activation='sigmoid'))   
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])

    print('Training')
    model.fit(A, all_y, epochs=nEpochs, batch_size=batch_size,  verbose=2)
    print('Done Training')
    pred_B = model.predict(B)
    pred_C = model.predict(C)

    return pred_B,pred_C
           
def LR(A,B,C,A_y):
    alg=LogisticRegression()
    alg.fit(A,A_y)
    pred_B=alg.predict_proba(B)
    pred_C=alg.predict_proba(C)
    return pred_B,pred_C

def RF(nTrees,A,B,C,A_y):
    alg=RandomForestClassifier(n_estimators = nTrees)
    alg.fit(A,A_y)
    pred_B=alg.predict_proba(B)
    pred_C=alg.predict_proba(C)
    return pred_B,pred_C

def ADA(nTrees,lr,A,B,C,A_y):
    from sklearn.ensemble import AdaBoostClassifier
    alg=AdaBoostClassifier(n_estimators = nTrees,learning_rate=lr)
    alg.fit(A,A_y)
    pred_B=alg.predict_proba(B)
    pred_C=alg.predict_proba(C)
    return pred_B,pred_C

def XGB_norm(pars,A,B,C,A_y):
    alg=xgb.train(pars,xgb.DMatrix(A, A_y))
    pred_B=alg.predict(xgb.DMatrix(B) )
    pred_C=alg.predict(xgb.DMatrix(C) )
    return pred_B,pred_C
    
def XGB_Class(alg, dtrain, predictors,dtrainY,target,B,C,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors], label=dtrainY[target])
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
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrainY[target], dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrainY[target], dtrain_predprob))

    pred_B=alg.predict_proba(B[predictors])[:,1]
    pred_C=alg.predict_proba(C[predictors])[:,1]
    return pred_B,pred_C

def BUILD(size='SMALL',USE_VALID=True):

    A_B_Split=0.5
    print ('Loading data')


    if USE_VALID:
        tag='_plusValid'
        df=pd.read_csv('../input/train_valid.csv')
    else:
        tag=''
        df=pd.read_csv('../input/numerai_training_data.csv')

    if size=='SMALL':
        xgb1_nTrees,xgb2_nTrees=5,10
        k0,k1,k2,k3,k10,k11,k12,k13=5,10,50,100,200,100,100,250
        e0,e1,e2,e3,e10,e11,e12,e13=20,50,200,20,10,10,10,5
    if size=='MED':
        xgb1_nTrees,xgb2_nTrees=10,15
        k0,k1,k2,k3,k10,k11,k12,k13=15,25,75,250,500,250,250,500
        e0,e1,e2,e3,e10,e11,e12,e13=50,100,500,50,25,10,10,5
    if size=='LARGE':
        xgb1_nTrees,xgb2_nTrees=20,100
        k0,k1,k2,k3,k10,k11,k12,k13=50,100,150,500,1000,500,500,1000
        e0,e1,e2,e3,e10,e11,e12,e13=100,200,1000,100,50,20,20,10
    if size=="XLARGE":
        xgb1_nTrees,xgb2_nTrees=100,200
        #No change from original LARGE
        k0,k1,k2,k3,k10,k11,k12,k13=50,100,150,500,1000,500,500,1000
        e0,e1,e2,e3,e10,e11,e12,e13=100,200,1000,100,50,20,20,10
    trainX=df.drop(['id','era','data_type','target'],axis=1)
    trainY=df['target']
    df=''
    features=trainX.columns

    df=pd.read_csv('../input/numerai_tournament_data.csv')
    C=df.drop(['id','era','data_type','target'],axis=1)
    A, B, A_y, B_y = train_test_split(trainX, trainY, test_size=A_B_Split, random_state=42)
    print ('Data Split into A/B frames')

    #Need to do this for the plusValid set for some reason.
    A_y=A_y.astype("int")
    B_y=B_y.astype("int")
    #print (A_y.describe())
    
    params1={'learning_rate' :0.1,
         'n_estimators':xgb1_nTrees,
         'max_depth':4,
         'min_child_weight':4,
         'gamma':0.2,
         'objective':'binary:logistic',
         'scale_pos_weight':1,
         'random_state':27}
    params2={'learning_rate' :0.3,
         'n_estimators':xgb2_nTrees,
         'max_depth':10,
         'min_child_weight':1,
         'gamma':0.0,
         'objective':'binary:logistic',
         'scale_pos_weight':1,
         'random_state':27}
    pred_B0,pred_C0=Keras(A,B,C,A_y,k0,'relu',0.5,0.8,'adam',e0)
    pred_B1,pred_C1=Keras(A,B,C,A_y,k1,'sigmoid',0.5,0.8,'adam',e1)
    pred_B2,pred_C2=Keras(A,B,C,A_y,k2,'relu',0.5,0.8,'sgd',e2)
    pred_B3,pred_C3=Keras(A,B,C,A_y,k3,'relu',0.5,0.8,'sgd',e3)
    pred_B10,pred_C10=Keras2(A,B,C,A_y,k10,'relu',0.5,0.8,'sgd',e10)
    if size!="XLARGE":
        pred_B11,pred_C11=Keras3(A,B,C,A_y,k11,'relu',0.5,0.8,'sgd',e11)
        pred_B12,pred_C12=Keras3(A,B,C,A_y,k12,'relu',0.5,0.8,'adam',e12)
        pred_B13,pred_C13=Keras3(A,B,C,A_y,k13,'relu',0.5,0.8,'sgd',e13)
    else:
        pred_B11,pred_C11=nKeras(A,B,C,A_y,k11,'relu',0.5,0.8,'sgd',e11,4)
        pred_B12,pred_C12=nKeras(A,B,C,A_y,k12,'relu',0.5,0.8,'adam',e12,5)
        pred_B13,pred_C13=nKeras(A,B,C,A_y,k13,'relu',0.5,0.8,'sgd',e13,10)
        
    print ("")
    print ("Starting Other Models")
    pred_B4,pred_C4=LR(A,B,C,A_y)
    print ("1/6 finished")
    pred_B5,pred_C5=RF(100,A,B,C,A_y)
    print ("2/6 finished")
    pred_B6,pred_C6=RF(200,A,B,C,A_y)
    print ("3/6 finished")    
    pred_B7,pred_C7=ADA(200,0.1,A,B,C,A_y)
    print ("4/6 finished")    
    print ("")
    print ("Starting XGBoost2")
    pred_B8,pred_C8=XGB_norm(params1,A,B,C,A_y)
    print ("5/6 finished")
    pred_B9,pred_C9=XGB_norm(params2,A,B,C,A_y)

    print ("")
    print ("Combining data sets")
    combined_B=pd.DataFrame({"pred_C0":pred_B0[:,1],"pred_C1": pred_B1[:,1],"pred_C2": pred_B2[:,1],
                        "pred_C3":pred_B3[:,1], "pred_C4":pred_B4[:,1],"pred_C5":pred_B5[:,1],
                        "pred_C6":pred_B6[:,1],"pred_C7":pred_B7[:,1],"pred_C8":pred_B8,
                        "pred_C9":pred_B9,"pred_C10":pred_B10[:,1],"pred_C11":pred_B11[:,1],
                        "pred_C12":pred_B12[:,1],"pred_C13":pred_B13[:,1],"target":B_y})

    combined_C=pd.DataFrame({"pred_C0":pred_C0[:,1],"pred_C1": pred_C1[:,1],"pred_C2": pred_C2[:,1],
                        "pred_C3":pred_C3[:,1], "pred_C4":pred_C4[:,1],"pred_C5":pred_C5[:,1],
                        "pred_C6":pred_C6[:,1],"pred_C7":pred_C7[:,1],"pred_C8":pred_C8,
                        "pred_C9":pred_C9,"pred_C10":pred_C10[:,1],"pred_C11":pred_C11[:,1],
                        "pred_C12":pred_C12[:,1],"pred_C13":pred_C13[:,1]})

    combined_B.to_csv("../input/STACK_B_"+size+tag+".csv")
    combined_C.to_csv("../input/STACK_C_"+size+tag+".csv")

#########################
size='XLARGE'
USE_VALID=True
    
BUILD(size=size,USE_VALID=USE_VALID)


if USE_VALID:
    tag='_plusValid'
else:
    tag=''
B=pd.read_csv("../input/STACK_B_"+size+tag+".csv")
C=pd.read_csv("../input/STACK_C_"+size+tag+".csv")

predictors=["pred_C0","pred_C1","pred_C2","pred_C3","pred_C4","pred_C5","pred_C6",
            "pred_C7","pred_C8","pred_C9","pred_C10","pred_C11","pred_C12",'pred_C13']
params1={'learning_rate' :0.1,
         'n_estimators':5,
         'max_depth':6,
         'min_child_weight':4,
         'gamma':0.0,
         'objective':'binary:logistic',
         'scale_pos_weight':1,
         'random_state':27}
alg=xgb.train(params1,xgb.DMatrix(B[predictors], B['target']))
pred_C=alg.predict(xgb.DMatrix(C[predictors]) )

fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(alg, max_num_features=12, height=0.8, ax=ax)
plt.show()

print ('Writing XGB Submission')
test=pd.read_csv('../input/numerai_tournament_data.csv')
filename='STACK_Out_'+size
submission = pd.DataFrame({"id": test["id"], "probability": pred_C})
submission=submission[['id','probability']]#For whatever reason, you have to flip these back.
print (submission.head())
submission.to_csv('../output/'+filename+tag+'.csv', index=False)	
print ('Finished submission')

alg=LogisticRegression()
alg.fit(B[predictors], B['target'])
pred_C=alg.predict_proba(C[predictors])

print ('Writing Submission')
test=pd.read_csv('../input/numerai_tournament_data.csv')
filename='STACK_LR_'+size
submission = pd.DataFrame({"id": test["id"], "probability": pred_C[:,1]})
submission=submission[['id','probability']]#For whatever reason, you have to flip these back.
print (submission.head())
submission.to_csv('../output/'+filename+tag+'.csv', index=False)	
print ('Finished submission')

from sklearn import metrics
print("")
print("Classification Metric Tests")
y_act=[1.0,1.0,1.0,1.0,1.0,1.0,0.0]
y_pred=[1.0,1.0,0.0,1.0,1.0,1.0,0.0]

print ("Accuracy Score",metrics.accuracy_score(y_act,y_pred) )#	Accuracy classification score.
print ("AUC Score",metrics.auc(y_act,y_pred) )#	Compute Area Under the Curve (AUC) using the trapezoidal rule
print ("Average Precision Score",metrics.average_precision_score(y_act,y_pred) )
print ("LogLoss",metrics.log_loss(y_act, y_pred) )#
print ("ROC AUC Score",metrics.roc_auc_score(y_act,y_pred) )

print("")
print("Regression Metric Tests")
y_true=[1.0,1.0,1.0,1.0,1.0,1.0,1.0]
y_pred=[1.0,1.0,1.25,1.0,1.0,1.0,0.75]
print ("Mean Absolute Score",metrics.mean_absolute_error(y_true, y_pred))#	Mean absolute error regression loss
print ("Median Squared Score",metrics.mean_squared_error(y_true, y_pred) )#	Mean squared error regression loss
print ("Median Absolute Score",metrics.median_absolute_error(y_true, y_pred) )#	Median absolute error regression loss
print ("R^2 Score",metrics.r2_score(y_true, y_pred) )#

#alg=RandomForestClassifier(n_estimators=100)
#alg.fit(B[predictors], B['target'])
#pred_C=alg.predict_proba(C[predictors])

#print ('Writing Submission')
#test=pd.read_csv('../input/numerai_tournament_data.csv')
#filename='STACK_RF_'+size
#submission = pd.DataFrame({"id": test["id"], "probability": pred_C[:,1]})
#submission=submission[['id','probability']]#For whatever reason, you have to flip these back.
#print (submission.head())
#submission.to_csv('../output/'+filename+'.csv', index=False)	
#print ('Finished submission')
