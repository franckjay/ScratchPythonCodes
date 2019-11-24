import numpy as np
import pandas as pd
import h2o
from h2o.estimators import H2ORandomForestEstimator
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
####################
       
print ('Loading data')
tag=''	
train= h2o.import_file('../input/numerai_training_data.csv')
features=['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14', 'feature15', 'feature16', 'feature17', 'feature18', 'feature19', 'feature20', 'feature21']
target='target'
nTrees=50	
print('Starting H2O training: nTrees=',nTrees)
# Define model
model = H2ORandomForestEstimator(ntrees=nTrees, max_depth=20, nfolds=10)
# Train model
model.train(x=features, y=target, training_frame=train)
print('Finished H20 training')

print('Testing H2O')

test=pd.read_csv('../input/numerai_tournament_data.csv')
valid=test[test['data_type']=='validation']

print ('Do your scoring here')
validX=valid.drop(['id','era','data_type','target'],axis=1)
validY=valid['target']
valid_pred=[]

performance = model.model_performance(test_data=test)
print ("Model Performance: ",performance)

#score=log_loss(validY, valid_pred, eps=1e-15, normalize=True)
#roc=roc_auc_score(validY, valid_pred)
#print ('Done Scoring! LogLoss of ',score,'should submit: ', score<-np.log(0.5))
#print ("ROC AUC score: ",roc)
	
testX=test.drop(['id','era','data_type','target'],axis=1)

testY_XGB=[]
testY_XGB+= list(gbm.predict(xgb.DMatrix(testX)))
print('Testing done.')				
 
print ('Writing Submission')
filename='h2o_Out'+tag+'_'+str(nTrees)
submission = pd.DataFrame({"id": test["id"], "probability": testY_XGB})
submission=submission[['id','probability']]#For whatever reason, you have to flip these back.
print (submission.head())
submission.to_csv('../output/'+filename+'_'+str(score)[0:7]+'.csv', index=False)	
print ('Finished submission')


                   
