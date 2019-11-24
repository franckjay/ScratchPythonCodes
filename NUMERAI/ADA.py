import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
####################
def ADA():	
	prob_limit=0.001#Probability that Training=Testing Data
	nTrees,lr=100,0.1
	Adversarial=True#Use Adversarial Input?
	COMBINED=False
	dropEra=True

	print ('Loading data')
	if Adversarial:
		tag='_Adversarial_'+str(prob_limit)
		if COMBINED:
			df=pd.read_csv('../input/train_sorted_COMB.csv')#Slightly lower consistency.
			tag+='_COMB'
		else:
			df=pd.read_csv('../input/train_sorted.csv')
		trainX=df[df['p']>prob_limit]#Keep everything that has a good chance of being similar to the test set. 25% is about the sweet spot for 
		print ('Using only this fraction of the data: ',float(len(trainX))/float(len(df)))
		#this data set. Any more and things get shaky, and values less are bad as well.
		trainX=trainX.drop(['p'],axis=1)
		trainY=trainX[['target']]
		if dropEra:
			trainX=trainX.drop(['era','data_type','target'],axis=1)
		else:
			trainX=trainX[['data_type','target']]		
	else:
		tag=''	
		df=pd.read_csv('../input/numerai_training_data.csv')
		if dropEra:
			trainX=df.drop(['id','era','data_type','target'],axis=1)
		else:
			trainX=df.drop(['id','data_type','target'],axis=1)
		trainY=df[['target']]
	df=''
	features=trainX.columns
	target=['target']
	
	print('Starting Adabooster training: nTrees=',nTrees,' learning rate=',lr,'Adversarial? ',Adversarial,' with only ',prob_limit,' of the data')
	alg = AdaBoostClassifier(n_estimators=nTrees,learning_rate=lr)#Fold accuracy actually decreased at1E-5
	alg.fit(trainX[features],trainY['target'])
	#scores = cross_validation.cross_val_score(alg, trainX[features],trainY['target'], cv=5,')
	#strat=StratifiedKFold(trainY['target'],nfolds=3)
	#print scores.mean()
	print('Finished Adabooster training')

	print('Testing Adaboost')

	test=pd.read_csv('../input/numerai_tournament_data.csv')
	valid=test[test['data_type']=='validation']
	#test=test[test['data_type']=='test']#Contains no TARGETS
	#live=test[test['data_type']=='live']#Contains no TARGETS
	print ('Do your scoring here')
	if dropEra:
		validX=valid.drop(['id','era','data_type','target'],axis=1)
	else:
		validX=valid.drop(['id','data_type','target'],axis=1)
	validY=valid['target']
	valid_pred=[]
	valid_pred+=list(alg.predict_proba(validX[features]).astype(float)[:,1])
	score=log_loss(validY, valid_pred, eps=1e-15, normalize=True)
	roc=roc_auc_score(validY, valid_pred)
	print ('Done Scoring! LogLoss of ',score,'should submit: ', score<-np.log(0.5))
	print ("ROC AUC score: ",roc)

	if dropEra:
		testX=test.drop(['id','era','data_type','target'],axis=1)
	else:
		testX=test.drop(['id','data_type','target'],axis=1)
	testY_ADA=[]
	testY_ADA+=list(alg.predict_proba(testX[features]).astype(float)[:,1])
	print('Testing done.')				
 
	print ('Writing Submission')
	filename='ADA_Out'+tag+'_'+str(nTrees)+'_'+str(lr)
	submission = pd.DataFrame({"id": test["id"], "probability": testY_ADA})
	submission=submission[['id','probability']]#For whatever reason, you have to flip these back.
	print (submission.head())
	submission.to_csv('../output/'+filename+'_'+str(score)[0:7]+'.csv', index=False)	
	print ('Finished submission')
	return 0
################


####################
ADA()

df1=pd.read_csv('../output/XGB_OutPlusValid_12_0.69181.csv')
df2=pd.read_csv('../output/STACK_Out_SMALL.csv')
df3=pd.read_csv('../output/ADA_Out_Adversarial_0.05_100_0.1_0.69298.csv')
#df2=pd.read_csv('../output/STACK_LR_MED.csv')

#df1=pd.read_csv('../output/XGB_Out_30_0.69247_Bootes.csv')
#df2=pd.read_csv('../output/NN_Out_PCA__15_0.2_Cygnus.csv')
#df3=pd.read_csv('../output/NN_Keras_PCA__250_100.csv')

probs=df1[['probability']]
hist_ADA,bin_edges=np.histogram(probs,bins=np.arange(0.40,0.60,0.01))
print (hist_ADA)    

probs=df2[['probability']]
hist_NN,bin_edges=np.histogram(probs,bins=np.arange(0.40,0.60,0.01))
print (hist_NN)    

b1,b2,b3=0.2,0.4,0.4
#df1['probability']=(df1['probability']+df2['probability'])/2.#Even weighting
#df1['probability']=(df1['probability']*0.75)+(df2['probability']*0.25)
#df1['probability']=(df1['probability']*0.25)+(df2['probability']*0.75)
#df1['probability']=(df1['probability']*0.6)+(df2['probability']*0.4)
df1['probability']=(df1['probability']*b1)+(df2['probability']*b2)+(df3['probability']*b3)
out=pd.DataFrame({"id": df1["id"], 'probability':df1["probability"] })
submission=out[['id','probability']]#For whatever reason, you have to flip these back.
print (submission.head())
submission.to_csv('../output/COMBINED_NN_ADA_Uneven.csv', index=False)	
print ('Finished submission')

                   
