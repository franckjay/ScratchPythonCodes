import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout
####################


print('Repurposed from: https://gist.github.com/vinhkhuc/e53a70f9e5c3f55852b0')
RANDOM_SEED=11

def main():
    nEpochs=10 #
    h_size = 22 # Number of hidden nodes
    batch_size=100 #Process how many examples at once?
    nLayers=20
    active='sigmoid'
    optim='adam'
    tag=str(nLayers)+'layered_'+str(h_size)+"sized_"


    
    print('Single Layer with N=',h_size,' nodes, a batch size of ',batch_size)

    USE_VALID=False
    if USE_VALID:
        tag='plusValid'
        train_X=pd.read_csv('../input/train_valid.csv')
    else:
        tag=''
        train_X=pd.read_csv('../input/numerai_training_data.csv')
    train_X=train_X.drop(['id','era','data_type'],axis=1)
    target=train_X['target']
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]
    train_X=train_X.drop(['target'],axis=1)
    test=pd.read_csv('../input/numerai_tournament_data.csv')
    valid=test[test['data_type']=='validation']
    target=valid['target'].astype('int')
    num_labels = len(np.unique(target))
    valid_Y = np.eye(num_labels)[target]

    valid=valid.drop(['id','era','data_type','target'],axis=1)
    test=test.drop(['id','era','data_type','target'],axis=1)				
    test=test.as_matrix() #We want this in matrix form, not Pandas dataframes

				
    N,M=train_X.shape
    all_X = np.ones((N, M+ 1))#Shape of the data andA one Bias layer
    all_X[:, 1:] = train_X #Paste the test data on top	
			
     #We want this in matrix form, not Pandas dataframes
    N, M  = test.shape
    test_X = np.ones((N, M + 1))#Shape of the data and one Bias layer
    test_X[:, 1:] = test #Paste the test data on top
    test_X=test_X.astype('float32') #Make sure it is in the right data-type
    test,valid='',''#Save Space

    model = Sequential()
    for i in range(nLayers):
        model.add(Dense(h_size, input_dim=M+1, init='uniform', activation=active))
        model.add(Dropout(0.8, noise_shape=None, seed=RANDOM_SEED))
    model.add(Dense(2, init='uniform', activation='sigmoid'))   
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])

    print('Training')
    model.fit(all_X, all_Y, nb_epoch=nEpochs, batch_size=batch_size,  verbose=2)
    print('Done Training.')    
    train_X, train_Y, all_X= '','','' #Save space for your father's sake  
    print ('Testing')
    predictions = model.predict(test_X)
    predictions=predictions[:,1]
    print (predictions[0:5])
    print('Done testing!')
    

    print ('Writing Submission')
    print(predictions.shape)
    test=pd.read_csv('../input/numerai_tournament_data.csv')
    filename='NN_Keras'+tag+'_'+str(h_size)+'_'+str(nEpochs)
    submission = pd.DataFrame({"id": test["id"], "probability": predictions})
    print(submission.describe())
    submission=submission[['id','probability']]#For whatever reason, you have to flip these back.
    print (submission.head() )
    submission.to_csv('../output/'+filename+'_10LAYER.csv', index=False)	
    print ('Finished submission')
    return 0							    		
 


####################
main()

