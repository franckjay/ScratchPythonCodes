import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout
####################


print('Repurposed from: https://gist.github.com/vinhkhuc/e53a70f9e5c3f55852b0')
RANDOM_SEED=11

def main():
    '''
    Did not outperform the TF NN, and had lower % of era fits. However, have only done 500 epochs.
    Adding dropout helped it a bit, but maybe increasing the nEpochs will help more.
    '''
    
    nEpochs=100 #
    h_size = 10 # Number of hidden nodes
    batch_size=1000 #Process how many examples at once?
    PCA=True
    print ('http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/')
    print ('Loading data')
    if PCA:
	    tag='_PCA_'
	    train_X=pd.read_csv('../input/PCA_train_15.csv')
	    #this data set. Any more and things get shaky, and values less are bad as well.
	    #train_Y=train_X[['target']]
	    target=train_X['target']
	    train_X=train_X.drop(['target'],axis=1)
	    #train_Y=train_Y.as_matrix()
	    #print (train_Y.shape)

	    num_labels = len(np.unique(target))
	    all_Y = np.eye(num_labels)[target]
	    print (all_Y[0:2][:],all_Y.shape)
	    #print (train_Y[0:2],all_Y[0:2][:])
	    train_X=train_X.as_matrix()
    print('daenris: my current real submission (daenris, not the obviously way overfit daenris1) is also a pretty simple tensorflow model with a single hidden layer, relu on the hidden layer sigmoid on the output layer, using the ADAM optimizer and using dropout on the hidden layer. Im also preprocessing the data with PCA on the training set, as the data can be described fully by far fewer than 21 features.')	    
    print('Single Layer with N=',h_size,' nodes, a batch size of ',batch_size)
				
    N,M=train_X.shape
    all_X = np.ones((N, M+ 1))#Shape of the data andA one Bias layer
    all_X[:, 1:] = train_X #Paste the test data on top	
    test=pd.read_csv('../input/PCA_test_15.csv')			
    test=test.as_matrix() #We want this in matrix form, not Pandas dataframes
    N, M  = test.shape
    test_X = np.ones((N, M + 1))#Shape of the data and one Bias layer
    test_X[:, 1:] = test #Paste the test data on top
    test_X=test_X.astype('float32') #Make sure it is in the right data-type
    test,valid='',''#Save Space

    model = Sequential()
    model.add(Dense(h_size, input_dim=M+1, init='uniform', activation='relu'))
    model.add(Dropout(0.5, noise_shape=None, seed=RANDOM_SEED))
    model.add(Dense(h_size, init='uniform', activation='relu'))
    model.add(Dropout(0.8, noise_shape=None, seed=RANDOM_SEED))
    model.add(Dense(2, init='uniform', activation='sigmoid'))   
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

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
    submission.to_csv('../output/'+filename+'.csv', index=False)	
    print ('Finished submission')

def TEN_Layer():
    nEpochs=100 #
    h_size = 20 # Number of hidden nodes
    batch_size=1000 #Process how many examples at once?
    PCA=True
    print ('http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/')
    print ('Loading data')
    if PCA:
	    tag='_PCA_'
	    train_X=pd.read_csv('../input/PCA_train_15.csv')
	    #this data set. Any more and things get shaky, and values less are bad as well.
	    #train_Y=train_X[['target']]
	    target=train_X['target']
	    train_X=train_X.drop(['target'],axis=1)
	    #train_Y=train_Y.as_matrix()
	    #print (train_Y.shape)

	    num_labels = len(np.unique(target))
	    all_Y = np.eye(num_labels)[target]
	    print (all_Y[0:2][:],all_Y.shape)
	    #print (train_Y[0:2],all_Y[0:2][:])
	    train_X=train_X.as_matrix()
    print('daenris: my current real submission (daenris, not the obviously way overfit daenris1) is also a pretty simple tensorflow model with a single hidden layer, relu on the hidden layer sigmoid on the output layer, using the ADAM optimizer and using dropout on the hidden layer. Im also preprocessing the data with PCA on the training set, as the data can be described fully by far fewer than 21 features.')	    
    print('Single Layer with N=',h_size,' nodes, a batch size of ',batch_size)
				
    N,M=train_X.shape
    all_X = np.ones((N, M+ 1))#Shape of the data andA one Bias layer
    all_X[:, 1:] = train_X #Paste the test data on top	
    test=pd.read_csv('../input/PCA_test_15.csv')			
    test=test.as_matrix() #We want this in matrix form, not Pandas dataframes
    N, M  = test.shape
    test_X = np.ones((N, M + 1))#Shape of the data and one Bias layer
    test_X[:, 1:] = test #Paste the test data on top
    test_X=test_X.astype('float32') #Make sure it is in the right data-type
    test,valid='',''#Save Space

    model = Sequential()
    model.add(Dense(h_size, input_dim=M+1, init='uniform', activation='relu'))
    model.add(Dropout(0.5, noise_shape=None, seed=RANDOM_SEED))
    model.add(Dense(h_size, init='uniform', activation='relu'))
    model.add(Dropout(0.8, noise_shape=None, seed=RANDOM_SEED))
    model.add(Dense(h_size, init='uniform', activation='relu'))
    model.add(Dropout(0.8, noise_shape=None, seed=RANDOM_SEED))
    model.add(Dense(h_size, init='uniform', activation='relu'))
    model.add(Dropout(0.8, noise_shape=None, seed=RANDOM_SEED))
    model.add(Dense(h_size, init='uniform', activation='relu'))
    model.add(Dropout(0.8, noise_shape=None, seed=RANDOM_SEED))
    model.add(Dense(h_size, init='uniform', activation='relu'))
    model.add(Dropout(0.8, noise_shape=None, seed=RANDOM_SEED))
    model.add(Dense(h_size, init='uniform', activation='relu'))
    model.add(Dropout(0.8, noise_shape=None, seed=RANDOM_SEED))
    model.add(Dense(h_size, init='uniform', activation='relu'))
    model.add(Dropout(0.8, noise_shape=None, seed=RANDOM_SEED))
    model.add(Dense(h_size, init='uniform', activation='relu'))
    model.add(Dropout(0.8, noise_shape=None, seed=RANDOM_SEED))
    model.add(Dense(h_size, init='uniform', activation='relu'))
    model.add(Dropout(0.8, noise_shape=None, seed=RANDOM_SEED))
    model.add(Dense(h_size, init='uniform', activation='relu'))
    model.add(Dropout(0.8, noise_shape=None, seed=RANDOM_SEED))
    model.add(Dense(2, init='uniform', activation='sigmoid'))   
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

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
#main()
TEN_Layer()
