import numpy as np
import pandas as pd
import tensorflow as tf
####################


print('Repurposed from: https://gist.github.com/vinhkhuc/e53a70f9e5c3f55852b0')
RANDOM_SEED=11
def init_weights(shape):
    """ Weight initialization """
    #Can also use Random normal if you prefer, but does slightly worse
    #weights = tf.truncated_normal(shape, stddev=0.1)
    weights = tf.truncated_normal(shape, stddev=1.0/shape[1])#Inversely proportional to the number of input units
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    #h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    h    = tf.nn.relu(tf.matmul(X, w_1))  # 
    yhat = tf.matmul(h, w_2)  # The \varphi function. Should the RELU be here?
    return yhat

def main():
    #Prob limit=[0.1    ,0.01
    #Ending_Valid=[51.31,
    prob_limit=0.00001
    nEpochs=10000 #Doing 100 did not seem to help that much.
    LR=0.2 #Learning rate.
    h_size = 10 # Number of hidden nodes
    batch_size=100 #Process how many examples at once?
    Adversarial=True

    print ('Loading data')
    #print('daenris: my current real submission (daenris, not the obviously way overfit daenris1) is also a pretty simple tensorflow model with a single hidden layer, relu on the hidden layer sigmoid on the output layer, using the ADAM optimizer and using dropout on the hidden layer. I'm also preprocessing the data with PCA on the training set, as the data can be described fully by far fewer than 21 features.')
    if Adversarial:
	    tag='_Adversarial_'+str(prob_limit)
	    df=pd.read_csv('../input/train_sorted.csv')
	    #print (df.describe())
	    train_X=df[df['p']>prob_limit]#Keep everything that has a good chance of being similar to the test set. 25% is about the sweet spot for 
	    print ('Using only this fraction of the data: ',float(len(train_X))/float(len(df)) )
	    #this data set. Any more and things get shaky, and values less are bad as well.
	    train_X=train_X.drop(['era','data_type','p'],axis=1)
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
        
	    
    print('Single Layer with N=',h_size,' nodes, a batch size of ',batch_size,'and learning rate=',LR)
				
    N,M=train_X.shape
    all_X = np.ones((N, M+ 1))#Shape of the data and one Bias layer
    all_X[:, 1:] = train_X #Paste the test data on top	
    test=pd.read_csv('../input/numerai_tournament_data.csv')
    valid=test[test['data_type']=='validation']
    target=valid['target'].astype('int')
    num_labels = len(np.unique(target))
    valid_Y = np.eye(num_labels)[target]
    print (valid_Y[0:2][:],valid_Y.shape)
    valid=valid.drop(['id','era','data_type','target'],axis=1)
    test=test.drop(['id','era','data_type','target'],axis=1)				
    test=test.as_matrix() #We want this in matrix form, not Pandas dataframes
    N, M  = test.shape
    test_X = np.ones((N, M + 1))#Shape of the data and one Bias layer
    N, M  = valid.shape
    valid_X=np.ones((N, M + 1))
    test_X[:, 1:] = test #Paste the test data on top
    valid_X[:,1:]=valid
    test_X=test_X.astype('float32') #Make sure it is in the right data-type
    valid_X=valid_X.astype('float32')
    test,valid='',''#Save Space

    # Layer's sizes
    x_size = all_X.shape[1]   # Number of input nodes: 784 features and 1 bias
    y_size = all_Y.shape[1]   # Number of outcomes
    #print(x_size,y_size,test_X.shape[1])

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)
    #predict = tf.argmax( tf.matmul( tf.nn.relu(tf.matmul(X, w_1)), w_2), axis=1)
    predict_prob =tf.matmul( tf.nn.relu(tf.matmul(X, w_1)), w_2)

    
    # Backward propagation
    #cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
#https://nathanbrixius.wordpress.com/2016/05/23/a-simple-predictive-model-in-tensorflow/ Might help?				
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    #cost = tf.sqrt((yhat - y) * (yhat - y))

    regularization=False
    if regularization:
        beta=0.01
        reg=tf.nn.l2_loss(w_1)+tf.nn.l2_loss(w_2)
        cost=tf.reduce_mean(cost+beta*reg)
        
    #updates = tf.train.AdamOptimizer(LR).minimize(cost)#Adam works only alright.
    updates = tf.train.GradientDescentOptimizer(LR).minimize(cost)
    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    print('Starting to Train')
    nBatches=int(float(len(train_X))/float(batch_size))
    for epoch in range(nEpochs):
        # Train with each example
        n=0
        #for i in range(len(train_X)):
        for i in range(nBatches):
            #sess.run(updates, feed_dict={X: all_X[n:n+batch_size], y:train_Y[n:n+batch_size]})
            sess.run(updates, feed_dict={X: all_X[n:n+batch_size], y:all_Y[n:n+batch_size]})
            n+=batch_size
        #train_accuracy = np.mean(np.argmax(train_Y, axis=1) == sess.run(predict, feed_dict={X: all_X, y: train_Y}))
        train_accuracy = np.mean(np.argmax(all_Y, axis=1) == sess.run(predict, feed_dict={X: all_X, y: all_Y}))
        valid_accuracy = np.mean(np.argmax(valid_Y, axis=1) == sess.run(predict, feed_dict={X: valid_X, y: valid_Y}))
        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy, 100. * valid_accuracy))

        
    print('Done Training.')    
    train_X, train_Y, all_X= '','','' #Save space for your father's sake  
    print ('Testing')
    predictions=sess.run(predict_prob, feed_dict={X: test_X})
    print(predictions[0:5][:])
    soft_prob=sess.run(tf.nn.softmax(predictions))
    print(soft_prob[0:5][:])
    #print(soft_prob[0:5][1])
    #predict_1=np.argmax(soft_prob,axis=1)
    #print (predict_1)
    #predict_1=predictions=sess.run(predict, feed_dict={X: test_X})
    soft_prob=pd.DataFrame(soft_prob)
    soft_prob=soft_prob[1]
    sess.close()
    print('Done testing!')
    

    print ('Writing Submission')
    print(predictions.shape)
    test=pd.read_csv('../input/numerai_tournament_data.csv')
    filename='NN_Out_OneLayer'+tag+'_'+str(h_size)+'_'+str(LR)+'_'+str(nEpochs)
    submission = pd.DataFrame({"id": test["id"], "probability": soft_prob})
    print(submission.describe())
    submission=submission[['id','probability']]#For whatever reason, you have to flip these back.
    print (submission.head() )
    submission.to_csv('../output/'+filename+'.csv', index=False)	
    print ('Finished submission')
    return 0				
				    		
 


####################
main()
