import numpy as np
import pandas as pd
import tensorflow as tf
####################


RANDOM_SEED=11
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden): 
    # this network is the same as the previous one except with an extra hidden layer 
    # + dropout
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.matmul(X, w_h))

    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))

    h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.matmul(h2, w_o)

def main():
    prob_limit=0.001
    nEpochs=100 #
    LR=0.3 #Learning rate.
    h_size = 25 # Number of hidden nodes
    batch_size=100 #Process how many examples at once?
    Adversarial=False

    print ('Loading data')
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
    else:
            train_X=pd.read_csv('../input/numerai_training_data.csv')
            train_X=train_X.drop(['id','era','data_type'],axis=1)
            target=train_X['target']
            train_X=train_X.drop(['target'],axis=1)
	    #train_Y=train_Y.as_matrix()
	    #print (train_Y.shape)

            num_labels = len(np.unique(target))
            all_Y = np.eye(num_labels)[target]
            print (all_Y[0:2][:],all_Y.shape)
            train_X=train_X.as_matrix()
	    
    print('Many Layer with N=',h_size,' nodes, a batch size of ',batch_size,'and learning rate=',LR)
				
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
    x_size = all_X.shape[1]   # Number of input nodes
    y_size = all_Y.shape[1]   # Number of outcomes
    

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations

    w_h = init_weights([x_size, h_size])
    w_h2 = init_weights([h_size, h_size])
    w_o = init_weights([h_size, y_size])

    p_keep_input = tf.placeholder("float")#Percentage of X that is retained from dropout
    p_keep_hidden = tf.placeholder("float")#Percentage of hidden layer that is kept 
    predict = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)
    predict_op = tf.argmax(predict, 1)
    
    # Backward propagation
    #cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
#https://nathanbrixius.wordpress.com/2016/05/23/a-simple-predictive-model-in-tensorflow/ Might help?				
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predict))
    regularize=False# Did worse on first step than others.
    if regularize:
        beta=0.01
        #Mess with your cost function, for your father's sake
        regularizer = tf.nn.l2_loss(w_h) + tf.nn.l2_loss(w_h2)+tf.nn.l2_loss(w_o)
        cost = tf.reduce_mean(cost + beta * regularizer)

    #updates = tf.train.AdamOptimizer(LR).minimize(cost)#
    #updates = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
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
            sess.run(updates, feed_dict={X: all_X[n:n+batch_size], y:all_Y[n:n+batch_size],p_keep_input: 0.5, p_keep_hidden:0.8})
            n+=batch_size
        #train_accuracy = np.mean(np.argmax(train_Y, axis=1) == sess.run(predict, feed_dict={X: all_X, y: train_Y}))
        train_accuracy = np.mean(np.argmax(all_Y, axis=1) == sess.run(predict_op, feed_dict={X: all_X, y:all_Y,p_keep_input: 1.0, p_keep_hidden:1.0}))
        valid_accuracy = np.mean(np.argmax(valid_Y, axis=1) == sess.run(predict_op, feed_dict={X: valid_X, y:valid_Y,p_keep_input: 1.0, p_keep_hidden:1.0}))
        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy, 100. * valid_accuracy))
        #sess.run(tf.print(w_1))
        
    print('Done Training.')    
    train_X, train_Y, all_X= '','','' #Save space for your father's sake  
    print ('Testing')
    predictions=sess.run(predict, feed_dict={X: test_X,p_keep_input: 0.8, p_keep_hidden:0.5})
    print(predictions[0:5][:])
    soft_prob=sess.run(tf.nn.softmax(predictions))
    soft_prob=pd.DataFrame(soft_prob)
    soft_prob=soft_prob[1]
    sess.close()
    print('Done testing!')
    

    print ('Writing Submission')
    print(predictions.shape)
    test=pd.read_csv('../input/numerai_tournament_data.csv')
    filename='NN_2Layer'+'_'+str(h_size)+'_'+str(LR)
    submission = pd.DataFrame({"id": test["id"], "probability": soft_prob})
    print(submission.describe())
    submission=submission[['id','probability']]#For whatever reason, you have to flip these back.
    print (submission.head() )
    submission.to_csv('../output/'+filename+'.csv', index=False)	
    print ('Finished submission')
    return 0				
				    		
 


####################
main()
