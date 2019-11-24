

#!/usr/bin/env python


import numpy as np
import pandas as pd
#

train_file = '../input/numerai_training_data.csv'
test_file = '../input/numerai_tournament_data.csv'

print ("Add in Validation set to the training set")
print ("loading...")
train = pd.read_csv( train_file )
test = pd.read_csv( test_file )
valid=test[test['data_type']=='validation']

x=pd.concat([train, valid])
	
output_file = '../input/train_valid.csv'
x.to_csv( output_file, index = False )


#print(train.describe() )
#print(x.describe() )
