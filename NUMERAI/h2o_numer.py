import numpy as np
import pandas as pd
import h2o


print ('Loading data')
h2o.init()
feats=["id",'era','data_type']
pred_columns=[]
for i in range(50):
    pred_columns.append("feature"+str(i+1).strip())
    feats.append("feature"+str(i+1).strip())
feats.append("target")
df=h2o.import_file("../input/numerai_training_data.csv")

test=h2o.import_file('../input/numerai_tournament_data.csv')
#valid=test[test['data_type']=='validation']



from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.deepwater import H2ODeepWaterEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator


#GBM=H2OGradientBoostingEstimator(
#        ntrees=10,
#        learn_rate=0.2,
#        learn_rate_annealing = 0.99,
#        sample_rate = 0.8,
#        col_sample_rate = 0.8,
#        seed = 1234,
#        score_tree_interval = 10, 
#        stopping_rounds = 5,
#        stopping_tolerance = 1e-4)
#
#GBM.train(x=pred_columns, y='target', training_frame=df)
#predictions = GBM.predict(test)
#h2o.download_csv(predictions,"../output/predictions_GBM.h2o")

nfolds = 5
my_gbm = H2OGradientBoostingEstimator(nfolds=nfolds, fold_assignment="Modulo", keep_cross_validation_predictions=True)
my_gbm.train(x=pred_columns, y='target', training_frame=df)
my_rf = H2ORandomForestEstimator(nfolds=nfolds, fold_assignment="Modulo", keep_cross_validation_predictions=True)
my_rf.train(x=pred_columns, y='target', training_frame=df)
#deep_water=H2ODeepWaterEstimator(nfolds=nfolds, fold_assignment="Modulo", keep_cross_validation_predictions=True)
#deep_water.train(x=pred_columns, y='target', training_frame=df)
deep_learn=H2ODeepLearningEstimator(nfolds=nfolds, hidden=[10,10,10,10,10,10,10,10,10],activation="Tanh", fold_assignment="Modulo", keep_cross_validation_predictions=True)
deep_learn.train(x=pred_columns, y='target', training_frame=df)
lin=H2OGeneralizedLinearEstimator(nfolds=nfolds, fold_assignment="Modulo", keep_cross_validation_predictions=True)
lin.train(x=pred_columns, y='target', training_frame=df)

stack = H2OStackedEnsembleEstimator(model_id="my_ensemble", training_frame=df, base_models=[my_gbm.model_id, my_rf.model_id,deep_learn.model_id,lin.model_id])
#stack = H2OStackedEnsembleEstimator(model_id="my_ensemble", training_frame=df, base_models=[my_gbm.model_id, my_rf.model_id, deep_water.model_id,deep_learn.model_id,lin.model_id])
#stack = H2OStackedEnsembleEstimator(model_id="my_ensemble", training_frame=df, base_models=[my_gbm.model_id, my_rf.model_id])
stack.train(x=pred_columns, y='target', training_frame=df)
stack.model_performance()

predictions = stack.predict(test)
h2o.download_csv(predictions,"../output/predictions_BIG.h2o")



#gbm_grid = H2OGradientBoostingEstimator(
#        ## more trees is better if the learning rate is small enough 
#        ## here, use "more than enough" trees - we have early stopping
#        ntrees=10000,
#        ## smaller learning rate is better
#        ## since we have learning_rate_annealing, we can afford to start with a 
#        #bigger learning rate
#        learn_rate=0.05,
#        ## learning rate annealing: learning_rate shrinks by 1% after every tree 
#        ## (use 1.00 to disable, but then lower the learning_rate)
#        learn_rate_annealing = 0.99,
#        ## sample 80% of rows per tree
#        sample_rate = 0.8,
#        ## sample 80% of columns per split
#        col_sample_rate = 0.8,
#        ## fix a random number generator seed for reproducibility
#        seed = 1234,
#        ## score every 10 trees to make early stopping reproducible 
#        #(it depends on the scoring interval)
#        score_tree_interval = 10, 
#        ## early stopping once the validation AUC doesn't improve by at least 0.01% for 
#        #5 consecutive scoring events
#        stopping_rounds = 5,
#        stopping_metric = "AUC",
#        stopping_tolerance = 1e-4)
#
##Build grid search with previously made GBM and hyper parameters
#from h2o.grid.grid_search import H2OGridSearch
#grid = H2OGridSearch(gbm_grid,hyper_params,
#                         grid_id = 'depth_grid',
#                         search_criteria = {'strategy': "Cartesian"})
#
#
##Train grid search
#grid.train(x=pred_columns, 
#           y="target",
#           training_frame = df)
#
#predictions = grid.predict(test)
#h2o.download_csv(predictions,"../output/predictions_GRID.h2o")

test_df=pd.read_csv('../input/numerai_tournament_data.csv')
#predictions=pd.read_csv('../output/predictions.h20')
predictions=pd.read_csv('../output/predictions_BIG.h2o')
print ('Writing Submission')
submission = pd.DataFrame({"id": test_df["id"], "probability": predictions['predict']})
submission=submission[['id','probability']]#For whatever reason, you have to flip these back.
print (submission.head())
submission.to_csv('../output/H2O.csv', index=False)	
print ('Finished submission')


"""                   
feats = [id	era	data_type	feature1	feature2	feature3	feature4	feature5	feature6	feature7	feature8	feature9	feature10	feature11	feature12	feature13	feature14	feature15	feature16	feature17	feature18	feature19	feature20	feature21	feature22	feature23	feature24	feature25	feature26	feature27	feature28	feature29	feature30	feature31	feature32	feature33	feature34	feature35	feature36	feature37	feature38	feature39	feature40	feature41	feature42	feature43	feature44	feature45	feature46	feature47	feature48	feature49	feature50	target]
"""