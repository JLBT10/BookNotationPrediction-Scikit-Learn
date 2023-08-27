from train import *
from sklearn.model_selection import GridSearchCV

#xgb = joblib.load('./runs/exp22/weights/XGBClassifier.pkl')
X_train,y_train,X_test,y_test=processing()

# Definie the models
model=XGBClassifier()

#define the parameters for the grid search
'''params=[{"n_estimators":[100,200],
    "max_depth":[10,20,50],
    "learning_rate":[0.1,0.9]}]
'''
params=[{"n_estimators":[100],
    "max_depth":[20,50],
    "learning_rate":[0.04,0.05,0.08]}]


#runs grid_Search
print('**** Starting grid search**** ')
grid=GridSearchCV(model,params,cv=10,scoring="accuracy",return_train_score=True)
grid.fit(X_train,y_train)

#Get the best model with its parameters
best_model=grid.best_estimator_
#Save parameters and best_models
with open('./../grid_search/grid_search_{}.txt'.format(best_model.__class__),'w') as f:
    f.write(str(grid.param_grid )+ '\n')
    f.write(str(grid.best_estimator_))
    joblib.dump(model, './../grid_search/{}.pkl'.format('XGBClassifier_grid_search')) # Saving model


#Train with best_model
print(best_model.get_params())
train([best_model.__class__],['accuracy','precision_weighted'],[best_model.get_params()],X_train,y_train,X_test,y_test,10)