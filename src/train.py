import pandas as pd
import joblib
##Scitkit learn
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

# Importing necessary libraries
from imblearn.over_sampling import SMOTE
from collections import Counter

## construction de pipeline
from sklearn.pipeline import Pipeline
from transformer_functions import *

##Split data
from sklearn.model_selection import train_test_split

## Model-scitkit-learn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 

#Test metrics
from eval import *


def processing():
    '''
    Load train and test set, process them using pipleline, encode the variable y_train and y_test and SMOTE the training set.

    Returns : X_train,y_train,X_test,y_test : ready to be passed to the models
    
    '''
    #Load train and test data
    #Let's perform a stratify split by average_rating

    print('**** LOADING THE DATASET **** \n')

    df=pd.read_csv('./../datasets/books.csv',on_bad_lines='skip')
    df.rename(columns={'  num_pages':'num_pages'},inplace=True)

    print('\n **** CREATING CATEGORY CLASSES**** \n')
    df['average_ratings_category']=pd.cut(df['average_rating'],bins=[-1,2,3,4,5],labels=['0','1','2','3'])
    print(df['average_ratings_category'].value_counts())

    print('**** SPLITING THE DATASET INTO TEST AND TRAIN **** \n')
    #Split the data into train and test.
    train, test= train_test_split(df,test_size=0.2,random_state=42,stratify=df['average_ratings_category'])

    # Save train and test
    train.to_csv('./../datasets/train.csv',index=False)
    test.to_csv('./../datasets/test.csv',index=False)

    #train=pd.read_csv('datasets/train.csv')
    #test=pd.read_csv('datasets/test.csv')

    #Columns to be delete 
    dropped_columns=['publication_date', 'isbn13', 'title','isbn', 'bookID','authors','average_rating','language_code','publisher','authors_split']
    print('**** STARTING PIPELINE PROCESSING **** \n')
    #Let's define the pipeline
    pipeline = Pipeline([
    ("bpt",BooksProcessingTransformer())
    ,('ant',AuthorsNotationTransformer()),
    ('aft',AuthorsFeaturesTransformer()),
    ('crt',ColumnRemoverTransformer(dropped_columns)),
    ("stsc",StandardScalerWithNames())
    ])
    
    # Fit and transform train and test sets
    train=pipeline.fit_transform(train)
    test=pipeline.transform(test)


    #y_train and X_train
    y_train=train['average_ratings_category']
    X_train=train.drop(['average_ratings_category'],axis=1)

    #-y_test and X_test
    y_test=test['average_ratings_category']
    X_test=test.drop(['average_ratings_category'],axis=1)

    # Encoding labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train=label_encoder.transform(y_train)
    y_test=label_encoder.transform(y_test)

    print('**** SMOTE TRAIN DATASET **** \n')
    # Applied SMOTE on the train set for training
    print("Class distribution before SMOTE:", Counter(y_train))
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Displaying the class distribution after applying SMOTE
    print("Class distribution after SMOTE:", Counter(y_train),'\n')

    return X_train,y_train,X_test,y_test


def train(models,metrics,params,X_train,y_train,X_test,y_test,cv):

    '''
        Takes in argument : 
        models(list): list of models to train
        metrics(list): list of metrics to compute
        params(list of dictionnary): list of parameters if no parameters --> [{}]
        X_train,y_train,X_test,y_test : variables to be use for training and testing
        cv : the number of split for the cross validation

        example : 
                    train([XGBClassifier, SVC, RandomForestClassifier],
                    ['accuracy', 'precision_weighted'],
                    params=[{"n_estimators":200,"max_depth":15,"learning_rate":0.1,"subsample":0.1,"colsample_bytree":0.3},{},{}])
                    X_train,y_train,X_test,y_test,10)
    
    '''
    #Create runs directory if not already created
    if not os.path.exists('./../runs/'):
        os.makedirs('./../runs/')
    saved_file=sorted([file for file in os.listdir('../runs') if not file.startswith('.')])

    #Create different saving_dir for different runs
    if len(saved_file)==0:
        os.mkdir("./../runs/exp0")
        saving_dir='exp0'
    else:
        saved_file=sorted([file for file in os.listdir('./../runs') if not file.startswith('.')])
        ordered_list= sorted([int(file.replace('exp','')) for file in saved_file])
        last_number=ordered_list[-1]
        os.mkdir("./../runs/exp{}".format(last_number+1) ) 
        saving_dir='exp{}'.format(last_number+1)

    #Create weights directory for saving weights
    if not os.path.exists('./../runs/{}/weights/'.format(saving_dir)):
        os.makedirs('./../runs/{}/weights/'.format(saving_dir))
        
   
    #Create dictionnary to save final results
    results={}
    for met in metrics:
        results['{}_train'.format(met)]={}

    #Iteration through the different models
    for model_class,par in zip(models,params):
            model_name = model_class.__name__ # get the name of the model
            if par=={}:
                model = model_class() # load the model
            else:
                model = model_class(**par) # load the model

            print('\n **** STARTING TRAINING WITH {}**** \n '.format(model_name))  
            model.fit(X_train, y_train) # Train 

            # Evaluate on each metric contains in metrics
            print('**** COMPUTING TRAIN METRICS ...**** \n')
            for metric in metrics: 
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric, n_jobs=-1)
                results['{}_train'.format(metric)][model_name] = round(scores.mean(), 4)
                print('Train_{}'.format(metric), ':', '{}\n'.format(round(scores.mean(), 4)) )
                

            print('**** STARTING EVALUATION WITH {}**** \n'.format(model_name))  
            evaluation(model,saving_dir,model_name,X_test,y_test) #Evaluation on test sets

            print('Saving the model in runs/{}'.format(saving_dir))
            joblib.dump(model, './../runs/{}/weights/{}.pkl'.format(saving_dir,model_name)) # Saving model

    results=pd.DataFrame(results)
    results.to_csv('./../runs/{}/train_results.csv'.format(saving_dir,model_name)) # Saving results
    return results


def main():

    '''apply processing and runs the model'''

    #Get X_train,y_train,X_test,y_test
    X_train,y_train,X_test,y_test=processing()

    #Models
    models = [ DecisionTreeClassifier ,XGBClassifier, RandomForestClassifier]

    #Metrics
    metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

    #Params
    params=[
        {'random_state': 1,
            'max_depth': 15,
            'criterion': 'gini',
            'min_samples_split': 4},

    {'random_state': 1,
    'n_estimators': 100,
    'learning_rate': 0.07,
    'n_jobs': -1},

    {'random_state': 1,
    'n_estimators': 100,
    'max_depth': 30,'n_jobs': -1}]

    #Train
    res=train(models,metrics,params,X_train,y_train,X_test,y_test,10)

    #Save result and print
    #res.to_csv('resultats.csv')
    print(res)

if __name__ == "__main__":
    main()