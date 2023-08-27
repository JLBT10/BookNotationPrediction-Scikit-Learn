
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os



def evaluation(model,saving_dir,model_name,X_test,y_test):
    
    #Confusion_matrix dir
    if not os.path.exists('./../runs/{}/confusion_matrix/'.format(saving_dir)):
        os.makedirs('./../runs/{}/confusion_matrix/'.format(saving_dir))

    #Classification_report dir
    if not os.path.exists('./../runs/{}/Classification_report/'.format(saving_dir)):
        os.makedirs('./../runs/{}/Classification_report/'.format(saving_dir))

    y_test_predict = model.predict(X_test)
    with open('./../runs/{}/Classification_report/{}_classification_report'.format(saving_dir,model_name),'w') as f :
        f.write(classification_report(y_test, y_test_predict))

    print('Classification report on test set {}\n'.format(model_name))
    print(classification_report(y_test, y_test_predict))
    conf_matrix = confusion_matrix(y_test, y_test_predict.round())

    # Display the confusion matrix using a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('./../runs/{}/confusion_matrix/{}_confusion_matrix'.format(saving_dir,model_name))  # Provide the desired file name and path
