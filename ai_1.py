import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn import svm


### Grid search for SVM
def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 100]
    gammas = [0.001, 0.01, 0.1, 10]
    # kernels = ['rbf','linear']
    param_grid = {'C': Cs, 'gamma' : gammas, 'kernel':('linear','rbf')}
    svr = svm.SVC()
    grid_search = model_selection.GridSearchCV(svr, param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    grid_search.best_score_

    print("BEST PARAMS: ",grid_search.best_params_)
    print("BEST SCORE: ",grid_search.best_score_)



# Pandas library is used to read the data set with .data extension
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data')
X = data.ix[:, 1:]  # Select columns 1 through end because the first column contains the target variable or the variable that we are going to predict
 


## Divide the dataset into 70% training and 30% testing , random_state is set to 42 so we can reproduce the same results.
# By default, shuffle is already set to True which means the training set and testing set are randomly chosen from the whole dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


## This should have been the part where we'll be able to choose the best parameters for our SVM model.
## However, it took me 10 hours to run it and it's still running. Do not delete this line muna kasi we'll take care of this later.
# svc_param_selection(X_train,y_train,10)



## Set the parameters for our model.
svm_clf = svm.SVC(gamma = 0.01, C = 100,probability=True)
## .fit basically means we're gonna train our model using the training data with the values we have from the train_test_split
svm_clf.fit(X_train,y_train)
## Test the accuracy of the trained model by feeding it the testing data
score_now = svm_clf.score(X_test,y_test)
print("SCORE: ",score_now)



## Check out the Random Forests,KNN and Naive Bayes at the SKLEARN documentation. 
