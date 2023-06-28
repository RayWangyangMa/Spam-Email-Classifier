import pandas as pd  # import necessary libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score
import os  # os library is used to work with local system directories

dir_path = os.path.dirname(os.path.realpath(__file__))  # get the directory of the current script
data = pd.read_csv(os.path.join(dir_path, 'spam.csv'))  # read the csv file from the directory where the script is located

X = data.drop(['Prediction', 'Email No.'], axis=1)  # drop the 'Prediction' and 'Email No.' columns and use the rest as features
y = data['Prediction']  # 'Prediction' column is the target

# split the data into train and test sets, with 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10]}  # define the parameter grid for grid search

svc = svm.SVC()  # instantiate the SVM classifier
model = GridSearchCV(svc, parameters)  # setup the GridSearchCV with the SVM classifier and the parameter grid
model.fit(X_train, y_train)  # fit the model on the training data

print("Best Parameters:\n", model.best_params_)  # print the best hyperparameters found by grid search

predictions = model.predict(X_test)  # make predictions on the test data
print("Accuracy:", accuracy_score(y_test, predictions))  # print the accuracy of the model on the test data
