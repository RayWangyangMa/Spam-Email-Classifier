import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score

data = pd.read_csv('Desktop\\ML project\\spam.csv')

X = data.drop(['Prediction', 'Email No.'], axis=1)
y = data['Prediction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10]}

svc = svm.SVC()
model = GridSearchCV(svc, parameters)
model.fit(X_train, y_train)

print("Best Parameters:\n", model.best_params_)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
