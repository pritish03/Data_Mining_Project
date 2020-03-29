import numpy as np 
import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB	

import pandas as pd
df = pd.read_csv("pulsar_stars.csv")
		
X = df[['Mean1','Std1','EK1','Skew1','Mean2','Std2','EK2','Skew2']]
y = df[['targetclass']]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3,random_state=109) # 70% training and 30% test

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(X_train,y_train.values.ravel())

y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

