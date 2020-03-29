import numpy as np 
import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB	
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
import random
random.seed(500)

#Fetching the dataset
import pandas as pd
df = pd.read_csv("pulsar_stars.csv")

#data splitting		
X = df[['Mean1','Std1','EK1','Skew1','Mean2','Std2','EK2','Skew2']]
y = df[['targetclass']]


#Randomly replace 40% of the first column with NaN values
column = y['targetclass']
print(column.size)
missing_pct = int(column.size * 0.4)
i = [random.choice(range(column.shape[0])) for _ in range(missing_pct)]
column[i] = np.NaN
print(column.shape[0])
print(column)

#Impute the values using scikit-learn SimpleImpute Class
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer( strategy='median') 
imp_mean.fit(y)
imputed_train_df = imp_mean.transform(y)
print(imputed_train_df)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, imputed_train_df , test_size=0.3,random_state=500) # 70% training and 30% test

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets6
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



 
    
