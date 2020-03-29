from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
import numpy as np 
import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB	
from sklearn.preprocessing import Imputer
from sklearn.metrics import f1_score, classification_report
import random
random.seed(0)
#Fetching the dataset
import pandas as pd
df = pd.read_csv("pulsar_stars.csv")

#data splitting		
X = df[['Mean1','Std1','EK1','Skew1','Mean2','Std2','EK2','Skew2']]
y = df[['targetclass']]

#Randomly replace 30% of the first column with NaN values
column = X['Skew2']
print(column.size)
missing_pct = int(column.size * 0.3)
i = [random.choice(range(column.shape[0])) for _ in range(missing_pct)]
column[i] = np.NaN
print(column.shape[0])
print(column)

import datawig

#Initialize a SimpleImputer model
imputer = datawig.SimpleImputer(
    input_columns=['EK2'], # column(s) containing information about the column we want to impute
    output_column= 'Skew2', # the column we'd like to impute values for
    output_path = 'imputer_model' # stores model data and metrics
    )

#Fit an imputer model on the train data
imputer.fit(train_df=X)

#Impute missing values and return original dataframe with predictions
X = imputer.predict(X)
X['Skew2'] = X['Skew2_imputed']
del X['Skew2_imputed']
print(X)

from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1000) # 70% training and 30% test



#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets6
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
