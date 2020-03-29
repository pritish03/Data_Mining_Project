import numpy as np 
import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB	
from sklearn.preprocessing import Imputer
import random
random.seed(500)

#Fetching the dataset
import pandas as pd
df = pd.read_csv("pulsar_stars.csv")

#data splitting		
X = df[['Mean1','Std1','EK1','Skew1','Mean2','Std2','EK2','Skew2']]
y = df[['targetclass']]


#Randomly replace 40% of the first column with NaN values
column = X['Mean1']
print(column.size)
missing_pct = int(column.size * 0.4)
i = [random.choice(range(column.shape[0])) for _ in range(missing_pct)]
column[i] = np.NaN
print(column.shape[0])
print(column)

imputer = Imputer(missing_values='NaN' , strategy = 'mean' , axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)
print(X[:,0])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y , test_size=0.3,random_state=500) # 70% training and 30% test

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets6
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



 
    
