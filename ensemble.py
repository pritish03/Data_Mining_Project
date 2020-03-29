import numpy as np 
import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB	
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

#twilio
from twilio.rest import Client
account_sid = ''
auth_token = ''
client = Client(account_sid, auth_token)


df = pd.read_csv('datafile1.csv')
df = df.dropna(axis=0, how='any')
states = []
temp = []

#converting strings to number
def convYear(year):
	if year == "2012-13":
		return 0
	if year == "2013-14":
		return 1		
	if year == "2014-2015":
		return 2

def convMonth(Month):
	if Month == "January":
		return 1
	if Month == "February":
		return 2
	if Month == "March":
		return 3
	if Month == "April":
		return 4
	if Month == "May":
		return 5
	if Month == "June":
		return 6
	if Month == "July":
		return 7
	if Month == "August":
		return 8
	if Month == "September":
		return 9
	if Month == "October":
		return 10
	if Month == "November":
		return 11
	if Month == "December":
		return 12
	else:
		print(error)												

def convProduct(product):
	if product == "DAP":
		return 0
	if product == "MAP":
		return 1
	if product == "MOP":
		return 2
	if product == "NPK":
		return 3
	if product == "TSP":
		return 4
	if product == "UREA":
		return 5
	if product == "SSP":
		return 5						

def convState():
	list = df['State']
	for state in list:
		if state not in states:
			states.append(state) 
	for element in df['State']:
		temp.append(states.index(element))

df['Year'] = df['Year'].apply(convYear)
df['Month'] = df['Month'].apply(convMonth)
df['Product'] = df['Product'].apply(convProduct)
convState() #to generate temp that is then re-assigned to State
df['State'] = temp

		
#data splitting		
X = df[['Year','Month','Product','State','Availability']]
y = df[['Requirement']]
score = []
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)


#Random forest
rf = RandomForestRegressor(n_estimators = 50, random_state=42)
#random_state is the seed for the reandom genrator 
rf.fit(X_train, y_train.values.ravel())
score.append(rf.score(X_test, y_test))
	
#bagging classifier with knn	
bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
bagging.fit(X_train, y_train.values.ravel())
score.append(bagging.score(X_test, y_test))

#AdaBoost classifier 
m = AdaBoostClassifier(base_estimator=None, n_estimators=100)
m.fit(X_train, y_train.values.ravel())
score.append(m.score(X_test, y_test))

#GradientBoostingClassifier
m = GradientBoostingClassifier(n_estimators=100)
#warm start to use new training data to improve the model
m.fit(X_train, y_train.values.ravel())
score.append(m.score(X_test, y_test))

#voting classifier
m = VotingClassifier(estimators=[
	('lr', LogisticRegression()),
	('knn', KNeighborsClassifier()),
	('gnb', GaussianNB())],
	voting='hard')
m.fit(X_train, y_train.values.ravel())
score.append(m.score(X_test, y_test))



#sending the sms
string = ('Training report:- \n'
                  'Random forest = '+str(score[0])+'\n'
                  'bagging classifier with knn = '+str(score[1])+'\n'
                  'AdaBoost classifier = '+str(score[2])+'\n'
                  'GradientBoostingClassifier = '+str(score[3])+'\n'
                  'voting classifier = '+str(score[4]))

print(string)

message = client.messages.create(
            body= string,
            from_='',
            to='+919585583918')

message = client.messages.create(
            body= string,
            from_='',
            to='+919652291375')




