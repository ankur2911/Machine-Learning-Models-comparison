
# coding: utf-8

# In[3]:

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


# In[4]:

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['Number of time preg', 'plasma', 'BP', 'Triceps skin', 'serum insulin', 'body mass', 'pedigree', 'age', 'class']

df = pd.read_csv(url, names=names)


# In[27]:

print(df.shape)
array = df.values


# In[6]:

X = array[:,0:8]
Y = array[:,8]
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X)


# In[7]:

# split data into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,random_state=99)


# In[25]:

#Naive Bayes
from sklearn import naive_bayes
clf = naive_bayes.GaussianNB()

clf.fit(x_train, y_train)
print(roc_auc_score(y_test, clf.predict_proba(x_test)[:,1], average='weighted'))
prdict = clf.predict(x_test)
print(classification_report(y_test, prdict))


# In[12]:

#Decision Tree 
from sklearn.tree import DecisionTreeClassifier
X_train_scaled = ss.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X_train_scaled, Y)
clf1 = DecisionTreeClassifier(criterion='entropy', splitter='best' ,random_state=40, presort=False, min_samples_split=5)
scores = cross_val_score(clf1, X_train_scaled, Y, cv=10)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clf1 = clf1.fit(x_train, y_train)
print(roc_auc_score(y_test, clf1.predict_proba(x_test)[:,1], average='weighted'))
prdict1 = clf1.predict(x_test)
print(classification_report(y_test, prdict1))


# In[14]:

#Perceptron
from sklearn.linear_model import Perceptron
X_train_scaled = ss.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X_train_scaled, Y)
clf2=Perceptron(random_state=79,penalty='l1', alpha=0.000001,shuffle=True)
scores = cross_val_score(clf2, X_train_scaled, Y, cv=10)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clf2 = clf2.fit(x_train, y_train)
prdict2 = clf2.predict(x_test)
print(classification_report(y_test, prdict2))


# In[17]:

#SVM
from sklearn.svm import LinearSVC
X_train_scaled = ss.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X_train_scaled, Y)
clf3 = LinearSVC(penalty='l2', loss='hinge', dual=True, tol=0.001, C=1.0, random_state=216)
scores = cross_val_score(clf3, X_train_scaled, Y, cv=10)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clf3.fit(x_train, y_train) 
prdict3 = clf3.predict(x_test)
print(classification_report(y_test, prdict3))


# In[18]:

#Logistic Regression
from sklearn.linear_model import LogisticRegression
X_train_scaled = ss.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X_train_scaled, Y)
clf4=LogisticRegression(max_iter=10,random_state=90, penalty='l2')
scores = cross_val_score(clf4, X_train_scaled, Y, cv=10)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clf4.fit(x_train, y_train) 
prdict4 = clf4.predict(x_test)
print(classification_report(y_test, prdict4))


# In[19]:

#k-Nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
X_train_scaled = ss.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X_train_scaled, Y)
clf5 = KNeighborsClassifier(n_neighbors=10, algorithm='auto',leaf_size=40, p=6, n_jobs=6)
scores = cross_val_score(clf5, X_train_scaled, Y, cv=10)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clf5.fit(x_train, y_train) 
prdict5 = clf5.predict(x_test)
print(classification_report(y_test, prdict5))


# In[20]:

#Random Forest
from sklearn.ensemble import RandomForestClassifier
X_train_scaled = ss.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X_train_scaled, Y)
clf6 = RandomForestClassifier(n_estimators=35, max_depth=15, random_state=95, min_samples_leaf=5)
scores = cross_val_score(clf6, X_train_scaled, Y, cv=10)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clf6.fit(x_train, y_train) 
prdict6 = clf6.predict(x_test)
print(classification_report(y_test, prdict6))


# In[21]:

#Bagging
from sklearn.ensemble import BaggingClassifier
X_train_scaled = ss.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X_train_scaled, Y)
clf7 = BaggingClassifier(random_state=90, n_estimators=21, max_samples=1.0)
scores = cross_val_score(clf7, X_train_scaled, Y, cv=10)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clf7.fit(x_train, y_train) 
prdict7 = clf7.predict(x_test)
print(classification_report(y_test, prdict7))


# In[26]:

#Adaboost
from sklearn.ensemble import AdaBoostClassifier
X_train_scaled = ss.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X_train_scaled, Y)
clf8 =  AdaBoostClassifier(learning_rate=0.8, n_estimators=7, random_state=90)
clf8.fit(x_train, y_train) 
scores = cross_val_score(clf8, X_train_scaled, Y, cv=10)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
prdict8 = clf8.predict(x_test)
print(classification_report(y_test, prdict8))


# In[23]:

#Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
X_train_scaled = ss.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X_train_scaled, Y)
clf9 =  GradientBoostingClassifier(loss='exponential', learning_rate=1.0, n_estimators=10, subsample=1.0, criterion='mae', min_samples_split=2)
clf9.fit(x_train, y_train) 
scores = cross_val_score(clf9, X_train_scaled, Y, cv=10)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
prdict9 = clf9.predict(x_test)
print(classification_report(y_test, prdict9))


# In[24]:

#Neural Network
from sklearn.neural_network import MLPClassifier
X_train_scaled = ss.fit_transform(X)
clf10 = MLPClassifier(hidden_layer_sizes=[130, 110], max_iter=2100, random_state=100, solver='sgd', learning_rate='adaptive', alpha=0.001, learning_rate_init=0.9)
scores = cross_val_score(clf10, X_train_scaled, Y, cv=10)
clf10.fit(x_train, y_train)
prdict10 = clf10.predict(x_test)
print(classification_report(y_test, prdict10))
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

