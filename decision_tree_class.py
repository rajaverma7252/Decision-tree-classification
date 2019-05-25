# Decision tree classification
"""
Created on Fri Jun 29 09:16:33 2018

@author: Raja
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv('C:\\Users\\Raja\\Desktop\\ml\\decision_tree_classification\\Social_Network_Ads.csv')

X=dataset.iloc[:,2:4].values
y=dataset.iloc[:,4].values


#splitting the dataset into training set and test set
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)

#Feature scalling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier =DecisionTreeClassifier(criterion='entropy') #can also add random state here
classifier.fit(X_train,y_train)

#Predicting the test set results
y_predict = classifier.predict(X_test)

#Making the confusion Matrix and it is used to no. of prediction values which are wrong
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predict)


#Visualising the Test set results
from matplotlib.colors import ListedColormap

#we can write one of the following two set 1.for test, 2.for train here i used for test, i can also use for train
X_set,y_set = X_test,y_test
#X_set,y_set = X_train,y_train

X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min() -1, stop = X_set[:,0].max() +1,step = 0.01), 
                    np.arange(start = X_set[:,1].min() -1, stop = X_set[:,1].max() +1, step = 0.01))
plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0],X_set[y_set == j,1],
                c= ListedColormap(('red','green'))(i),label=j)
plt.title('Logestic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
