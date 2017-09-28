
import pandas as pd
ds_train=pd.read_csv('train.csv')
ds_test=pd.read_csv('test.csv')

#Divison Of Columns for Training
X=ds_train.iloc[:,:-1].values
Y=ds_train.iloc[:,[-1]].values

#Division Of Columns for Testing
x=ds_test.iloc[:,:-1].values
y=ds_test.iloc[:,[-1]].values

#Encoding the data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
Y=labelencoder.fit_transform(Y)
y=labelencoder.fit_transform(y)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X,Y)

y_pred=classifier.predict(x)

acc=(classifier.score(x,y))*100

"""from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y,y_pred)"""

#K-NN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
knn.fit(X,Y)

y_predk=knn.predict(x)

acc_k=(knn.score(x,y))*100

#DecisionTree_GINI
from sklearn.tree import DecisionTreeRegressor
DTC=DecisionTreeRegressor(random_state=0)
DTC.fit(X,Y)

y_pred_tree=DTC.predict(x)
acc_tree=DTC.score(x,y)*100

#DecisionTree_Entropy
from sklearn.tree import DecisionTreeClassifier
DTC1=DecisionTreeClassifier(criterion='gini',random_state=0)
DTC1.fit(X,Y)

y_pred_tree1=DTC1.predict(x)
acc_tree1=DTC.score(x,y)*100

#RandomForest_GINI
from sklearn.ensemble import RandomForestRegressor
RFG=RandomForestRegressor(n_estimators=10,random_state=0)
RFG.fit(X,Y)

y_pred_RFG=RFG.predict(x)

acc_rfg=RFG.score(x,y)*100

#RandomForest_Entropy
from sklearn.ensemble import RandomForestClassifier
RFG1=RandomForestClassifier(n_estimators=10,criterion='gini',random_state=0)
RFG1.fit(X,Y)

y_pred_RFG1=RFG.predict(x)

acc_rfg1=RFG1.score(x,y)*100

import matplotlib.pyplot as plt
values=[acc,acc_k,acc_rfg,acc_rfg1,acc_tree,acc_tree1]
colors=['red','blue','black','cyan','yellow','orange']
plt.bar(range(0,6),values,color=colors,align='center')
index=range(0,5)
plt.xticks(index,('L-Reg','K_NN','Forest','Forest_E','Decision_G','Decision_E'))



