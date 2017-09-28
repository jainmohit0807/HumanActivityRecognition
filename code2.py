#Importing DataSet
import pandas as pd
ds_train=pd.read_csv('train.csv')
ds_test=pd.read_csv('test.csv')

#Divison Of Columns for Training
Xl=ds_train.iloc[:,:-1].values
Yl=ds_train.iloc[:,-1:].values

#Division Of Columns fot Testing
xl=ds_test.iloc[:,:-1].values
yl=ds_test.iloc[:,-1].values

#Encoding the data
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
Yl=labelencoder.fit_transform(Yl)
yl=labelencoder.fit_transform(yl)

#onehotencoder=OneHotEncoder()
#Yl=onehotencoder.fit_transform(Yl).toarray()

#onehotencoder=OneHotEncoder(categorical_features=[0])
#yl.reshape(-1,1)
#yl=onehotencoder.fit_transform(yl).toarray()

#onehotencoder=OneHotEncoder(categorical_features=[0])
#Yl=onehotencoder.fit_transform(Yl).toarray()

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=2)

Xl=lda.fit_transform(Xl,Yl)
xl=lda.transform(xl)

##pca
#import numpy as np
#from sklearn.decomposition import PCA
#pca=PCA(n_components=2)
#Xl=pca.fit_transform(Xl)
#xl=xl.astype(np.float64)
#xl=pca.fit_transform(xl)
#xp=pca.explained_variance_ratio_

#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(Xl,Yl)

y_pred=classifier.predict(xl)

acc=(classifier.score(xl,yl))*100

#K-NN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
knn.fit(Xl,Yl)

y_predk=knn.predict(xl)
cm=confusion_matrix(yl,y_predk)
acc_k=(knn.score(xl,yl))*100

#DecisionTree_GINI
from sklearn.tree import DecisionTreeRegressor
DTC=DecisionTreeRegressor(random_state=0)
DTC.fit(Xl,Yl)

y_pred_tree=DTC.predict(xl)
acc_tree=DTC.score(xl,yl)*100

#DecisionTree_Entropy
from sklearn.tree import DecisionTreeClassifier
DTC1=DecisionTreeClassifier(criterion='gini',random_state=0)
DTC1.fit(Xl,Yl)

y_pred_tree1=DTC1.predict(xl)
acc_tree1=DTC.score(xl,yl)*100

#RandomForest_GINI
from sklearn.ensemble import RandomForestRegressor
RFG=RandomForestRegressor(n_estimators=10,random_state=0)
RFG.fit(Xl,Yl)

y_pred_RFG=RFG.predict(xl)

acc_rfg=RFG.score(xl,yl)*100

#RandomForest_Entropy
from sklearn.ensemble import RandomForestClassifier
RFG1=RandomForestClassifier(n_estimators=10,criterion='gini',random_state=0)
RFG1.fit(Xl,Yl)

y_pred_RFG1=RFG.predict(xl)

acc_rfg1=RFG1.score(xl,yl)*100

import matplotlib.pyplot as plt
values=[acc,acc_k,acc_rfg,acc_rfg1,acc_tree,acc_tree1]
colors=['red','blue','black','cyan','yellow','orange']
plt.bar(range(0,6),values,color=colors,align='center')
index=range(0,5)
plt.xticks(index,('L-Reg','K_NN','Forest','Forest_E','Decision_G','Decision_E'))







