import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,classification_report

data = pd.read_csv(r"C:\Users\navee\Downloads\diabetes.csv")
print(data.info(),data.head())

X=data.drop(columns=["Outcome"])
y=data["Outcome"]
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 

neigh = KNeighborsClassifier(n_neighbors=3) 
neigh.fit(X, y)

print("Accuracy for K=3:",neigh.score(X_train, y_train))
print("Accuracy for K=3:",neigh.score(X_test, y_test))

y_train_predicted=neigh.predict(X_train)
y_test_predicted=neigh.predict(X_test)

print("Perdicted_Values:",y_train_predicted)
print("Perdicted_Values:",y_test_predicted)

neigh_1 = KNeighborsClassifier(n_neighbors=1) 
neigh_1.fit(X, y)

print("Accuracy for K=1:",neigh_1.score(X_test, y_test))

accuracy_score=[]
for k in range(1,12):
 model = KNeighborsClassifier(n_neighbors=k) 
 model.fit(X, y)
 score=model.score(X_test,y_test)
 accuracy_score.append(score)
print("Accuracy from k=1 to 11:",accuracy_score)

plt.plot(range(1,12),accuracy_score,marker="o",linestyle="-",color="green")
plt.xlabel("No.of neighbours (k)")
plt.ylabel("Accuracy")
plt.show()

##Confusion matrix

confusion_matrix_train=confusion_matrix(y_train,y_train_predicted)
confusion_matrix_test=confusion_matrix(y_test,y_test_predicted)

print("Confusion_matrix for train=",confusion_matrix_train)
print("Confusion_matrix for test=",confusion_matrix_test)

print(classification_report(y_train,y_train_predicted))
print(classification_report(y_test,y_test_predicted))

#### Precision

precision_train=precision_score(y_train,y_train_predicted)
precision_test=precision_score(y_test,y_test_predicted)

print("Precision for train=",precision_train)
print("Precision for test=",precision_test)

###  Recall

Recall_train=recall_score(y_train,y_train_predicted)
Recall_test=recall_score(y_test,y_test_predicted)

print("Recall for train=",Recall_train)
print("Recall for test=",Recall_test)

### F1-Score

f1score_train=f1_score(y_train,y_train_predicted)
f1score_test=f1_score(y_test,y_test_predicted)

print("F1-Score for train=",f1score_train)
print("F1-Score for test=",f1score_test)
