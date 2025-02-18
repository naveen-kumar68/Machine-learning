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
