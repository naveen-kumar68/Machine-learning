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
