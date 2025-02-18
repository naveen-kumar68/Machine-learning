import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,classification_report

data = pd.read_csv(r"C:\Users\navee\Downloads\diabetes.csv")
print(data.info(),data.head())

class_independent=data[data["Outcome"]==0].drop(columns=["Outcome"])
class_dependent=data[data["Outcome"]==1].drop(columns=["Outcome"])

mean_class_ind=class_independent.mean(axis=0)
mean_class_dep=class_dependent.mean(axis=0)

print("Mean_class_0:",mean_class_ind)
print("Mean of Class_1:",mean_class_dep)

std_class_ind=class_independent.std(axis=0)
std_class_dep=class_dependent.std(axis=0)

print("STD_class_0:",std_class_ind)
print("STD of Class_1:",std_class_dep)

Eucledian_Distance=np.linalg.norm(mean_class_ind-mean_class_dep)

###minkwoski distance

feature_vec0=data["Glucose"]
feature_vec1=data["BloodPressure"]

print(feature_vec0)
print(feature_vec1)

feature_0=np.array(feature_vec0,dtype=float)
feature_1=np.array(feature_vec1,dtype=float)

for r in range(1,11):
 minkowski_distance=np.sum(np.abs(feature_0-feature_1)**r)**(1/r)
 print("Minkwoski Distance for r:{r}",minkowski_distance)

###### BY importing distance
minkowski_dist=[]
for r in range(1,11):
 minkowski_distance=distance.minkowski(feature_0,feature_1,r)
 print(f"Minkwoski Distance for r={r}={minkowski_distance}")
 minkowski_dist.append(minkowski_distance)

###plot distance

plt.plot(range(1,11),minkowski_dist,marker="o",linestyle="-",color="blue")
plt.xlabel("Feature_0")
plt.ylabel("Feature_1")
plt.show()


var0=class_independent.var(axis=0)
var1=class_dependent.var(axis=1)

print("Variance_class_0:",var0)
print("ariance of Class_1:",var1)

plt.hist(data["Outcome"],bins=2,edgecolor="black")
plt.xlabel("Outcome")
plt.ylabel("Frequency")
plt.show()

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

##print("Eucledian Dsitance:",Eucledian_Distance)
##feature_vectors=data[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]].values
##feature_mean=np.mean(feature_vectors,axis=0)
##print("Feature Means:",feature_mean)
##feature_std=np.std(data)
##print("Feature STD:",feature_std)####
