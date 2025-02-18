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
