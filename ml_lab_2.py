import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean, variance
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df=pd.read_csv(r"C:\Users\navee\Downloads\Lab Session Data(Purchase data).csv")
df_numeric = df.select_dtypes(include=['number'])
A = df_numeric.iloc[2:10, :3].values
C = df_numeric.iloc[2:10, 3:4].values

print("A",A)
print("C",C)

dimensionality = np.linalg.matrix_rank(A)
print("Dimesional of vector space",dimensionality)
no_of_vectors=np.unique(A,axis=0)
print(f"Number of distinct vectors in matrix A: {no_of_vectors.shape[0]}")
pseudo_inverse=np.linalg.pinv(A)
print("Pseudo inverse of A",pseudo_inverse)
X = pseudo_inverse @ C
print("Model Vector X:\n", X)

data = df.iloc[:, :5].drop(columns=["Customer"])
data.columns = ["Candies", "Mangoes", "Milk_Packets", "Payment"]


X = np.column_stack((np.ones(len(data)), data[["Candies", "Mangoes", "Milk_Packets"]].values))
Y = data["Payment"].values

X_pinv = np.linalg.pinv(X)
model_parameters = X_pinv @ Y


print("Estimated Model Parameters (Intercept and Coefficients):")
print(model_parameters)

data["Customer Type"] = ["RICH" if amount > 200 else "POOR" for amount in data["Payment (Rs)"]]

features = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]]
target = data["Customer Type"]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.5, random_state=42)


knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)


predictions = knn_model.predict(X_test)


print("Classification Report")
print(classification_report(y_test, predictions))

price_column = data["Price"]
print(f"D = {price_column}")

price_mean = mean(price_column)
price_variance = variance(price_column)
print(f"The mean of column D is = {price_mean}")
print(f"The variance of column D is = {price_variance}")


data["Date"] = pd.to_datetime(data["Date"])
data["Weekday"] = data["Date"].dt.weekday

wednesday_prices = data[data["Weekday"] == 2]["Price"]
wednesday_mean = wednesday_prices.mean()
print(f"The sample mean for all Wednesdays in the dataset is = {wednesday_mean}")

data["Month"] = data["Date"].dt.month
april_prices = data[data["Month"] == 4]["Price"]
april_mean = mean(april_prices)
print(f"The sample mean for April in the dataset is = {april_mean}")


loss_probability = (data["Chg%"] < 0).mean()
print(f"The probability of making a loss in the stock is {loss_probability}")


profit_wednesdays = (data.loc[data["Weekday"] == 2, "Chg%"] > 0).mean()
print(f"The probability of making a profit in the stock on Wednesday is {profit_wednesdays}")


num_wed = len(wednesday_prices)
num_profitable_wed = (wednesday_prices > 0).sum()
conditional_prob_wed = num_profitable_wed / num_wed
print(f"The conditional probability of making a profit, given that today is Wednesday = {conditional_prob_wed}")

sns.scatterplot(x="Weekday", y="Chg%", data=data, hue="Weekday", palette="hls")
plt.xlabel("Day of the Week")
plt.ylabel("Chg%")
plt.title("Chg% Distribution by Day of the Week")
plt.show()

#Identify attribute types
categorical_cols = df.select_dtypes(include=['object']).columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Encoding categorical variables
label_encoders = {}
for col in categorical_cols:
    unique_values = df[col].nunique()
    if unique_values <= 10:  # Assuming ordinal encoding for variables with limited unique values
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    else:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# Data range for numeric variables
numeric_ranges = df[numeric_cols].describe()

# Missing values count
missing_values = df.isnull().sum()

# Outlier detection using boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numeric_cols])
plt.xticks(rotation=90)
plt.show()

# Mean and variance for numeric variables
numeric_stats = df[numeric_cols].agg(['mean', 'var'])

print("Data Types:\n", df.dtypes)
print("\nNumeric Ranges:\n", numeric_ranges)
print("\nMissing Values:\n", missing_values)
print("\nMean and Variance:\n", numeric_stats)



# Handling missing values

categorical_cols = df.select_dtypes(include=['object']).columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if col in numeric_cols:
            if np.any((df[col] - df[col].median()).abs() > 3 * df[col].std()):
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing values after imputation:\n", df.isnull().sum())


###
from sklearn.preprocessing import MinMaxScaler, StandardScaler

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
outlier_cols = [col for col in numeric_cols if np.any((df[col] - df[col].median()).abs() > 3 * df[col].std())]
non_outlier_cols = list(set(numeric_cols) - set(outlier_cols))

scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

df[non_outlier_cols] = scaler_standard.fit_transform(df[non_outlier_cols])
df[outlier_cols] = scaler_minmax.fit_transform(df[outlier_cols])

print("Normalized data sample:\n", df.head())


########
import numpy as np

binary_cols = [col for col in df.columns if set(df[col].unique()).issubset({0, 1})]
vector1 = df.iloc[0][binary_cols].values
vector2 = df.iloc[1][binary_cols].values

f11 = np.sum((vector1 == 1) & (vector2 == 1))
f00 = np.sum((vector1 == 0) & (vector2 == 0))
f01 = np.sum((vector1 == 0) & (vector2 == 1))
f10 = np.sum((vector1 == 1) & (vector2 == 0))

JC = f11 / (f01 + f10 + f11)
SMC = (f11 + f00) / (f00 + f01 + f10 + f11)

print(f"Jaccard Coefficient (JC): {JC}")
print(f"Simple Matching Coefficient (SMC): {SMC}")


##########
from sklearn.metrics.pairwise import cosine_similarity

vector1 = df.iloc[0].values.reshape(1, -1)
vector2 = df.iloc[1].values.reshape(1, -1)

cosine_sim = cosine_similarity(vector1, vector2)[0][0]

print(f"Cosine Similarity: {cosine_sim}")


