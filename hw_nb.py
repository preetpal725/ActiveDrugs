# data preprocessing
# pca
# decision tree
# random forest
# f1 score

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# Importing the dataset
dataset = pd.read_csv("C:/Users/ishme/Downloads/train_drugs.txt", header=None, sep = "\t")
test = pd.read_csv("C:/Users/ishme/Downloads/test.txt", header=None, sep = "\t")
y_test_df = pd.read_csv("C:/Users/ishme/Downloads/format.txt", header=None)


# Getting the dataframe in right format
temp_df = dataset[1].str.split(" ", n = 0, expand = True)
for col in temp_df.columns: 
    temp_df[col] = pd.to_numeric(temp_df[col])
dataset = dataset.drop([1], axis = 1)
df = pd.concat([dataset, temp_df], axis = 1)

# Dropping NaN columns
df = df.dropna(axis = 1)


# Getting the test dataframe in right format
temp_df = test[0].str.split(" ", n = 0, expand = True)
for col in temp_df.columns: 
    temp_df[col] = pd.to_numeric(temp_df[col])
test_df = temp_df

# Dropping NaN columns
test_df = test_df.dropna(axis = 1)


# Converting 
X_train = df.iloc[:, 1:test_df.shape[1]+1].values
y_train = df.iloc[:, 0].values
X_test = test_df.values
y_test = y_test_df.values


# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Applying LDA
lda = LDA(n_components = 50)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# F1 Score
f1 = f1_score(y_test, y_pred, average='binary')
print(f1)



