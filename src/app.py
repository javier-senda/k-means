import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


URL = "https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv"

df = pd.read_csv(URL, usecols=["Latitude", "Longitude", "MedInc"])

X_train, X_test = train_test_split(
    df, test_size=0.20, random_state=42, shuffle=True
)

X_train.to_csv("../data/train_with_clusters.csv", index=False)
X_test.to_csv("../data/test_with_clusters.csv", index=False)

model_kmeans = KMeans(n_clusters = 6, n_init = "auto", random_state = 42)
model_kmeans.fit(X_train)

y_train = list(model_kmeans.labels_)
X_train["cluster"] = y_train

import matplotlib.pyplot as plt
import seaborn as sns

fig, axis = plt.subplots(1, 3, figsize = (15, 5))

sns.scatterplot(ax = axis[0], data = X_train, x = "Latitude", y = "Longitude", hue = "cluster", palette = "deep")
sns.scatterplot(ax = axis[1], data = X_train, x = "Latitude", y = "MedInc", hue = "cluster", palette = "deep")
sns.scatterplot(ax = axis[2], data = X_train, x = "Longitude", y = "MedInc", hue = "cluster", palette = "deep")
plt.tight_layout()

plt.show()

y_test = list(model_kmeans.predict(X_test))
X_test["cluster"] = y_test

fig, axis = plt.subplots(1, 3, figsize = (15, 5))

sns.scatterplot(ax = axis[0], data = X_train, x = "Latitude", y = "Longitude", hue = "cluster", palette = "deep", alpha  = 0.1)
sns.scatterplot(ax = axis[1], data = X_train, x = "Latitude", y = "MedInc", hue = "cluster", palette = "deep", alpha  = 0.1)
sns.scatterplot(ax = axis[2], data = X_train, x = "Longitude", y = "MedInc", hue = "cluster", palette = "deep", alpha  = 0.1)

sns.scatterplot(ax = axis[0], data = X_test, x = "Latitude", y = "Longitude", hue = "cluster", palette = "deep", marker = "+")
sns.scatterplot(ax = axis[1], data = X_test, x = "Latitude", y = "MedInc", hue = "cluster", palette = "deep", marker = "+")
sns.scatterplot(ax = axis[2], data = X_test, x = "Longitude", y = "MedInc", hue = "cluster", palette = "deep", marker = "+")

plt.tight_layout()

for ax in axis:
    ax.legend([],[], frameon=False)

plt.show()

model_rf = RandomForestClassifier(random_state = 42)
model_rf.fit(X_train, y_train)

y_pred = model_rf.predict(X_test)
y_pred

accuracy_score(y_test, y_pred)

with open("../models/model_kmeans.pkl", "wb") as f:
    pickle.dump(model_kmeans, f)
    
with open("../models/random_forest.pkl", "wb") as f:
    pickle.dump(model_rf, f)