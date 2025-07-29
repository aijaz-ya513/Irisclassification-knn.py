# Irisclassification-knn.py
✨ Iris Flower Classification using K-Nearest Neighbors

This project demonstrates a simple machine learning model using the K-Nearest Neighbors (KNN) algorithm to classify iris flowers based on their features. The Iris dataset is one of the most well-known datasets in pattern recognition and is built into Scikit-learn.

✨ 🔍 Project Overview

- **Algorithm Used:** K-Nearest Neighbors (KNN)
- **Dataset:** Iris dataset from `sklearn.datasets`
- **Goal:** Predict the species of an iris flower based on 4 input features.

✨ 📂 Dataset Details

The Iris dataset contains 150 samples of iris flowers, each with:
- **Features (4):**
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- **Target (Labels):**
  - 0 = Setosa
  - 1 = Versicolor
  - 2 = Virginica

✨ 🧠 Steps Performed

1. **Import Libraries**
2. **Load the Iris dataset**
3. **Explore data structure and sample features**
4. **Train the KNN model using all data**
5. **Predict the class of a sample flower**

✨✨✨🧪 Sample Code

```python
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
iris = datasets.load_iris()
features = iris.data
labels = iris.target

# Print one sample
print(features[0], labels[0])

# Train the model
clf = KNeighborsClassifier()
clf.fit(features, labels)

# Predict class for a new sample
pred = clf.predict([[5.1, 3.5, 1.4, 0.2]])
print(pred)