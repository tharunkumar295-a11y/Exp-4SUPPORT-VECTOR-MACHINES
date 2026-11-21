# Experiment 4: Support Vector Machine (SVM)

## Aim

To implement and analyze the performance of a Support Vector Machine
(SVM) model for classification using Python.

## Overview

A Support Vector Machine is a supervised machine-learning algorithm used
for classification. It finds a decision boundary (hyperplane) that
separates classes while maximizing the margin between the closest data
points, called **support vectors**.

This experiment uses the **Iris dataset** and compares two SVM kernels:

-   **Linear kernel** -- suitable when classes can be separated
    approximately by a straight decision boundary.
-   **RBF (Radial Basis Function) kernel** -- can model non-linear
    decision boundaries.

For visualization, the experiment uses the first two features of the
Iris dataset.

## Algorithm

1.  **Start.**
2.  Load the Iris dataset and select the first two features for
    visualization.
3.  Split the data into training and testing sets.
4.  Standardize the features using `StandardScaler`.
5.  Train an SVM classifier with a linear kernel.
6.  Predict the test labels and calculate the classification accuracy.
7.  Train another SVM classifier with the RBF kernel.
8.  Predict the test labels and calculate the classification accuracy.
9.  Plot the decision boundaries for both models.
10. **Stop.**

## Program
# Experiment 4: Support Vector Machine (SVM)

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # First two features for visualization
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM with Linear Kernel
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)

# Predict and evaluate
y_pred_linear = svm_linear.predict(X_test)
linear_accuracy = accuracy_score(y_test, y_pred_linear)

# Train SVM with RBF Kernel
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)

# Predict and evaluate
y_pred_rbf = svm_rbf.predict(X_test)
rbf_accuracy = accuracy_score(y_test, y_pred_rbf)

# Print accuracies
print("Linear Kernel Accuracy:", linear_accuracy)
print("RBF Kernel Accuracy:", rbf_accuracy)

# Function to plot decision boundary
def plot_decision_boundary(model, X, y, title, subplot):
    plt.subplot(1, 2, subplot)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

# Plot decision boundaries
plt.figure(figsize=(12, 5))

plot_decision_boundary(
    svm_linear,
    scaler.transform(X),
    y,
    "SVM with Linear Kernel",
    1
)

plot_decision_boundary(
    svm_rbf,
    scaler.transform(X),
    y,
    "SVM with RBF Kernel",
    2
)

plt.tight_layout()
plt.show()


## output
<img width="737" height="336" alt="image" src="https://github.com/user-attachments/assets/b479e60a-fe7e-4594-885a-074137a70dbe" />



## Result
The Support Vector Machine (SVM) classifier was successfully implemented using both Linear and RBF kernels. The classification accuracy was evaluated, and the decision boundaries were visualized. The RBF kernel generally provided better classification performance for non-linear class separations compared to the Linear kernel.
