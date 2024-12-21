# Grid Search for Model Tuning and Evaluation

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# The dataset contains Age, Estimated Salary, and Purchased columns
dataset = pd.read_csv(r"D:\FSDS Material\Dataset\Classification\Vehicle Purchase Prediction.csv")
x = dataset.iloc[:, 2:4].values  # Independent variables: Age and Estimated Salary
y = dataset.iloc[:, -1].values   # Dependent variable: Purchased (0 or 1)

# Feature Scaling
# Standardizing the data to ensure features have zero mean and unit variance
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

# Splitting the dataset into the Training set and Test set
# Using an 80-20 split for training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC()  # Default kernel is 'rbf'
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix to evaluate the predictions
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculating accuracy of the model
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

# Bias evaluation (training accuracy)
bias = classifier.score(x_train, y_train)
bias

# Variance evaluation (Testing accuracy)
variance = classifier.score(x_test, y_test)
variance


# Applying k-Fold Cross Validation
# Splitting the training set into k-folds and calculating mean accuracy
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X =x_train, y=y_train, cv=5)
print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
# print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))


# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},  # Linear kernel options
    {'C': [1, 10, 100, 1000], 'kernel': ['rbf'],      # RBF kernel options
     'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}  # Regularization parameter
]
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10)  # 10-fold cross-validation
grid_search = grid_search.fit(x_train, y_train)
best_accuracy = grid_search.best_score_  # Best accuracy from grid search
best_parameters = grid_search.best_params_  # Best parameters
print("Best Accuracy: {:.2f} %".format(best_accuracy * 100))
print("Best Parameters:", best_parameters)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Kernel SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


