import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from perceptron import Perceptron

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # Set up marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the decision surface.
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)

    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot class examples.
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    label=f"Class {cl}",
                    edgecolor='black',
                )

s = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
print("From URL: ", s)

df = pd.read_csv(s,
        header=None,
        encoding="utf-8")
# print(df.tail())

# Select setosa and versicolor.
y = df.iloc[:100, 4].values
y = np.where(y == "Iris-setosa", 0, 1)
# Extract sepal length and petal length.
X = df.iloc[:100, [0, 2]].values
# Plot data
# plt.scatter(X[:50, 0], X[:50, 1],
#         color='#5A4FCF', marker='o', label='Setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1],
#             color='#6A5ACD', marker='s', label='Versicolor')
# plt.xlabel('Sepal length [cm]')
# plt.ylabel('Petal length [cm]')
# plt.legend(loc='upper left')
# plt.show()

pnn = Perceptron(eta=0.1, n_iter=10)
pnn.fit(X, y)

# Plot the error convergence.
# plt.plot(range(1, len(pnn.errors_) + 1),
#         pnn.errors_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Number of updates')
# plt.show()

plot_decision_regions(X, y, classifier=pnn)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()