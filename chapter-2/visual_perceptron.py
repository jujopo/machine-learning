import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tools.models.perceptron import Perceptron
from tools.plotting.decision_regions import decision_regions

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
plt.scatter(X[:50, 0], X[:50, 1],
        color='#5A4FCF', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='#6A5ACD', marker='s', label='Versicolor')
plt.title('Iris Data')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()

pnn = Perceptron(eta=0.1, n_iter=10)
algo = 'Perceptron'
pnn.fit(X, y)

decision_regions(X, y, classifier=pnn)
plt.title(f"Iris Classification [{algo}]")
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()

# Plot the error convergence.
plt.plot(range(1, len(pnn.errors_) + 1),
        pnn.errors_, marker='o')
plt.title(f'Error convergence [{algo}]')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()
    