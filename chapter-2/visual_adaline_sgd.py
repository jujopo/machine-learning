import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tools.models.adaline import AdalineSGD
from tools.plotting.decision_regions import decision_regions

s = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
print("From URL: ", s)

df = pd.read_csv(s,
        header=None,
        encoding="utf-8")

# Select setosa and versicolor.c
y = df.iloc[:100, 4].values
y = np.where(y == "Iris-setosa", 0, 1)
# Extract sepal length and petal length.
X = df.iloc[:100, [0, 2]].values

# Standardize the features
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada_sgd.fit(X_std, y)
decision_regions(X=X_std, y=y, classifier=ada_sgd)
plt.title('Adaline - Stochastic gradient descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada_sgd.losses_) + 1),
         ada_sgd.losses_, marker='o')
plt.title("Loss convergence")
plt.xlabel('Epochs')
plt.ylabel('Average loss')
plt.tight_layout()
plt.show()