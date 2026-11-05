import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tools.models.adaline import AdalineGD
from tools.plotting.decision_regions import decision_regions

s = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
print("From URL: ", s)

df = pd.read_csv(s,
        header=None,
        encoding="utf-8")

# Select setosa and versicolor.
y = df.iloc[:100, 4].values
y = np.where(y == "Iris-setosa", 0, 1)
# Extract sepal length and petal length.
X = df.iloc[:100, [0, 2]].values

# Graph using learning rate of 0.1
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ada1 = AdalineGD(n_iter=15, eta=0.1).fit(X, y)
ax[0].plot(range(1, len(ada1.losses_) + 1),
           np.log10(ada1.losses_), marker='o', color='red')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('$log(MSE)$')
ax[0].set_title('Adaline - Learning rate 0.1')

# Graph using learning rate of 0.0001
ada2 = AdalineGD(n_iter=15, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.losses_) + 1),
           ada2.losses_, marker='o', color='blue')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('MSE')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada_gd = AdalineGD(n_iter=20, eta=0.5)
ada_gd.fit(X_std, y)
decision_regions(X=X_std, y=y, classifier=ada_gd)
plt.title('Adaline - Gradient descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Plot loss convergence.
plt.plot(range(1, len(ada_gd.losses_) + 1),
         ada_gd.losses_, marker='o')
plt.title("Loss convergence")
plt.xlabel('Epochs')
plt.ylabel('Mean squared error')
plt.tight_layout()
plt.show()