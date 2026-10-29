import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

from iris_classification import plot_decision_regions

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print('Class labels:', np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y,
)

# print("Label counts in y:", np.bincount(y))
# print("Label counts in y_train:", np.bincount(y_train))
# print("Label counts in y_test:", np.bincount(y_test))

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

pnn = Perceptron(eta0=0.1, random_state=1)
pnn.fit(X_train_std, y_train)

y_pred = pnn.predict(X_test_std)
print("Misclassified examples: %d" % (y_test != y_pred).sum())
print("Accuracy: %.3f" % accuracy_score(y_test, y_pred))
# Another way using the score method which combines predict 
# call with accuracy score
print(f"Accuracy: {pnn.score(X_test_std, y_test):.3f}")

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std,
                      y=y_combined,
                      classifier=pnn,
                      test_idx=range(105, 150),
                    )

plt.xlabel("Petal length [standardized]")
plt.ylabel("Petal width [standardized]")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()