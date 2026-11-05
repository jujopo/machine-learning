import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tools.models.logistic_regression import LogisticRegressionGD
from tools.plotting.decision_regions import decision_regions


iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print("Class labels:", np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

lrgd = LogisticRegressionGD(
        eta=0.3,
        n_iter=1000,
        random_state=1,
        )

lrgd.fit(X_train_01_subset,
         y_train_01_subset)
decision_regions(
    X=X_train_01_subset,
    y=y_train_01_subset,
    classifier=lrgd,
)
plt.title('Classification [Logistic Regression]')
plt.xlabel('Petal lenght [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()