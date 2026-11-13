import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from tools.plotting.decision_regions import decision_regions

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

lr = LogisticRegression(C=100.0, solver='lbfgs')
lr.fit(X_train_std, y_train)
decision_regions(X_combined_std,
                 y_combined,
                 classifier=lr,
                 test_idx=range(105, 150),
                )
plt.title('Classification [Logistic Regression - Scikit]')
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# This gives us the probability of each class for a given example(s)
print(lr.predict_proba(X_test_std[:3, :]))
# If we want to know the predicted class instead, we use this:
print(lr.predict(X_test_std[:3, :]))