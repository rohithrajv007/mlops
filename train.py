from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
clf= LogisticRegression(random_state=0, max_iter=200)
clf.fit(X, y)
print("Model trained successfully.")
