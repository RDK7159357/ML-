import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

inputs = np.array([[1],[2],[3],[4],[5]])
outputs = np.array([7,14,21,28,35])
alg =SVC()
alg.fit(inputs, outputs)
model = LinearRegression()
model.fit(inputs, outputs)
result1 = model.predict(np.array([[69],[420],[49]]))
result2 = alg.predict(np.array([[20],[300],[180]]))
print(result1) 
print(result2)
