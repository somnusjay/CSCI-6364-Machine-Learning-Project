#  C to M
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression

dir = "D:/Graduate/CSCI 6364 Machine Learning/project/"
data = pd.read_csv(dir + "us_states_covid19_daily.csv")
data = data[['date', 'state', 'positive', 'negative', 'death']]
data.sort_values(['date', 'positive'], axis=0, ascending=False, inplace=True)
values = data.values[:56, 1:5]
print(values)
x = values[:, 1].reshape((-1, 1))
y = values[:, 3].reshape((-1, 1))
lr = LinearRegression()
model = lr.fit(x, y)
y_pred = lr.predict(x)
score = model.score(x, y)
print('score: {:0.3f}'.format(score))
print('RMSE: {:0.3f}'.format(np.sqrt(metrics.mean_squared_error(y, y_pred))))
k = model.intercept_
b = model.coef_

plt.figure()
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.xlim(x.min() * 0.9, x.max() * 1.1)
plt.ylim(y.min() * 0.9, y.max() * 1.1)
plt.title('positive-death')
plt.xlabel('positive')
plt.ylabel('death')
plt.show()