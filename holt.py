import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('train.csv', nrows = 3096)

#Creating train and test set

train=df[1028:]
test=df[1:1028]

from statsmodels.tsa.api import Holt

import statsmodels.api as sm
sm.tsa.seasonal_decompose(train.Value, freq=3).plot()
result = sm.tsa.stattools.adfuller(train.Value)
plt.show()

y_hat_avg = test.copy()

fit1 = Holt(np.asarray(train['Value'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
y_hat_avg['Holt_linear'] = fit1.forecast(len(test))

plt.figure(figsize=(16,8))
plt.plot(train['Value'], label='Train')
plt.plot(test['Value'], label='Test')
plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
plt.legend(loc='best')
plt.show()

from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(test.Value, y_hat_avg.Holt_linear))
print(rms)