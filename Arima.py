import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
df = pd.read_csv('train.csv', nrows = 3096)
train=df[1:1028]
test=df[1028:1200]
y_hat_avg = test.copy()
fit1 = sm.tsa.statespace.SARIMAX(train.Value, order=(2, 1, 4),seasonal_order=(0,1,1,7)).fit()
y_hat_avg['SARIMA'] = fit1.predict(start=1027, end=1200, dynamic=True)
plt.figure(figsize=(16,8))
plt.plot( train['Value'], label='Train')
plt.plot(test['Value'], label='Test')
plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
plt.legend(loc='best')
plt.show()
print(test['Value'])
print(y_hat_avg['SARIMA'])
from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(test.Value, y_hat_avg.SARIMA))
print(rms)