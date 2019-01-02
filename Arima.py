import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('train.csv', nrows = 3096)

#Creating train and test set

train=df[1028:]
test=df[1:1028]

import statsmodels.api as sm
y_hat_avg = test.copy()
fit1 = sm.tsa.statespace.SARIMAX(train.Value, order=(2, 1, 4),seasonal_order=(0,1,1,7)).fit()
y_hat_avg['SARIMA'] = fit1.predict(start="2015-01-02", end="2003-01-02", dynamic=True)
plt.figure(figsize=(16,8))
plt.plot( train['Value'], label='Train')
plt.plot(test['Value'], label='Test')
plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
plt.legend(loc='best')
plt.show()