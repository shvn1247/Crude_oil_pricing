import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('train.csv', nrows = 3096)

#Creating train and test set

train=df[1028:]
test=df[1:1028]
from statsmodels.tsa.api import  SimpleExpSmoothing
y_hat_avg = test.copy()
fit2 = SimpleExpSmoothing(np.asarray(train['Value'])).fit(smoothing_level=0.5,optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot(train['Value'], label='Train')
plt.plot(test['Value'], label='Test')
plt.plot(y_hat_avg['SES'], label='SES')
plt.legend(loc='best')
plt.show()
from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(test.Value, y_hat_avg.SES))
print(rms)
