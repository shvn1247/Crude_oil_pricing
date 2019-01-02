import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('train.csv', nrows = 3096)

#Creating train and test set

train=df[1028:]
test=df[1:1028]

y_hat_avg = test.copy()
y_hat_avg['avg_forecast'] = train['Value'].mean()
plt.figure(figsize=(12,8))
plt.plot(train['Value'], label='Train')
plt.plot(test['Value'], label='Test')
plt.plot(y_hat_avg['avg_forecast'], label='Average Forecast')
plt.legend(loc='best')
plt.show()

from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(test.Value, y_hat_avg.avg_forecast))
print(rms)
