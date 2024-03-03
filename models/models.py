import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('ithaca_2010_2024_tminmax.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Days_Since_Start'] = (df['Date'] - df['Date'].min()).dt.days
df['Tmax_lag1'] = df['Tmax'].shift(1)
df['Tmin_lag1'] = df['Tmin'].shift(1) 
df['Tmax_rolling7'] = df['Tmax'].rolling(window=7).mean()
df['Tmin_rolling7']  = df['Tmin'].rolling(window=7).mean()
df.dropna(inplace=True)

X = df[['Days_Since_Start', 'Tmax_lag1','Tmin_lag1', 'Tmax_rolling7', 'Tmin_rolling7' ]]
# X = df[['Days_Since_Start', 'Tmax_lag1','Tmin_lag1']]
y = df['Tmax']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')

df.set_index('Date', inplace=True)
df.sort_values(by='Date', inplace=True)
pred_date_str = '2021-06-02'
pred_date = pd.to_datetime(pred_date_str)
formatted_date = pred_date.strftime('%B %dth, %Y')

# X_pred = pd.DataFrame({
#     'Days_Since_Start': [df.loc[pred_date, 'Days_Since_Start']],
#     'Tmax_lag1': [df.loc[pred_date, 'Tmax_lag1']],
#     'Tmin_lag1': [df.loc[pred_date, 'Tmin_lag1']]
# })

X_pred = pd.DataFrame({
    'Days_Since_Start': [df.loc[pred_date, 'Days_Since_Start']],
    'Tmax_lag1': [df.loc[pred_date, 'Tmax_lag1']],
    'Tmin_lag1': [df.loc[pred_date, 'Tmin_lag1']],
    'Tmax_rolling7': [df.loc[pred_date, 'Tmax_rolling7']],
    'Tmin_rolling7': [df.loc[pred_date, 'Tmin_rolling7']],
    
})

predicted_TMAX = model.predict(X_pred)
print(f'The predicted TMAX for {formatted_date} is: {predicted_TMAX[0]}')