import pandas as pd
import numpy as np
import mpld3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('data/madrid1.csv') 
df = df[df['Max Wind Speed'] <= 100] # Remove wind speeds higher than 100 km/h, as analyzed as outliers by box plot
df[['Mean Temperature','Max Temperature', 'Min Temperature', 'Mean Dewpoint']] = df[['Mean Temperature','Max Temperature', 'Min Temperature', 'Mean Dewpoint']].apply(lambda c: (c * 9/5) + 32)
df[['Mean Visibility', 'Max Wind Speed']] = df[['Mean Visibility', 'Max Wind Speed']].apply(lambda km: km * 0.621371)
df['Precipitation'] = df['Precipitation'].apply(lambda mm: mm * .0393701)
df_cleaned = df[['Min Temperature', 'Max Temperature', 'Mean Temperature', 'Mean Dewpoint', 'Mean Humidity', 'Mean Sea Level Pressure', 'Precipitation', 'Mean Visibility', 'Max Wind Speed', 'WindDirDegrees', 'CloudCover']]
df_cleaned.dropna(inplace=True) # Drop any NaN values from the remaining dataset
corr = df_cleaned.corr()
plt.figure(figsize=(7, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.tight_layout()
plt.close()

X = df_cleaned.drop(['Mean Temperature', 'Max Temperature', 'Min Temperature'], axis = 1)
y = df_cleaned['Mean Temperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)


plt.figure(figsize=(7,7)) 
plt.scatter(y_test, y_pred, alpha=0.5, label='Predicted vs Actual')
plt.xlabel('Actual Mean Temperature (F)')  
plt.ylabel('Predicted Mean Temperature (F)')  
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')
plt.text(0.95, 0.05, f'MAE: {mae:.2f}\nMSE: {mse:.2f}',
         fontsize=12, color='darkgreen', horizontalalignment='right', verticalalignment='bottom', 
         transform=plt.gca().transAxes)
plt.title('Comparison of Actual and Predicted Mean Temperatures')
plt.legend()
plt.tight_layout()
# mpld3_html = mpld3.fig_to_html(plt.gcf())
# filename = 'app/static/graphs/lr2.html'
# with open(filename, 'w') as f:
#     f.write(mpld3_html)
plt.close()

X = df_cleaned.drop(['Mean Sea Level Pressure'], axis = 1)
y = df_cleaned['Mean Sea Level Pressure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

model2 = LinearRegression()
model2.fit(X_train, y_train)

y_pred = model2.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

plt.figure(figsize=(7,7)) 
plt.scatter(y_test, y_pred, alpha=0.5, label='Predicted vs Actual')
plt.xlabel('Actual Mean Sea Level Pressure (mb)')  
plt.ylabel('Predicted Mean Sea Level Pressure (mb)')  
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')
plt.text(0.95, 0.05, f'MAE: {mae:.2f}\nMSE: {mse:.2f}',
         fontsize=12, color='darkgreen', horizontalalignment='right', verticalalignment='bottom', 
         transform=plt.gca().transAxes)
plt.title('Comparison of Actual and Predicted Sea Level Pressure')
plt.legend()
plt.tight_layout()
mpld3_html = mpld3.fig_to_html(plt.gcf())
filename = 'app/static/graphs/lr4.html'
with open(filename, 'w') as f:
    f.write(mpld3_html)
plt.close()

