import pandas as pd
import mpld3
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('data/madrid1.csv')
df['DATE'] = pd.to_datetime(df['DATE'])

df.set_index('DATE', inplace=True)

print(df.head(10))
df.dropna(inplace=True)
train_df = df.loc["2008-01-01":"2012-12-31"]
test_df = df["2013-01-01":]

X_train = train_df[[]]
y_train = train_df['TMAX']
X_test = test_df[[]]
y_test = test_df['TMAX']

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')

plot_df = test_df.copy()
plot_df['Predicted Tmax'] = y_pred

plt.figure(figsize=[7, 7])
plt.plot(plot_df.index, plot_df['TMAX'], label='Actual Tmax', color='blue')
plt.plot(plot_df.index, plot_df['Predicted Tmax'], label='Predicted TMAX', color='red', linestyle='--')
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1)) 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.text(0.95, 0.05, f'MAE: {mae:.2f}\nMSE: {mse:.2f}', 
        fontsize=12, color='darkgreen', horizontalalignment='right', verticalalignment='bottom', 
        transform=plt.gca().transAxes)
plt.title('Actual vs Predicted Maximum Temperature (2023-24)${features}')
plt.xlabel('Date')
plt.ylabel('TMAX')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
mpld3_html = mpld3.fig_to_html(plt.gcf())
filename = output_html
with open(filename, 'w') as f:
    f.write(mpld3_html)

