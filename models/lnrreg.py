import pandas as pd
import mpld3
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

def analyze_temperature_data(csv_file, num_days, output_html, lag=1, window=None):
    df = pd.read_csv(csv_file)
    df['DATE'] = pd.to_datetime(df['DATE'])

    df.set_index('DATE', inplace=True)
    lag_str = str(lag)
    Tmax_lag_name = 'Tmax_lag' + lag_str
    Tmin_lag_name = 'Tmin_lag' + lag_str
    df[Tmax_lag_name] = df['TMAX'].shift(lag)
    df[Tmin_lag_name] = df['TMIN'].shift(lag)

    if window:
        window_int = int(window)
        Tmax_val = f'Tmax_rolling{window}'
        Tmin_val = f'Tmin_rolling{window}'
        df[Tmax_val] = df['TMAX'].rolling(window=window_int).mean()
        df[Tmin_val] = df['TMIN'].rolling(window=window_int).mean()
    else:
        Tmax_val = Tmax_lag_name
        Tmin_val = Tmin_lag_name

    df.dropna(inplace=True)
    print(df.head(10))
    train_df = df[:-num_days]
    test_df = df[-num_days:]

    X_train = train_df[[Tmax_val]]
    y_train = train_df['TMAX']
    X_test = test_df[[Tmax_val]]
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
    plt.title('Actual vs Predicted Maximum Temperature')
    plt.xlabel('Date')
    plt.ylabel('TMAX')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    mpld3_html = mpld3.fig_to_html(plt.gcf())
    filename = output_html
    with open(filename, 'w') as f:
        f.write(mpld3_html)

analyze_temperature_data('data/data2.csv', num_days=60, output_html='app/static/html/graphs/g1.html', lag=1)
