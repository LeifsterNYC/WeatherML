import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv('data/madrid1.csv')
df['CET'] = pd.to_datetime(df['CET'])
df.drop(['GUSTMAX', 'CC', 'Events', 'Dew Point', 'VISMIN', 'VISMAX', 'VISMEAN'], axis=1, inplace=True)
df.set_index('CET', inplace=True)
df[['TMEAN','TMAX', 'TMIN', 'DEWMEAN', 'DEWMIN']] = df[['TMEAN','TMAX', 'TMIN', 'DEWMEAN', 'DEWMIN']].apply(lambda c: (c * 9/5) + 32)
df[['WINDMAX']] = df[['WINDMAX']].apply(lambda km: km * 0.621371)
df['PRCP'] = df['PRCP'].apply(lambda mm: mm * .0393701)
df = df[df['WINDMAX'] <= 61]
df.dropna(inplace=True)
df = df["2004-01-01":]

print(tf.__version__)

features = df.columns # Includes target
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=features, index=df.index)

def predict(target):
    def create_sequences(df, past_days, predict_days):
        X, y = [], []
        for i in range(len(df) - past_days - predict_days + 1):
            X.append(df.iloc[i:(i + past_days)].to_numpy())
            y.append(df.iloc[i + past_days:i + past_days + predict_days][target].to_numpy())
        return np.array(X), np.array(y)

    X, y = create_sequences(df_scaled, 10, 1)
    data_length = len(X)
    split= int(data_length * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(tf.keras.layers.LSTM(50, activation='tanh'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
    y_pred = model.predict(X_test)
    target_index = df.columns.get_loc(target)
    y_test_unscaled = np.zeros((len(y_test), df_scaled.shape[1])) 
    y_pred_unscaled = np.zeros((len(y_pred), df_scaled.shape[1]))  
    y_test_unscaled[:, target_index] = y_test.ravel() 
    y_pred_unscaled[:, target_index] = y_pred.ravel() 
    y_test_unscaled = scaler.inverse_transform(y_test_unscaled)[:, target_index] 
    y_pred_unscaled = scaler.inverse_transform(y_pred_unscaled)[:, target_index]
    fig = go.Figure()

    mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
    mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[split:], y=y_test_unscaled, mode='lines', name=f'Actual {target}', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index[split:], y=y_pred_unscaled, mode='lines', name=f'Predicted {target}', line=dict(color='red', dash='dash')))
    fig.add_annotation(x=1, y=0, xref='paper', yref='paper',
                   text=f'MAE: {mae:.2f}<br>MSE: {mse:.2f}', showarrow=False,
                   font=dict(size=12, color='darkgreen'),
                   align='right', xanchor='right', yanchor='bottom')
    features_list = df.drop([target], axis=1).columns.tolist()  # 
    first_line_features = ", ".join(features_list[:7]) 
    second_line_features = ", ".join(features_list[7:]) 
    fig.add_annotation(x=0, y=0, xref='paper', yref='paper',
                   text=f'Features:<br>{first_line_features}<br>{second_line_features}',
                   showarrow=False, font=dict(size=7, color='darkgreen'),
                   align='left', xanchor='left', yanchor='bottom')
    fig.add_annotation(x=0, y=0.05, xref='paper', yref='paper',
                   text=f'Epochs: 20', showarrow=False,
                   font=dict(size=7, color='darkgreen'),
                   align='left', xanchor='left', yanchor='bottom')
    fig.update_layout(title=f'Actual vs Predicted {target}', xaxis_title='Date', yaxis_title=f'{target}')
    pio.write_html(fig, 'app/static/graphs/deep5.html', config={'responsive': True})

predict('WDD')

