import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.figure_factory as ff
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('data/madrid1.csv')
df['CET'] = pd.to_datetime(df['CET'])
df.drop(['GUSTMAX', 'CC', 'Events', 'Dew Point', 'VISMIN', 'VISMAX', 'VISMEAN'], axis=1, inplace=True)

df.set_index('CET', inplace=True)

df[['TMEAN','TMAX', 'TMIN', 'DEWMEAN', 'DEWMIN']] = df[['TMEAN','TMAX', 'TMIN', 'DEWMEAN', 'DEWMIN']].apply(lambda c: (c * 9/5) + 32)
df[['WINDMAX']] = df[['WINDMAX']].apply(lambda km: km * 0.621371)
df['PRCP'] = df['PRCP'].apply(lambda mm: mm * .0393701)

#df['TMAX_lead1'] = df['TMAX'].shift(-1)
df['TMAX_lag1'] = df['TMAX'].shift(1)
df['TMAX_lag2'] = df['TMAX'].shift(2)
df['TMAX_rolling4'] = df['TMAX'].shift(1).rolling(window=4).mean()
df['TMIN_rolling4'] = df['TMIN'].shift(1).rolling(window=4).mean()
df['TMEAN_rolling4'] = df['TMEAN'].shift(1).rolling(window=4).mean()
df['MSLPMEAN_rolling4'] = df['MSLPMEAN'].shift(1).rolling(window=4).mean()
df['DEWMEAN_rolling4'] = df['DEWMEAN'].shift(1).rolling(window=4).mean()
df['HUMMEAN_rolling4'] = df['HUMMEAN'].shift(1).rolling(window=4).mean()
df['TMAX_rolling2'] = df['TMAX'].shift(1).rolling(window=2).mean()
df['TMIN_rolling2'] = df['TMIN'].shift(1).rolling(window=2).mean()
df['DEWMEAN_lag1'] = df['DEWMEAN'].shift(1)
df['TMEAN_lag1'] = df['TMEAN'].shift(1)
df['HUMMEAN_lag1'] = df['HUMMEAN'].shift(1)
df['TMIN_lag1'] = df['TMIN'].shift(1)
df['PRCP_lag1'] = df['PRCP'].shift(1)
df['MSLPMEAN_lag1'] = df['MSLPMEAN'].shift(1)
df.dropna(inplace=True)

corr_select = df[['TMEAN', 'TMEAN_lag1', 'TMAX', 'TMAX_lag1', 'TMIN', 'TMIN_lag1', 'HUMMEAN', 'HUMMEAN_lag1', 'MSLPMEAN', 'MSLPMEAN_lag1', 'PRCP', 'PRCP_lag1']]

corr = corr_select.corr()
fig = ff.create_annotated_heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.index.tolist(),
    colorscale='bluered',
    annotation_text=corr.round(2).values,
    showscale=True,  
)

fig.update_layout(
    title='Correlation Matrix',
    xaxis=dict(tickangle=-45),  
    margin=dict(l=50, r=50, t=50, b=50), 
    autosize=True,
    font=dict(
        size = 9
    )
)
pio.write_html(fig, 'app/static/graphs/lrts1.html', config={'responsive': True})

predictors = predictors = ['TMAX_rolling4', 'TMIN_rolling4', 'MSLPMEAN_rolling4', 'HUMMEAN_rolling4', 'DEWMEAN_rolling4', 'TMEAN_rolling4']

target = 'TMEAN'


train = df["2010-01-01":"2014-12-31"]
test = df["2015-01-01":]

X_train = train[predictors]
y_train = train[target]
X_test = test[predictors]
y_test = test[target]

fig = go.Figure()

fig.add_trace(go.Scatter(x=train.index, y=train['TMEAN'], name='Train', mode='lines', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=test.index, y=test['TMEAN'], name='Test', mode='lines', line=dict(color='orange')))

fig.update_layout(
    title='Training and Testing Data Visualization',
    xaxis_title='Date',
    yaxis_title='Temperature (F)',
    legend_title='Data Type'
)

model = LinearRegression()
model.fit(X_train, y_train)

intercept = model.intercept_
coefficients = model.coef_
formula = f'y = {intercept:.2f}'
for i, coef in enumerate(coefficients):
    formula += f' + ({coef:.2f}) * x{i+1}'
# print(f"Linear Regression Formula:\n{formula}")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

pred_df = pd.concat([y_test, pd.Series(y_pred, index=test.index)], axis=1)
pred_df.columns = ['Actual VAR', 'Predicted VAR']

fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Actual VAR'], mode='lines', name=f'Actual {target}', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Predicted VAR'], mode='lines', name=f'Predicted {target}', line=dict(color='red', dash='dash')))
fig.add_annotation(x=1, y=0, xref='paper', yref='paper',
                   text=f'MAE: {mae:.2f}<br>MSE: {mse:.2f}', showarrow=False,
                   font=dict(size=12, color='darkgreen'),
                   align='right', xanchor='right', yanchor='bottom')
fig.add_annotation(x=0, y=0, xref='paper', yref='paper',
                   text=f'Features:{predictors}', showarrow=False,
                   font=dict(size=7, color='darkgreen'),
                   align='left', xanchor='left', yanchor='bottom')
fig.update_layout(title={
        'text': "Comparison of Predicted and Actual Mean Temperature (F)",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }, autosize=True,
                  xaxis_title='Actual Mean Temperature (F)',
                  yaxis_title='Predicted Mean Temperature (F)',
                  legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
pio.write_html(fig, 'app/static/graphs/lrts5.html', config={'responsive': True})
