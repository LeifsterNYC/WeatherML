import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.figure_factory as ff
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv('data/madrid1.csv')
df['CET'] = pd.to_datetime(df['CET'])

df.set_index('CET', inplace=True)
df[['TMEAN','TMAX', 'TMIN', 'DEWMEAN', 'DEWMIN']] = df[['TMEAN','TMAX', 'TMIN', 'DEWMEAN', 'DEWMIN']].apply(lambda c: (c * 9/5) + 32)
df[['WINDMAX']] = df[['WINDMAX']].apply(lambda km: km * 0.621371)
df['PRCP'] = df['PRCP'].apply(lambda mm: mm * .0393701)
df = df[['TMIN', 'TMAX', 'TMEAN', 'DEWMEAN', 'HUMMEAN', 'MSLPMEAN', 'PRCP', 'VISMEAN', 'WINDMAX', 'WDD', 'CC']]
df.dropna(inplace=True)

fig = go.Figure()
for column in df.columns:
    fig.add_trace(go.Box(y=df[column], name=column))

fig.update_layout(
    title='Box Plots of all Data',
    yaxis_title='Values',
    showlegend=True
)

pio.write_html(fig, 'app/static/graphs/rf1.html', config={'responsive': True})

df = df[df['WINDMAX'] <= 61]


# n_lags = 5
# for var in df.columns:
#     for i in range(1, n_lags+1):
#         df[f'{var}_lag{i}'] = df[var].shift(i)

target = 'TMEAN'
predictors = df.drop(['TMAX', 'TMIN', 'TMEAN'], axis=1).columns



# train = df["2010-01-01":"2014-12-31"]
# test = df["2015-01-01":]

X = df[predictors]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44)

# X_train = train[predictors]
# y_train = train[target]
# X_test = test[predictors]
# y_test = test[target]
trees = 100
model = RandomForestRegressor(n_estimators=trees, random_state=44)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
r2 = r2_score(y_test, y_pred)
print('R2 Score:', r2)

fig = go.Figure()

fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predicted vs Actual',
                         marker=dict(color='LightSkyBlue', opacity=0.7)))
fig.add_trace(go.Line(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], name='Perfect Prediction',
                      line=dict(color='black', dash='dash')))
fig.add_annotation(x=0.95, y=0.05, xref='paper', yref='paper',
                   text=f'MAE: {mae:.2f}<br>MSE: {mse:.2f}<br>R2: {r2:.2f}',
                   showarrow=False, align='right', font_color='darkgreen')
fig.add_annotation(x=0, y=0, xref='paper', yref='paper',
                   text=f'Features:{predictors.to_list()}', showarrow=False,
                   font=dict(size=7, color='darkgreen'),
                   align='left', xanchor='left', yanchor='bottom')
fig.add_annotation(x=0, y=0.02, xref='paper', yref='paper',
                   text=f'Trees: {trees}', showarrow=False,
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
pio.write_html(fig, 'app/static/graphs/rf3.html', config={'responsive': True})

target = 'VISMEAN'
predictors = df.drop(['VISMEAN'], axis=1).columns

X = df[predictors]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44)

# X_train = train[predictors]
# y_train = train[target]
# X_test = test[predictors]
# y_test = test[target]
trees = 100
model2 = RandomForestRegressor(n_estimators=trees, random_state=44)
model2.fit(X_train, y_train)

y_pred = model2.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
r2 = r2_score(y_test, y_pred)
print('R2 Score:', r2)

fig = go.Figure()

fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predicted vs Actual',
                         marker=dict(color='LightSkyBlue', opacity=0.7)))
fig.add_trace(go.Line(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], name='Perfect Prediction',
                      line=dict(color='black', dash='dash')))
fig.add_annotation(x=0.95, y=0.05, xref='paper', yref='paper',
                   text=f'MAE: {mae:.2f}<br>MSE: {mse:.2f}<br>R2: {r2:.2f}',
                   showarrow=False, align='right', font_color='darkgreen')
fig.add_annotation(x=0, y=0, xref='paper', yref='paper',
                   text=f'Features:{predictors.to_list()}', showarrow=False,
                   font=dict(size=7, color='darkgreen'),
                   align='left', xanchor='left', yanchor='bottom')
fig.add_annotation(x=0, y=0.02, xref='paper', yref='paper',
                   text=f'Trees: {trees}', showarrow=False,
                   font=dict(size=7, color='darkgreen'),
                   align='left', xanchor='left', yanchor='bottom')
fig.update_layout(title={
        'text': "Comparison of Predicted and Actual Mean Visibility (mi)",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }, autosize=True,
                  xaxis_title='Actual Mean Visibility (mi)',
                  yaxis_title='Predicted Mean Visibility (mi)',
                  legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
pio.write_html(fig, 'app/static/graphs/rf6.html', config={'responsive': True})
# pred_df = pd.concat([y_test, pd.Series(y_pred, index=X.index)], axis=1)
# pred_df.columns = ['Actual VAR', 'Predicted VAR']

# fig = make_subplots(rows=1, cols=1)
# fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Actual VAR'], mode='lines', name=f'Actual {target}', line=dict(color='blue')))
# fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Predicted VAR'], mode='lines', name=f'Predicted {target}', line=dict(color='red', dash='dash')))
# fig.add_annotation(x=1, y=0, xref='paper', yref='paper',
#                    text=f'MAE: {mae:.2f}<br>MSE: {mse:.2f}', showarrow=False,
#                    font=dict(size=12, color='darkgreen'),
#                    align='right', xanchor='right', yanchor='bottom')
# fig.add_annotation(x=0, y=0, xref='paper', yref='paper',
#                    text=f'Features:{predictors}', showarrow=False,
#                    font=dict(size=7, color='darkgreen'),
#                    align='left', xanchor='left', yanchor='bottom')
# fig.update_layout(title={
#         'text': "Comparison of Predicted and Actual Mean Temperature (F)",
#         'y':0.9,
#         'x':0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'
#     }, autosize=True,
#                   xaxis_title='Actual Mean Temperature (F)',
#                   yaxis_title='Predicted Mean Temperature (F)',
#                   legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
# pio.write_html(fig, 'app/static/graphs/rf2.html', config={'responsive': True})


