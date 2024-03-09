import pandas as pd
import numpy as np
import mpld3
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.io as pio
import seaborn as sns
import plotly.figure_factory as ff
from plotly.tools import mpl_to_plotly
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

fig.update_xaxes(side="bottom")
pio.write_html(fig, 'app/static/graphs/lr1.html', config={'responsive': True})

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


fig = go.Figure()

fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predicted vs Actual',
                         marker=dict(color='LightSkyBlue', opacity=0.7)))
fig.add_trace(go.Line(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], name='Perfect Prediction',
                      line=dict(color='black', dash='dash')))
fig.add_annotation(x=0.95, y=0.05, xref='paper', yref='paper',
                   text=f'MAE: {mae:.2f}<br>MSE: {mse:.2f}',
                   showarrow=False, align='right', font_color='darkgreen')
fig.update_layout(title={
        'text': "Comparison of Mean and Actual Predicted Temperature",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }, autosize=True,
                  xaxis_title='Actual Mean Temperature (F)',
                  yaxis_title='Predicted Mean Temperature (F)',
                  legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
pio.write_html(fig, 'app/static/graphs/lr2.html', config={'responsive': True})


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

fig = go.Figure()

fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predicted vs Actual',
                         marker=dict(color='LightSkyBlue', opacity=0.7)))
fig.add_trace(go.Line(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], name='Perfect Prediction',
                      line=dict(color='black', dash='dash')))
fig.add_annotation(x=0.95, y=0.05, xref='paper', yref='paper',
                   text=f'MAE: {mae:.2f}<br>MSE: {mse:.2f}',
                   showarrow=False, align='right', font_color='darkgreen')
fig.update_layout(title={
        'text': "Comparison of Mean and Actual Sea Level Pressure (mb)",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }, autosize=True,
                  xaxis_title='Actual Mean Sea Level Pressure (mb)',
                  yaxis_title='Predicted Mean Sea Level Pressure (mb)',
                  legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
pio.write_html(fig, 'app/static/graphs/lr4.html', config={'responsive': True})