import plotly.io as pio
import plotly.graph_objects as go


data = go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13])


fig = go.Figure(data=[data])
fig.update_layout(
    title="My Responsive Plot",
    autosize=True  
)

pio.write_html(fig, 'lr2.html', config={'responsive': True})
