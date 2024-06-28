import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def plotly_scatter_chart(data, x_col, y_col, category_col):
    data[category_col] = data[category_col].astype(str)

    unique_formats = data[category_col].unique()

    format_colors = px.colors.qualitative.Set1 * (len(unique_formats) // len(px.colors.qualitative.Set1) + 1)

    fig = make_subplots(rows=1, cols=1, subplot_titles=[f"{y_col} vs {x_col}"])

    xaxis_range = [data[x_col].min(), data[x_col].max()]
    yaxis_range = [data[y_col].min(), data[y_col].max()]

    traces = []
    buttons = [{'label': 'All',
                'method': 'update',
                'args': [{'visible': [True] * len(unique_formats)},
                         {'title': f"{y_col} vs {x_col} - All"}]}]
    for i, format in enumerate(unique_formats):
        trace = go.Scatter(
            x=data.loc[data[category_col] == format, x_col],
            y=data.loc[data[category_col] == format, y_col],
            mode='markers',
            marker=dict(color=format_colors[i]),
            name=format
        )
        traces.append(trace)

        button = dict(label=format,
                      method="update",
                      args=[{"visible": [format == f for f in unique_formats]},
                            {"title": f"{y_col} vs {x_col} - {format}"}])
        buttons.append(button)

    for trace in traces:
        fig.add_trace(trace)

    fig.update_layout(updatemenus=[{"buttons": buttons,
                                    "direction": "down",
                                    "showactive": True,
                                    "x": 0.15,
                                    "xanchor": "left",
                                    "y": 1.1,
                                    "yanchor": "top"}])

    fig.update_layout(showlegend=False, xaxis_range=xaxis_range, yaxis_range=yaxis_range)

    return fig
