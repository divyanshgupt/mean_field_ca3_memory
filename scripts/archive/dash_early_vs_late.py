import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# ----------------------
# Generate dummy simulation data
# Replace this with your real results
# ----------------------
g_values = [0.1, 0.5, 1.0]
J_values = [1, 2, 3]
alpha_values = [0.01, 0.05, 0.1]

time = np.linspace(0, 10, 200)

# Make a DataFrame with all results
data = []
for g in g_values:
    for J in J_values:
        for alpha in alpha_values:
            # Dummy dynamics: state1 is oscillatory, state2 is exponential decay
            state1 = np.sin(time * g) * np.exp(-alpha * time) + J * 0.1
            state2 = np.cos(time * alpha * 5) * np.exp(-0.1 * time) + g
            data.append(pd.DataFrame({
                "time": time,
                "state1": state1,
                "state2": state2,
                "g": g,
                "J": J,
                "alpha": alpha
            }))
df = pd.concat(data)

# ----------------------
# Build Dash app
# ----------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Grid Search Simulation Viewer"),

    html.Div([
        html.Label("Select g:"),
        dcc.Dropdown(
            id="dropdown-g",
            options=[{"label": str(val), "value": val} for val in g_values],
            value=g_values[0]
        ),
    ], style={"width": "30%", "display": "inline-block"}),

    html.Div([
        html.Label("Select J:"),
        dcc.Dropdown(
            id="dropdown-J",
            options=[{"label": str(val), "value": val} for val in J_values],
            value=J_values[0]
        ),
    ], style={"width": "30%", "display": "inline-block"}),

    html.Div([
        html.Label("Select alpha:"),
        dcc.Dropdown(
            id="dropdown-alpha",
            options=[{"label": str(val), "value": val} for val in alpha_values],
            value=alpha_values[0]
        ),
    ], style={"width": "30%", "display": "inline-block"}),

    dcc.Graph(id="timeseries-plot")
])

# ----------------------
# Callbacks
# ----------------------
@app.callback(
    Output("timeseries-plot", "figure"),
    Input("dropdown-g", "value"),
    Input("dropdown-J", "value"),
    Input("dropdown-alpha", "value")
)
def update_plot(g, J, alpha):
    # Filter dataframe
    dff = df[(df.g == g) & (df.J == J) & (df.alpha == alpha)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dff["time"], y=dff["state1"],
                             mode="lines", name="State 1"))
    fig.add_trace(go.Scatter(x=dff["time"], y=dff["state2"],
                             mode="lines", name="State 2"))

    fig.update_layout(
        title=f"Simulation Results (g={g}, J={J}, α={alpha})",
        xaxis_title="Time",
        yaxis_title="State Variables"
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True)