import os
import sys
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# make sure your package imports work the same as in the notebook
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.linear_response import response, response_regime_metric

app = Dash(__name__)
server = app.server

def compute_grids(J, g, delta_1, delta_2, a, b, c, density):
    alpha_arr = np.linspace(0.1, 2.0, density)
    beta_arr = np.linspace(0.1, 2.0, density)
    R_l = np.zeros((density, density))
    R_e = np.zeros((density, density))
    Det = np.zeros((density, density))

    for i, alpha in enumerate(alpha_arr):
        for j, beta in enumerate(beta_arr):
            W = np.array([[J, beta * J, - delta_1 * J],
                          [alpha * J, J, - delta_2 * J],
                          [J, J, - g * J]])
            I = np.array([a, b, c])
            Det[i, j] = np.linalg.det(np.eye(3) - W)
            R_l[i, j], R_e[i, j] = response(W, I)

    R_eff = response_regime_metric(R_l, R_e)
    return alpha_arr, beta_arr, Det, R_l, R_e, R_eff

def make_figure(alpha_arr, beta_arr, Det, R_l, R_e, R_eff):
    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=("Determinant(I-W)", "Response: Late Exc (R_L)", 
                       "Response: Early Exc (R_E)", "Response Regimes"),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )

    # Determinant heatmap
    fig.add_trace(go.Heatmap(
        z=Det, x=beta_arr, y=alpha_arr, 
        colorscale='Viridis',
        colorbar=dict(title='Det', x=0.46, len=0.4, y=0.77)
    ), row=1, col=1)
    
    fig.add_trace(go.Contour(
        z=Det, x=beta_arr, y=alpha_arr, 
        contours=dict(start=0, end=0, size=1),
        showscale=False, 
        line=dict(color='white', width=2, dash='dash'), 
        hoverinfo='skip'
    ), row=1, col=1)

    # R_L heatmap + zero contour
    fig.add_trace(go.Heatmap(
        z=R_l, x=beta_arr, y=alpha_arr, 
        colorscale='RdBu', zmid=0,
        colorbar=dict(title='R_L', x=1.02, len=0.4, y=0.77)
    ), row=1, col=2)
    
    fig.add_trace(go.Contour(
        z=R_l, x=beta_arr, y=alpha_arr, 
        contours=dict(start=0, end=0, size=1),
        showscale=False, 
        line=dict(color='black', width=2, dash='dash'), 
        hoverinfo='skip'
    ), row=1, col=2)

    # R_E heatmap + zero contour
    fig.add_trace(go.Heatmap(
        z=R_e, x=beta_arr, y=alpha_arr, 
        colorscale='RdBu', zmid=0,
        colorbar=dict(title='R_E', x=0.46, len=0.4, y=0.23)
    ), row=2, col=1)
    
    fig.add_trace(go.Contour(
        z=R_e, x=beta_arr, y=alpha_arr, 
        contours=dict(start=0, end=0, size=1),
        showscale=False, 
        line=dict(color='black', width=2, dash='dash'), 
        hoverinfo='skip'
    ), row=2, col=1)

    # Response regimes
    fig.add_trace(go.Heatmap(
        z=R_eff, x=beta_arr, y=alpha_arr, 
        colorscale='Portland',
        colorbar=dict(
            title='Regime', 
            x=1.02, len=0.4, y=0.23,
            tickvals=[0, 1, 2],
            ticktext=['Acquisition', 'Early Recall', 'Late Recall']
        )
    ), row=2, col=2)

    # Update axes labels
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(title_text="β (Early Inh)", row=row, col=col)
            fig.update_yaxes(title_text="α (Late Inh)", row=row, col=col)

    fig.update_layout(
        height=800, 
        width=1400,
        title_text="3-Population Linear Model: Differential Inhibition Analysis",
        title_x=0.5,
        title_font_size=18,
        font=dict(size=12),
        margin=dict(t=80, b=40, l=60, r=60)
    )
    
    return fig

def create_slider(id, label, min_val, max_val, step, value):
    return html.Div([
        html.Label(label, style={'fontWeight': 'bold', 'marginBottom': '5px'}),
        dcc.Slider(
            id=id, 
            min=min_val, 
            max=max_val, 
            step=step, 
            value=value,
            marks={min_val: str(min_val), max_val: str(max_val)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={'marginBottom': '20px'})

app.layout = html.Div([
    html.Div([
        html.H1("Differential Inhibition Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P("Interactive exploration of 3-population linear model dynamics",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'})
    ]),
    
    html.Div([
        # Left panel: Connection parameters
        html.Div([
            html.H3("Connection Parameters", style={'color': '#34495e', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
            create_slider('J', 'J (Excitatory strength)', 0.0, 2.0, 0.01, 1.0),
            create_slider('g', 'g (Inhibitory strength)', 0.1, 1.0, 0.01, 0.6),
            create_slider('delta_1', 'δ₁ (Late → Inh)', 0.1, 1.0, 0.01, 0.3),
            create_slider('delta_2', 'δ₂ (Early → Inh)', 0.1, 1.0, 0.01, 0.7),
        ], style={
            'flex': '1', 
            'padding': '20px', 
            'backgroundColor': '#ecf0f1', 
            'borderRadius': '10px',
            'marginRight': '10px'
        }),
        
        # Right panel: Input parameters
        html.Div([
            html.H3("Input Currents", style={'color': '#34495e', 'borderBottom': '2px solid #e74c3c', 'paddingBottom': '10px'}),
            create_slider('a', 'a (Input to Late Exc)', 0.0, 2.0, 0.05, 1.0),
            create_slider('b', 'b (Input to Early Exc)', 0.0, 2.0, 0.05, 1.0),
            create_slider('c', 'c (Input to Inh)', 0.0, 2.0, 0.05, 1.0),
            create_slider('density', 'Grid Density', 40, 300, 10, 150),
        ], style={
            'flex': '1', 
            'padding': '20px', 
            'backgroundColor': '#ecf0f1', 
            'borderRadius': '10px',
            'marginLeft': '10px'
        }),
    ], style={'display': 'flex', 'marginBottom': '20px'}),

    dcc.Loading(
        id="loading",
        type="circle",
        children=[dcc.Graph(id='heatmaps', style={'height': '80vh'})]
    ),
    
    html.Div(id='status', style={
        'textAlign': 'center', 
        'padding': '10px', 
        'color': '#7f8c8d',
        'fontStyle': 'italic'
    })
], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f8f9fa'})


@app.callback(
    Output('heatmaps', 'figure'),
    Output('status', 'children'),
    Input('J', 'value'),
    Input('g', 'value'),
    Input('delta_1', 'value'),
    Input('delta_2', 'value'),
    Input('a', 'value'),
    Input('b', 'value'),
    Input('c', 'value'),
    Input('density', 'value'),
)
def update(J, g, delta_1, delta_2, a, b, c, density):
    density = int(density)
    alpha_arr, beta_arr, Det, R_l, R_e, R_eff = compute_grids(J, g, delta_1, delta_2, a, b, c, density)
    fig = make_figure(alpha_arr, beta_arr, Det, R_l, R_e, R_eff)
    status = f"✓ Computed {density}×{density} grid | Det range: [{Det.min():.3f}, {Det.max():.3f}]"
    return fig, status

if __name__ == "__main__":
    app.run(debug=True, port=8050)