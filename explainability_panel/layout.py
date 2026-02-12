from dash import html
import dash_bootstrap_components as dbc

layout = dbc.Container([
    dbc.Row([
        html.Div("Explainability Panel for Power Grids", className="text-primary text-center h1")
    ]),
    dbc.Row([
            dbc.Col(html.Div("Statistics panel"), className="bg-primary text-center h2", style={"color": "white"}),
            dbc.Col(html.Div("Visualization panel"), className="bg-danger text-center h2", style={"color": "white"})
        ]),
    dbc.Row(
        [
            dbc.Col([
                dbc.Row([
                    dbc.Col(html.P("> Environment Name: ", className="fs-4", style={"width": 450})),
                    dbc.Col(html.Div("", id="env-name-output", className="fs-4 border", style={"width": 400})),
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Col(html.P("> Action distance to the suggested zone by the expert: ", className="fs-4", style={"width": 550})),
                    dbc.Col(html.Div("", id="distance-output", className="fs-4 border", style={"width": 130})),
                ]), 
                html.Hr(),
                dbc.Row([
                    dbc.Col(html.P("> Rho improvement after agent action: ", className="fs-4", style={"width": 550})),
                    dbc.Col(html.Div("", id="rho-imp-output", className="fs-4 border", style={"width": 130})),
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Col(html.P("> Rho difference between the agent action and the one suggested by the expert: ", className="fs-4", style={"width": 550})),
                    dbc.Col(html.Div("", id="rho-diff-output", className="fs-4 border", style={"width": 130})),
                ]),    
                html.Hr(),
                dbc.Row([
                    dbc.Col(html.P("> Wilcoxon test between agent output distribution and the ranked greedy actions:", className="fs-4")),
                    dbc.Row([
                        dbc.Col(html.Div("", id="test-stat-output", className="fs-4 border", style={"width": 300})),
                        dbc.Col(html.Div("", id="test-pval-output", className="fs-4 border", style={"width": 300}))    
                    ])
                    
                ])    
                   
            ],
            style={"width": "47%", "height": 600}),#, className="bg-light"),
            # dbc.Col(dcc.Graph(figure={}, id='graphic-output'), style={"width": 550, "height": 400})
            dbc.Col([
                html.Div(html.Div(id="graphic-output"), style={"width": "47%", "height": 600}, className="text-center"),
                dbc.Row([
                    dbc.Col(html.P("Time stamp")),
                    dbc.Col(html.P("...", id="time-stamp", className="border"))
                    ]),
            ]),
            # dbc.Placeholder(dcc.Graph(figure={}, id='loading-stats'), color="dark", style={"width": "47%", "height": 400}, animation="wave"),
            # dbc.Placeholder(dcc.Graph(figure={}, id='loading-graphic'), color="dark", style={"width": "47%", "height": 400}, animation="wave")
        ],
        align="center",
        justify="evenly"
    ),

    html.Div(
        [dbc.Button("Load Env", color="primary", id="load-button", n_clicks=0, className="me-1", size="lg", outline=True),
         dbc.Button("Recommend Action", color="primary", id="action-button", n_clicks=0, className="me-1", size="lg", outline=True, disabled=True),
         dbc.Button("Next scenario", color="primary", id="next-button", n_clicks=0, className="me-1", size="lg", outline=True, disabled=True)]
    ),
    html.Br(),
    dbc.Row([
        dbc.Col(html.P("Explanation: ", className="fs-3", style={"width": 200})),
        dbc.Col(html.P("Explanation of the agent recommendation ... ", id="explanation-text", className="border fs3 m-0", style={"width": 1000, "height": 150}))
    ]),
    dbc.Row([
        dbc.Col(html.P("Action description: ", className="fs-3", style={"width": 200})),
        dbc.Col(html.Div("Agent's action ... ", id="action-text", className="border fs3 m-0", style={"width": 1000, "height": 400, "whiteSpace": "pre"}))
    ]),
    # dbc.Row([
    #     html.Img(id="overload-graph", style={"height": "600", "width": "50%"}, alt="Overload graph will be shown here ...")
    # ])
], fluid=True)

