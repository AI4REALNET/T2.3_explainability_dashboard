from dash import html, dcc
import dash_bootstrap_components as dbc

layout = dbc.Container([
    # Main title
    dbc.Row(
        dbc.Col(
            html.Div(
                "Explainability Panel for Power Grids",
                className="text-primary text-center",
                style={"fontSize": "2.5rem", "fontWeight": "bold", "marginBottom": "20px"}
            )
        )
    ),

    # Panel headers
    dbc.Row(
        [
            dbc.Col(
                html.Div("Statistics Panel", className="text-center h4 text-white"),
                style={"backgroundColor": "#007bff", "padding": "10px", "borderRadius": "8px"}
            ),
            dbc.Col(
                html.Div(
                    [
                        html.Span("Visualization Panel", className="h4 text-white"),
                        html.Span(id="time-stamp", children="  ...", className="ml-3 border p-1", style={"backgroundColor": "white", "color": "#000"})
                    ],
                    className="d-flex justify-content-between align-items-center"
                ),
                style={"backgroundColor": "#dc3545", "padding": "10px", "borderRadius": "8px"}
            )
        ],
        className="mb-3"
    ),

    # Content
    dbc.Row(
        [
            # Statistics panel
            dbc.Col(
                [
                    # Explanation at top
                    html.Div(
                        id="explanation-text",
                        children="Explanation of the agent's recommendation will appear here...",
                        style={
                            "width": "100%",
                            "minHeight": "100px",
                            "border": "1px solid #dee2e6",
                            "borderRadius": "10px",
                            "padding": "10px",
                            "marginBottom": "15px",
                            "backgroundColor": "#f8f9fa",
                            "color": "#333333",
                            "whiteSpace": "pre-line",
                            "overflowY": "auto",
                            "fontSize": "0.95rem"
                        }
                    ),

                    # Environment name
                    dbc.Row(
                        [
                            dbc.Col(html.P("> Environment Name:", className="fs-5")),
                            dbc.Col(html.Div("", id="env-name-output", className="border p-2", style={"fontSize": "0.95rem", "backgroundColor": "#f8f9fa"}))
                        ],
                        className="mb-2"
                    ),
                    html.Hr(),

                    # Distance to expert zone
                    dbc.Row(
                        [
                            dbc.Col(html.P("> Action distance to expert zone:", className="fs-5")),
                            dbc.Col(html.Div("", id="distance-output", className="border p-2", style={"fontSize": "0.95rem", "backgroundColor": "#f8f9fa"}))
                        ],
                        className="mb-2"
                    ),
                    html.Hr(),

                    # Rho improvement
                    dbc.Row(
                        [
                            dbc.Col(html.P("> Rho improvement after agent action:", className="fs-5")),
                            dbc.Col(html.Div("", id="rho-imp-output", className="border p-2", style={"fontSize": "0.95rem", "backgroundColor": "#f8f9fa"}))
                        ],
                        className="mb-2"
                    ),
                    html.Hr(),

                    # Rho difference
                    dbc.Row(
                        [
                            dbc.Col(html.P("> Rho difference between agent & expert:", className="fs-5")),
                            dbc.Col(html.Div("", id="rho-diff-output", className="border p-2", style={"fontSize": "0.95rem", "backgroundColor": "#f8f9fa"}))
                        ],
                        className="mb-2"
                    ),
                    html.Hr(),

                    # Wilcoxon test
                    dbc.Row(
                        [
                            dbc.Col(html.P("> Wilcoxon test statistics:", className="fs-5")),
                            dbc.Col(
                                dbc.Row([
                                    dbc.Col(html.Div("", id="test-stat-output", className="border p-2", style={"fontSize": "0.95rem", "backgroundColor": "#f8f9fa"})),
                                    dbc.Col(html.Div("", id="test-pval-output", className="border p-2", style={"fontSize": "0.95rem", "backgroundColor": "#f8f9fa"})),
                                ])
                            )
                        ]
                    )
                ],
                width=6,
                style={
                    "padding": "15px",
                    "border": "1px solid #dee2e6",
                    "borderRadius": "10px",
                    "backgroundColor": "#ffffff",
                    "height": "520px",
                    "overflowY": "auto"
                }
            ),

            # Visualization panel
            dbc.Col(
                [
                    html.Div(
                        id="graphic-output",
                        style={"width": "100%", "height": "500px", "border": "1px solid #dee2e6", "borderRadius": "10px", "backgroundColor": "#f8f9fa"}
                    )
                    # dbc.Row(
                    #     [
                    #         dbc.Col(html.P("Time stamp:", className="fs-5")),
                    #         dbc.Col(html.Div("...", id="time-stamp", className="border p-2", style={"fontSize": "0.95rem", "backgroundColor": "#f8f9fa"}))
                    #     ],
                    #     className="mt-2"
                    # )
                ],
                width=6,
                style={"padding": "15px", "border": "1px solid #dee2e6", "borderRadius": "10px", "backgroundColor": "#ffffff", "height": "520px"}
            )
        ],
        className="g-4",
        align="start",
        justify="between"
    ),
    dbc.Row(
    [
        dbc.Col(
            dbc.Button(
                "Load Env",
                id="load-button",
                n_clicks=0,
                color="primary",
                size="lg",
                outline=True
            ),
            width="auto"
        ),
        dbc.Col(
            dbc.Button(
                "Recommend Action",
                id="action-button",
                n_clicks=0,
                color="primary",
                size="lg",
                outline=True,
                disabled=True
            ),
            width="auto"
        ),
        dbc.Col(
            dbc.Button(
                "Next overloaded scenario",
                id="next-button",
                n_clicks=0,
                color="primary",
                size="lg",
                outline=True,
                disabled=True
            ),
            width="auto"
        ),
    ],
    justify="center",
    className="mb-3",
    ),
    html.Br(),
    dbc.Row(
        [
            # Agent's action column
            dbc.Col(
                [
                    html.P("Agent's action:", className="fs-3"),
                    html.Div(
                        "Agent's action ...",
                        id="action-text",
                        className="border m-0",
                        style={
                            "width": "100%",
                            "height": 400,
                            "whiteSpace": "pre",
                            "color": "#999999",       # light gray text
                            "padding": "10px",
                            "fontSize": "14px",       # smaller text size
                            "lineHeight": "1.4",      # optional for readability
                            "overflowY": "auto",       # scroll if content exceeds height
                        },
                    ),
                ],
                width=6,
            ),
            # Expert's action column
            dbc.Col(
                [
                    html.P("Expert's action:", className="fs-3"),
                    html.Div(
                        "Expert action ...",
                        id="expert-text",
                        className="border m-0",
                        style={
                            "width": "100%",
                            "height": 400,
                            "whiteSpace": "pre",
                            "color": "#999999",
                            "padding": "10px",
                            "fontSize": "14px",
                            "lineHeight": "1.4",
                            "overflowY": "auto",
                        },
                    ),
                ],
                width=6,
            ),
        ],
        className="g-4",
    ),
    # dbc.Row([
    #     dbc.Col(html.P("Agent's action: ", className="fs-3", style={"width": 200})),
    #     dbc.Col(html.Div("Agent's action ... ", id="action-text", className="border fs3 m-0", style={"width": 600, "height": 400, "whiteSpace": "pre"})),
    #     dbc.Col(html.P("Expert's action: ", className="fs-3", style={"width": 200})),
    #     dbc.Col(html.Div("Expert action ... ", id="expert-text", className="border fs3 m-0", style={"width": 600, "height": 400, "whiteSpace": "pre"}))
    # ]),
    dcc.Store(id="env-loaded", data=False),
    dcc.Store(id="action-taken", data=False),
    dcc.Store(id="action-stats", data=None)
    # dbc.Row([
    #     html.Img(id="overload-graph", style={"height": "600", "width": "50%"}, alt="Overload graph will be shown here ...")
    # ])
], fluid=True)

