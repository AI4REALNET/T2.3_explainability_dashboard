########################################################################################################
#                                       Load button callbacks
########################################################################################################
import time
from dash import callback, Output, Input, State
from services import my_agent, my_env
from dash import dcc

@callback(
    # Output("graphic-output", "figure", allow_duplicate=True),
    Output("graphic-output", "children", allow_duplicate=True),
    Output("next-button", "disabled"),
    Output("action-button", "disabled"),
    # Output("overload-graph", "children", allow_duplicate=True),
    # Output("overload-graph", "src", allow_duplicate=True),
    [Input("load-button", "n_clicks")],
    running=[
        (Output("load-button", "disabled"), True, False),
        (Output("next-button", "disabled"), True, False),
        (Output("action-button", "disabled"), True, False)
    ],
    prevent_initial_call=True
)
def load_output(n):
    if n:
        env, _ = my_env.reset()
        my_agent.load(env)
        return dcc.Graph(figure=my_env.plot_overload_graph()), False, False
        # return  dcc.Graph(figure=my_env.plot_obs())
        # svg = my_env.plot_overload_graph()
        # cairosvg.svg2png(bytestring=svg, write_to="output.png")
        # time.sleep(3)
        # return "/assets/output.png"

@callback(
    Output("time-stamp", "children", allow_duplicate=True),
    Input("load-button", "n_clicks"),
    Input("graphic-output", "children"),
    prevent_initial_call=True
)
def load_time_stamp(n, graphic_output):
    if n and graphic_output:
        return my_env.get_time_stamp()


@callback(
    Output("env-name-output", "children", allow_duplicate=True),
    [Input("graphic-output", "children")],
    prevent_initial_call=True
)
def load_env_name(n):
    if my_env.env is not None:
        return my_env.env.env_name

# @callback(
#     Output("next-button", "disabled"),
#     Output("action-button", "disabled"),
#     Input("load-button", "n_clicks"),
#     running=[
#         (Output("next-button", "disabled"), True, False),
#         (Output("load-button", "disabled"), True, False),
#         (Output("start-button", "disabled"), True, False),
#     ],
#     prevent_initial_call=True
# )
# def enable_buttons(n_clicks):
#     if n_clicks:
#         return False, False
################################
# Statistics
################################
@callback(
    Output("distance-output", "children", allow_duplicate=True),
    [Input("load-button", "n_clicks")],
    prevent_initial_call=True
)
def load_distance_reset(n):
    Information = ""
    if n:
        return Information
    
@callback(
    Output("rho-imp-output", "children", allow_duplicate=True),
    [Input("load-button", "n_clicks")],
    prevent_initial_call=True
)
def load_rho_imrpovement_reset(n):
    Information = ""
    if n:
        return Information
    
@callback(
    Output("rho-diff-output", "children", allow_duplicate=True),
    [Input("load-button", "n_clicks")],
    prevent_initial_call=True
)
def load_rho_difference_reset(n):
    Information = ""
    if n:
        return Information
    
@callback(
    Output("test-stat-output", "children", allow_duplicate=True),
    [Input("load-button", "n_clicks")],
    prevent_initial_call=True
)
def load_test_wilcoxon_reset(n):
    Information = ""
    if n:
        return Information
    
@callback(
    Output("test-pval-output", "children", allow_duplicate=True),
    [Input("load-button", "n_clicks")],
    prevent_initial_call=True
)
def load_pval_wilcoxon_reset(n):
    Information = ""
    if n:
        return Information
    
################################
# Actions
################################
# Clean the action textbox
@callback(
    Output("action-text", "children", allow_duplicate=True),
    [Input("load-button", "n_clicks")],
    prevent_initial_call=True
)
def action_reset(n):
    if n:
        return "Agent's action ... "
    
# Clean the explanation textbox
@callback(
    Output("explanation-text", "children", allow_duplicate=True),
    [Input("load-button", "n_clicks")],
    prevent_initial_call=True
)
def explanation_reset(n):
    if n:
        return "Explanation of the agent recommendation ..."