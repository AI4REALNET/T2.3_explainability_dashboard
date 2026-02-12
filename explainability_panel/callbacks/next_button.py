########################################################################################################
#                                       Next button callbacks
########################################################################################################
import time
from dash import callback, Output, Input, State
from services import my_env, my_agent
from dash import dcc

@callback(
    # Output("graphic-output", "figure", allow_duplicate=True),
    Output("graphic-output", "children", allow_duplicate=True),
    [Input("next-button", "n_clicks")],
    running=[
        (Output("load-button", "disabled"), True, False),
        (Output("next-button", "disabled"), True, False),
        (Output("action-button", "disabled"), True, False)
    ],
    prevent_initial_call=True
)
def load_next_obs(n):
    if n:
        if my_env.env_reset:
        # new_obs = next_obs(env, None)
            # obs = my_env.next_obs(None)
            # TODO: for debugging purpose, to be replaced by the above line
            my_env.next = False
            obs = my_env.next_obs_to_overload()
            # return dcc.Graph(figure=my_env.plot_overload_graph())
            return dcc.Graph(figure=my_env.plot_obs())
            # return my_env.plot_obs()

@callback(
    Output("time-stamp", "children", allow_duplicate=True),
    [Input("next-button", "n_clicks")],
    # Input("graphic-output", "children"),
    prevent_initial_call=True
)
def reload_time_stamp(n):
    if n:
        return my_env.get_time_stamp()
    # if n:
    #     # while not my_env.next:
    #     #     pass
    #     # my_env.next = False
    #     # return my_env.get_time_stamp()
    #     if my_env.env_reset & my_env.next:
    #         return my_env.get_time_stamp()
    #     elif my_env.env_reset:
    #         time.sleep(1)
    #         my_env.next = False
    #         return my_env.get_time_stamp()
            
    # return None
    
################################
# Statistics
################################

@callback(
    Output("distance-output", "children", allow_duplicate=True),
    [Input("next-button", "n_clicks")],
    prevent_initial_call=True
)
def load_distance_next(n):
    Information = ""
    if n:
        return Information
    
@callback(
    Output("rho-imp-output", "children", allow_duplicate=True),
    [Input("next-button", "n_clicks")],
    prevent_initial_call=True
)
def load_rho_imrpovement_next(n):
    Information = ""
    if n:
        return Information
    
@callback(
    Output("rho-diff-output", "children", allow_duplicate=True),
    [Input("next-button", "n_clicks")],
    prevent_initial_call=True
)
def load_rho_difference_next(n):
    Information = ""
    if n:
        return Information
    
@callback(
    Output("test-stat-output", "children", allow_duplicate=True),
    [Input("next-button", "n_clicks")],
    prevent_initial_call=True
)
def load_test_wilcoxon_next(n):
    Information = ""
    if n:
        return Information
    
@callback(
    Output("test-pval-output", "children", allow_duplicate=True),
    [Input("next-button", "n_clicks")],
    prevent_initial_call=True
)
def load_pval_wilcoxon_next(n):
    Information = ""
    if n:
        return Information

################################
# Actions
################################
# Clean the action textbox
@callback(
    Output("action-text", "children", allow_duplicate=True),
    [Input("next-button", "n_clicks")],
    prevent_initial_call=True
)
def clean_action(n):
    if n:
        return "Agent's action ... "
    
# Clean the explanation textbox
@callback(
    Output("explanation-text", "children", allow_duplicate=True),
    [Input("next-button", "n_clicks")],
    prevent_initial_call=True
)
def clean_explanation(n):
    if n:
        return "Explanation of the agent recommendation ..."