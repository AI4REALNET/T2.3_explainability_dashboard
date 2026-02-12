########################################################################################################
#                                       Next button callbacks
########################################################################################################
import time
import numpy as np
from dash import callback, Output, Input, State
from services import my_env, my_agent
from dash import dcc, no_update

@callback(
    # Output("graphic-output", "figure", allow_duplicate=True),
    Output("graphic-output", "children", allow_duplicate=True),
    Output("explanation-text", "children", allow_duplicate=True),
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
            # Graphic
            # new_obs = next_obs(env, None)
            # obs = my_env.next_obs(None)
            # TODO: for debugging purpose, to be replaced by the above line
            my_env.next = False
            obs = my_env.next_obs_to_overload()
            
            # Explanation
            text = ""
            
            text += f"The Power Grid is charged at {obs.rho.max():.2f}.\n"
            if np.any(obs.rho > .99):
                overloaded_lines = np.where(obs.rho > .99)[0]
                text += f"The power lines {overloaded_lines} are overloaded.\n"    
                sub_ids = []
                for line_id in overloaded_lines:
                    sub_or = my_env.env.line_or_to_subid[line_id]
                    sub_ex = my_env.env.line_ex_to_subid[line_id]
                    sub_ids.append(sub_or)
                    sub_ids.append(sub_ex)
                sub_ids = list(np.unique(sub_ids))
                sub_ids = [int(el) for el in sub_ids]
                text += f"Substations {sub_ids} are concerned.\n" 
            # return dcc.Graph(figure=my_env.plot_overload_graph())
            return dcc.Graph(figure=my_env.plot_obs()), text
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

@callback(
    Output("expert-text", "children", allow_duplicate=True),
    [Input("next-button", "n_clicks")],
    prevent_initial_call=True
)
def clean_expert_action(n):
    if n:
        return "Expert action ... "