########################################################################################################
#                                           Recommend button callbacks
########################################################################################################
import dash
from dash import callback, Output, Input
from services import my_agent, my_env
from dash import dcc
from dash import ctx

# print the recommended action 
@callback(
    Output("action-text", "children", allow_duplicate=True),
    # Output("load-button", "disabled"),
    # Output("next-button", "disabled"),
    # Output("action-button", "disabled"),
    Input("action-button", "n_clicks"),
    running=[
        (Output("load-button", "disabled"), True, False),
        (Output("next-button", "disabled"), True, False),
        (Output("action-button", "disabled"), True, False)
    ],
    prevent_initial_call=True
)
def make_action(n):
    action = my_agent.my_act(my_env.obs)
    obs = my_env.next_obs(my_agent.action)
    # print("******************")
    # print("from action text")
    # print("action:", action)
    # print("my_agent.action:", my_agent.action)
    # print(my_env.obs.topo_vect)
    # print(obs.topo_vect)
    return action.__str__()

# visualize the recommended action
@callback(
    Output("graphic-output", "children", allow_duplicate=True),
    Input("action-button", "n_clicks"),
    Input("action-text", "children"),
    prevent_initial_call=True
)
def apply_action(n, text_value):
    # if ctx.triggered_id != "action-button" and not text_value:
    #     return dash.no_update
    # if not text_value:
    #     return dash.no_update
    # my_env.next_obs(my_agent.action)
    if n and (text_value != "Agent's action ... "):
        # print("-----------------")
        # print("from graphic output")
        # print("text_value:", text_value)
        # print(my_env.obs.topo_vect)
        # print(my_agent.action)
        return dcc.Graph(figure=my_env.plot_obs())

# update the time stamp
@callback(
    Output("time-stamp", "children", allow_duplicate=True),
    Input("action-button", "n_clicks"),
    Input("action-text", "children"),
    prevent_initial_call=True
)
def update_timestamp_upon_action(n, text_value):
    # if ctx.triggered_id != "action-button" and not text_value:
    #     return dash.no_update
    # if not text_value:
    #     return ""
    if n and (text_value != "Agent's action ... "):
        return my_env.get_time_stamp()
    
################################
# Statistics
################################
@callback(
    Output("distance-output", "children", allow_duplicate=True),
    Input("action-button", "n_clicks"),
    Input("action-text", "children"),
    prevent_initial_call=True
)
def load_distance(n, text_value):
    if n and (text_value != "Agent's action ... "):
        if my_env.env is None:
            raise Exception("The environment is not loaded properly!")
        if (my_env.expert_action is None) or (my_env.expert_action == my_env.env.action_space({})):
            return ""
        else:
            Information = f"{2} hops"
            return Information
    
@callback(
    Output("rho-imp-output", "children", allow_duplicate=True),
    Input("action-button", "n_clicks"),
    Input("action-text", "children"),
    prevent_initial_call=True
)
def load_rho_imrpovement(n, text_value):
    if n and (text_value != "Agent's action ... "):
        # Information = f"{0.4}" # for test
        if (my_env.old_rho is None) or (my_env.current_rho is None):
            return ""
        Information = f"{(my_env.old_rho - my_env.current_rho):.2f}"
        return Information
    
@callback(
    Output("rho-diff-output", "children", allow_duplicate=True),
    Input("action-button", "n_clicks"),
    Input("action-text", "children"),
    prevent_initial_call=True
)
def load_rho_difference(n,text_value):
    if n and (text_value != "Agent's action ... "):
        if my_env.env is None:
            raise Exception("The environment is not yet loaded.")
        if (my_env.expert_action is None) or (my_env.expert_action == my_env.env.action_space({})):
            return ""
        # Information = f"{0.2}"
        expert_action = my_env.suggest_expert_action()
        # old_rho = my_env.obs.rho.max()
        # old_rho = my_env.old_rho
        # TODO: Test if there is no gameover after application of this action
        new_obs_expert, *_ = my_env.env.simulate(expert_action)
        new_rho_expert = new_obs_expert.rho.max()
        
        # new_obs, *_ = my_env.env.simulate(my_agent.action)
        # new_rho_rl = new_obs.rho.max()
        
        rho_diff = my_env.current_rho - new_rho_expert 
        Information = f"{rho_diff:.2f}"
        return Information
    
@callback(
    Output("test-stat-output", "children", allow_duplicate=True),
    Input("action-button", "n_clicks"),
    Input("action-text", "children"),
    prevent_initial_call=True
)
def load_test_wilcoxon(n, text_value):
    if n and (text_value != "Agent's action ... "):
        if my_env.env is None:
            raise Exception("The environment is not yet loaded.")
        if (my_env.expert_action is None) or (my_env.expert_action == my_env.env.action_space({})):
            return ""
        Information = f"Test statistic: {0.2}"
        return Information
    
@callback(
    Output("test-pval-output", "children", allow_duplicate=True),
    Input("action-button", "n_clicks"),
    Input("action-text", "children"),
    prevent_initial_call=True
)
def load_pval_wilcoxon(n, text_value):
    if n and (text_value != "Agent's action ... "):
        if my_env.env is None:
            raise Exception("The environment is not yet loaded.")
        if (my_env.expert_action is None) or (my_env.expert_action == my_env.env.action_space({})):
            return ""
        Information = f"P-value: {0.03}"
        return Information
    