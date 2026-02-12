########################################################################################################
#                                           Recommend button callbacks
########################################################################################################
import dash
from dash import callback, Output, Input, State
from services import my_agent, my_env, compute_stats
from dash import dcc
from dash import no_update

import numpy as np

# print the recommended action 
@callback(
    Output("action-text", "children", allow_duplicate=True),
    Output("expert-text", "children", allow_duplicate=True),
    Output("action-taken", "data"),
    Output("action-stats", "data"),
    Input("action-button", "n_clicks"),
    running=[
        (Output("load-button", "disabled"), True, False),
        (Output("next-button", "disabled"), True, False),
        (Output("action-button", "disabled"), True, False)
    ],
    prevent_initial_call=True
)
def make_action(n):
    my_env.expert_action = None
    my_env.expert_rho = None
    # get the expert action before changing the environment with Agent's recommendation
    expert_action = my_env.suggest_expert_action()
    
    # get the action from the agent
    agent_action = my_agent.my_act(my_env.obs)
    # apply the action
    obs = my_env.next_obs(my_agent.action)
    
    stats_dict = compute_stats(obs, agent_action, expert_action)
    # print("******************")
    # print("from action text")
    # print("action:", action)
    # print("my_agent.action:", my_agent.action)
    # print(my_env.obs.topo_vect)
    # print(obs.topo_vect)
    return agent_action.__str__(), expert_action.__str__(), True, stats_dict

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
    Input("action-stats", "data"),
    prevent_initial_call=True
)
def load_distance(stats):
    if stats is None:
        return no_update
    return stats["distance"]
    
@callback(
    Output("rho-imp-output", "children", allow_duplicate=True),
    Input("action-stats", "data"),
    prevent_initial_call=True
)
def load_rho_imrpovement(stats):
    if stats is None:
        return no_update
    return stats["rho_imp"]
    
@callback(
    Output("rho-diff-output", "children", allow_duplicate=True),
    Input("action-stats", "data"),
    prevent_initial_call=True
)
def load_rho_difference(stats):
    if stats is None:
        return no_update
    return stats["rho_diff"]
    
@callback(
    Output("test-stat-output", "children", allow_duplicate=True),
    Input("action-stats", "data"),
    prevent_initial_call=True
)
def load_test_wilcoxon(stats):
    if stats is None:
        return no_update
    return stats["test_stat"]
    
@callback(
    Output("test-pval-output", "children", allow_duplicate=True),
    Input("action-stats", "data"),
    prevent_initial_call=True
)
def load_pval_wilcoxon(stats):
    if stats is None:
        return no_update
    return stats["test_pval"]
    
    
#####################
# Explanation
####################
@callback(
    Output("explanation-text", "children", allow_duplicate=True),
    Input("action-stats", "data"),
    prevent_initial_call=True
)
# def give_explanation(action_ready, action_text, distance, rho_imp, rho_diff, test_stat, test_pval):
def give_explanation(stats):
    if stats is None:
        return no_update
    
    distance = stats["distance"]
    rho_imp = stats["rho_imp"]
    rho_diff = stats["rho_diff"]
    test_stat = stats["test_stat"]
    test_pval = stats["test_pval"]
        
    text = f"The Power Grid was charged at {my_env.old_rho:.2f} before applying the Agent's action.\n"
    
    if np.any(my_env.old_obs.rho > .99):
        overloaded_lines = np.where(my_env.old_obs.rho > .99)[0]
        text += f"The power lines {overloaded_lines} were overloaded.\n"    
        sub_ids = []
        for line_id in overloaded_lines:
            sub_or = my_env.env.line_or_to_subid[line_id]
            sub_ex = my_env.env.line_ex_to_subid[line_id]
            sub_ids.append(sub_or)
            sub_ids.append(sub_ex)
        sub_ids = list(np.unique(sub_ids))
        sub_ids = [int(el) for el in sub_ids]
        text += f"Substations {sub_ids} were concerned.\n"    
            
            
    if (distance is not None) and (distance != ""):
        distance = int(distance)
        if distance <= 2:
            text += "The action selected by the agent is very close to the region suggested by the expert. \n"
        elif distance <= 5:
            text += "The action selecdted by the agent is somehow far from the zone suggested by the expert \n"
        else:
            text += "The action is completely independent from the zone suggested by the expert. See othe stats for the performance. \n"
            
    if (rho_imp is not None) and (rho_imp != ""):
        rho_imp = float(rho_imp)
        if rho_imp == 0:
            text += "The action recommended by the agent has no impact on current overload state of the grid. \n"
            text += f"The grid is curently charged at {my_env.current_rho:.2f}\n"
        elif rho_imp > 0:
            text += f"The action recommended by the agent helped to reduce the overload by {rho_imp}. \n"
        elif rho_imp < 0:
            text += f"The action recommended by the agent degraded the overload state of the grid by {rho_imp}. \n"
        
    # print("distance: ", distance)
    # print("rho_imp: ", rho_imp)
    # print("rho_diff: ", rho_diff)
    # print("test_stat: ",test_stat)
    # print("test_pval: ", test_pval)
    return text
