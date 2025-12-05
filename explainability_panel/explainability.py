# Import packages
import sys
sys.path.append("deepQExpert_v2")
sys.path.append("deepQExpert_v2/*")
import os
import configparser
import time
import numpy as np
import grid2op
from grid2op.Reward import L2RPNSandBoxScore, L2RPNReward
from grid2op.PlotGrid import PlotMatplot, PlotPlotly

from dash import Dash, html, dcc, callback, Output, Input, State
import dash_ag_grid as dag
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

import PIL
import io
import cairosvg

from deepQExpert_v2.DeepQ_NN import DeepQ_NN
from deepQExpert_v2.DeepQ_NNParam import DeepQ_NNParam
from deepQExpert_v2.DeepQSimple import DeepQSimple

from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation
from alphaDeesp.core.graphsAndPaths import OverFlowGraph

class MyEnv():
    def __init__(self, path_env, config_path="config_expert.ini"):
        self.path_env = path_env
        self.env = None
        self.obs = None
        self.env_reset = False
        self.next = False
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        # self.reset()
        
    def reset(self):
        env = grid2op.make(dataset=self.path_env, # args.env_name,
                           reward_class=L2RPNSandBoxScore,
                           other_rewards={
                               "reward": L2RPNReward
                           })
        self.obs = env.reset()
        self.env = env
        self.env_reset = True
        return self.env, self.obs
    
    def next_obs(self, action):
        if action is None:
            action = self.env.action_space({})
        # TODO: the other information as GameOver could be interesting to avoid errors in interface
        self.obs, *_ = self.env.step(action)
        self.next = True
        return self.obs

    def get_time_stamp(self):
        return self.env.time_stamp
    
    def plot_obs(self):
        plot_helper = PlotPlotly(self.env.observation_space, width=700, height=500)
        return plot_helper.plot_obs(self.obs)
    
    def plot_overload_graph(self):
        if not(self.env_reset):
            raise Exception("The environment should be loaded at first")
        ltc=list(np.where(self.obs.rho>1)[0])#overloaded line to solve
        # ltc = [4]
        if not(ltc):
            return self.plot_obs()
        sim = Grid2opSimulation(self.obs, 
                                self.env.action_space, 
                                self.env.observation_space, 
                                param_options=self.config["DEFAULT"], 
                                debug=False,
                                ltc=ltc,
                                plot=False)
        g_over =  OverFlowGraph(sim.topo, ltc, sim.get_dataframe())
        rescale_factor=1#for better layout, you can play with it to change the zoom level
        layout_rescale=[(e[0]/rescale_factor,e[1]/rescale_factor) for e in sim.layout]
        svg = g_over.plot(layout_rescale,save_folder="")
        im = PIL.Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg, dpi=500, output_width=1920, output_height=1080, scale=5))).convert("RGBA")
        fig = px.imshow(im, width=1920, height=1080, title="overload graph")
        fig = go.Figure(fig.data)
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return fig

class MyAgent():
    def __init__(self, load_path: str, name: str="DeepQExpert"):
        self.load_path = load_path
        self.name = name
        self.agent = None
        self.action = None
        self.loaded = False

    def load(self, env):
        path_model, path_target_model = DeepQ_NN.get_path_model(self.load_path, self.name)
        nn_archi = DeepQ_NNParam.from_json(os.path.join(path_model, "nn_architecture.json"))
        agent = DeepQSimple(env=env,
                        action_space=env.action_space,
                        name=self.name,
                        store_action=True,
                        nn_archi=nn_archi,
                        observation_space=env.observation_space)
        agent.load(self.load_path)
        self.agent = agent
        self.loaded = True
        
    def my_act(self, obs):
        if not(self.loaded):
            raise Exception("Agent not loaded")
        
        transformed_observation = self.agent.convert_obs(obs)
        act_id = self.agent.my_act(transformed_observation, None)
        action = self.agent.convert_act(act_id)
        self.action = action
        return action

ENV_PATH = "../ressources/ai4realnet_small"
# env, obs = load_environment(ENV_PATH)
my_env = MyEnv(ENV_PATH)
my_agent = MyAgent(load_path="deepQExpert_v2/ai4realnet_small")

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = dbc.Container([
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
                    dbc.Col(html.Div("", id="env-name-output", className="fs-4 border", style={"width": 230})),
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
         dbc.Button("Next scenario", color="primary", id="next-button", n_clicks=0, className="me-1", size="lg", outline=True),
         dbc.Button("Recommend Action", color="primary", id="action-button", n_clicks=0, className="me-1", size="lg", outline=True)]
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

##########################
# Load button callbacks
##########################
@app.callback(
    # Output("graphic-output", "figure", allow_duplicate=True),
    Output("graphic-output", "children", allow_duplicate=True),
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
        _ = my_env.reset()
        my_agent.load(my_env.env)
        return dcc.Graph(figure=my_env.plot_overload_graph())
        # return  dcc.Graph(figure=my_env.plot_obs())
        # svg = my_env.plot_overload_graph()
        # cairosvg.svg2png(bytestring=svg, write_to="output.png")
        # time.sleep(3)
        # return "/assets/output.png"
    
    return None

@app.callback(
    Output("time-stamp", "children", allow_duplicate=True),
    [Input("load-button", "n_clicks")],
    prevent_initial_call=True
)
def load_time_stamp(n):
    if n:
        time.sleep(2.5)
        if my_env.env_reset:
            return my_env.get_time_stamp()
        
    return None

@app.callback(
    Output("env-name-output", "children", allow_duplicate=True),
    [Input("load-button", "n_clicks")],
    prevent_initial_call=True
)
def load_env_name(n):
    if n:
        time.sleep(2.5)
        if my_env.env_reset:
            return my_env.env.env_name
    
    return None

@app.callback(
    Output("distance-output", "children", allow_duplicate=True),
    [Input("load-button", "n_clicks")],
    prevent_initial_call=True
)
def load_distance_reset(n):
    Information = ""
    if n:
        return Information
    
@app.callback(
    Output("rho-imp-output", "children", allow_duplicate=True),
    [Input("load-button", "n_clicks")],
    prevent_initial_call=True
)
def load_rho_imrpovement_reset(n):
    Information = ""
    if n:
        return Information
    
@app.callback(
    Output("rho-diff-output", "children", allow_duplicate=True),
    [Input("load-button", "n_clicks")],
    prevent_initial_call=True
)
def load_rho_difference_reset(n):
    Information = ""
    if n:
        return Information
    
@app.callback(
    Output("test-stat-output", "children", allow_duplicate=True),
    [Input("load-button", "n_clicks")],
    prevent_initial_call=True
)
def load_test_wilcoxon_reset(n):
    Information = ""
    if n:
        return Information
    
@app.callback(
    Output("test-pval-output", "children", allow_duplicate=True),
    [Input("load-button", "n_clicks")],
    prevent_initial_call=True
)
def load_pval_wilcoxon_reset(n):
    Information = ""
    if n:
        return Information

##########################
# Next button callbacks
##########################

@app.callback(
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
            obs = my_env.next_obs(None)
            return dcc.Graph(figure=my_env.plot_obs())
            # return my_env.plot_obs()
    return None

@app.callback(
    Output("time-stamp", "children", allow_duplicate=True),
    [Input("next-button", "n_clicks")],
    prevent_initial_call=True
)
def reload_time_stamp(n):
    if n:
        if my_env.env_reset & my_env.next:
            return my_env.get_time_stamp()
        elif my_env.env_reset:
            time.sleep(1)
            my_env.next = False
            return my_env.get_time_stamp()
            
    return None

@app.callback(
    Output("distance-output", "children", allow_duplicate=True),
    [Input("next-button", "n_clicks")],
    prevent_initial_call=True
)
def load_distance_next(n):
    Information = ""
    if n:
        return Information
    
@app.callback(
    Output("rho-imp-output", "children", allow_duplicate=True),
    [Input("next-button", "n_clicks")],
    prevent_initial_call=True
)
def load_rho_imrpovement_next(n):
    Information = ""
    if n:
        return Information
    
@app.callback(
    Output("rho-diff-output", "children", allow_duplicate=True),
    [Input("next-button", "n_clicks")],
    prevent_initial_call=True
)
def load_rho_difference_next(n):
    Information = ""
    if n:
        return Information
    
@app.callback(
    Output("test-stat-output", "children", allow_duplicate=True),
    [Input("next-button", "n_clicks")],
    prevent_initial_call=True
)
def load_test_wilcoxon_next(n):
    Information = ""
    if n:
        return Information
    
@app.callback(
    Output("test-pval-output", "children", allow_duplicate=True),
    [Input("next-button", "n_clicks")],
    prevent_initial_call=True
)
def load_pval_wilcoxon_next(n):
    Information = ""
    if n:
        return Information

##########################
# Action button callbacks
##########################

@app.callback(
    Output("action-text", "children", allow_duplicate=True),
    [Input("action-button", "n_clicks")],
    running=[
        (Output("load-button", "disabled"), True, False),
        (Output("next-button", "disabled"), True, False),
        (Output("action-button", "disabled"), True, False)
    ],
    prevent_initial_call=True
)
def make_action(n):
    if n:
        action = my_agent.my_act(my_env.obs)
        return action.__str__()

@app.callback(
    Output("distance-output", "children", allow_duplicate=True),
    [Input("action-button", "n_clicks")],
    running=[
        (Output("load-button", "disabled"), True, False),
        (Output("next-button", "disabled"), True, False),
        (Output("action-button", "disabled"), True, False)
    ],
    prevent_initial_call=True
)
def load_distance(n):
    Information = f"{2} hops"
    if n:
        return Information
    
@app.callback(
    Output("rho-imp-output", "children", allow_duplicate=True),
    [Input("action-button", "n_clicks")],
    prevent_initial_call=True
)
def load_rho_imrpovement(n):
    Information = f"{0.4}"
    if n:
        return Information
    
@app.callback(
    Output("rho-diff-output", "children", allow_duplicate=True),
    [Input("action-button", "n_clicks")],
    prevent_initial_call=True
)
def load_rho_difference(n):
    Information = f"{0.2}"
    if n:
        return Information
    
@app.callback(
    Output("test-stat-output", "children", allow_duplicate=True),
    [Input("action-button", "n_clicks")],
    prevent_initial_call=True
)
def load_test_wilcoxon(n):
    Information = f"Test statistic: {0.2}"
    if n:
        return Information
    
@app.callback(
    Output("test-pval-output", "children", allow_duplicate=True),
    [Input("action-button", "n_clicks")],
    prevent_initial_call=True
)
def load_pval_wilcoxon(n):
    Information = f"P-value: {0.03}"
    if n:
        return Information


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
