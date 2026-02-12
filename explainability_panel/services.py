import os
import copy
import configparser
import numpy as np
import networkx as nx
import grid2op
from grid2op.Reward import L2RPNSandBoxScore, L2RPNReward
from grid2op.PlotGrid import PlotMatplot, PlotPlotly
from lightsim2grid import LightSimBackend

import PIL
import io
import cairosvg

from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation
from alphaDeesp.core.graphsAndPaths import OverFlowGraph
from alphaDeesp.expert_operator import expert_operator

from ExpertAgent import ExpertAgentHeuristic
from explainability_panel import get_package_root

class MyEnv():
    def __init__(self, 
                 path_env: str, 
                 expert_config_path: str="config_expert.ini", 
                 seed: int=42, 
                 chronic_id: int=0):
        self.path_env = path_env
        self.env = None
        self.obs = None
        self.old_obs = None
        self.env_reset = False
        self.next = False
        self.config = configparser.ConfigParser()
        self.config.read(expert_config_path)
        self.seed = seed
        self.chronic_id = 0
        self.old_rho = None
        self.current_rho = None
        self.expert_action = None
        self.expert_rho = None
        # self.reset()
        
    def reset(self, seed=None, chronic_id=None):
        env = grid2op.make(dataset=self.path_env, # args.env_name,
                           reward_class=L2RPNSandBoxScore,
                           other_rewards={
                               "reward": L2RPNReward
                           },
                           )# backend=LightSimBackend())
        if seed is None:
            env.seed(self.seed)
        else:
            env.seed(seed)
        if chronic_id is None:
            env.set_id(self.chronic_id)
        else:
            env.set_id(chronic_id)
        self.obs = env.reset()
        self.old_rho = self.obs.rho.max()
        self.current_rho = self.obs.rho.max()
        self.env = env
        self.env_reset = True
        return self.env, self.obs
    
    def next_obs(self, action):
        if (self.env is None) or (self.obs is None):
            raise Exception("The environment should be loaded at first")
        if action is None:
            action = self.env.action_space({})
        # TODO: the other information as GameOver could be interesting to avoid errors in interface
        self.old_obs = copy.deepcopy(self.obs)
        self.old_rho = self.obs.rho.max()
        self.obs, *_ = self.env.step(action)
        self.current_rho = self.obs.rho.max()
        self.next = True
        return self.obs
    
    def next_obs_to_overload(self):
        if self.obs is None:
            raise Exception("The environment should be loaded at first")
        # TODO: This is for debugging purpose, could be removed once everything works as expected
        self.old_obs = copy.deepcopy(self.obs)
        self.old_rho = self.obs.rho.max()
        self.obs = progress_env_to_overload(self.env)
        self.current_rho = self.obs.rho.max()
        self.next = True
        return self.obs

    def get_time_stamp(self):
        if self.env is None:
            raise Exception("The environment should be loaded at first")
        return self.env.time_stamp
    
    def suggest_expert_action(self):
        if not(self.env_reset) or (self.obs is None) or (self.env is None):
            raise Exception("The environment should be loaded at first")
        ltc = list(np.where(self.obs.rho>=1.)[0])#overloaded line to solve
        print("ltc:", ltc)
        if not(ltc):
            self.expert_action = self.env.action_space({})
            return self.expert_action
        
        sim = Grid2opSimulation(self.obs, 
                                self.env.action_space, 
                                self.env.observation_space, 
                                param_options=self.config["DEFAULT"], 
                                debug=False,
                                ltc=ltc,
                                plot=False)
        
        ranked_combinations, expert_system_results, actions = expert_operator(sim, 
                                                                              plot=False, 
                                                                              debug=False)
        most_efficient_action = expert_system_results["Efficacity"].values.argmax()
        self.expert_action = actions[most_efficient_action]
        
        new_obs_expert, *_ = self.env.simulate(self.expert_action)
        self.expert_rho = new_obs_expert.rho.max()
        
        return self.expert_action

    def plot_obs(self):
        if self.env is None:
            raise Exception("The environment should be loaded at first")
        plot_helper = PlotPlotly(self.env.observation_space, width=700, height=500)
        return plot_helper.plot_obs(self.obs)
    
    def plot_overload_graph(self):
        if not(self.env_reset) or (self.obs is None) or (self.env is None):
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
        im = PIL.Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg, 
                                                        dpi=500, 
                                                        output_width=1920, 
                                                        output_height=1080, 
                                                        scale=5))).convert("RGBA")
        fig = px.imshow(im, width=1920, height=1080, title="overload graph")
        fig = go.Figure(fig.data)
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return fig

class MyAgent():
    def __init__(self):
        self.load_path = None
        self.name = None
        self.agent = None
        self.action = None

    def load(self, env):
        agent = ExpertAgentHeuristic(action_space=env.action_space,
                                     env=env,
                                    )
        self.agent = agent
        
    def my_act(self, obs):            
        if self.agent is not None:
            self.action = self.agent.act(obs, reward=None)
        else:
            raise Exception("Agent not loaded")
        return self.action
    
# TODO: for debugging purpose to be removed
def progress_env_to_overload(env):
    do_nothing = env.action_space({})
    for i in range(1000):
        obs, *_ = env.step(do_nothing)
        if obs.rho.max() > 1.:
            print(f"Time step: {i}")
            print(obs.get_time_stamp())
            print(obs.rho.max())
            return obs

def compute_stats(obs, agent_action, expert_action):
    distance = compute_distance(obs, agent_action, expert_action)
    rho_imp = compute_rho_improvement()
    rho_diff = compute_rho_difference()
    test_stat = compute_test_wilcoxon()
    test_pval = compute_pval_wilcoxon()
    
    return {
        "distance": distance,
        "rho_imp": rho_imp,
        "rho_diff": rho_diff,
        "test_stat": test_stat,
        "test_pval": test_pval
    }
    
def compute_distance(obs, agent_action, expert_action):
    if my_env.env is None or my_env.obs is None:
        raise Exception("The environment is not loaded properly!")
    do_nothing = my_env.env.action_space({})
    if agent_action == do_nothing or expert_action == do_nothing:
        return ""
    graph = obs.get_energy_graph()
    print(expert_action)
    print(agent_action)
    expert_action_sub = np.where(expert_action.get_topological_impact()[1])[0][0]
    agent_action_sub = np.where(agent_action.get_topological_impact()[1])[0][0]
    Information = nx.shortest_path_length(graph, 
                                          source=expert_action_sub,
                                          target=agent_action_sub)
    return Information

def compute_rho_improvement():
    if (my_env.old_rho is None) or (my_env.current_rho is None):
        return ""
    Information = f"{(my_env.old_rho - my_env.current_rho):.2f}"
    return Information

def compute_rho_difference():
    if my_env.env is None:
        raise Exception("The environment is not yet loaded.")
    # # Information = f"{0.2}"
    # expert_action = my_env.suggest_expert_action()
    # print("expert_action: ", expert_action)
    if (my_env.expert_action is None) or (my_env.expert_action == my_env.env.action_space({})):
        return ""
    # # old_rho = my_env.obs.rho.max()
    # # old_rho = my_env.old_rho
    # # TODO: Test if there is no gameover after application of this action
    # new_obs_expert, *_ = my_env.env.simulate(expert_action)
    # new_rho_expert = new_obs_expert.rho.max()
    
    # # new_obs, *_ = my_env.env.simulate(my_agent.action)
    # # new_rho_rl = new_obs.rho.max()
    
    # rho_diff = my_env.current_rho - new_rho_expert
    rho_diff = my_env.current_rho - my_env.expert_rho 
    Information = f"{rho_diff:.2f}"
    return Information

def compute_test_wilcoxon():
    if my_env.env is None:
        raise Exception("The environment is not yet loaded.")
    if (my_env.expert_action is None) or (my_env.expert_action == my_env.env.action_space({})):
        return ""
    Information = f"Test statistic: {0.2}"
    return Information

def compute_pval_wilcoxon():
    if my_env.env is None:
        raise Exception("The environment is not yet loaded.")
    if (my_env.expert_action is None) or (my_env.expert_action == my_env.env.action_space({})):
        return ""
    Information = f"P-value: {0.03}"
    return Information

ENV_PATH = "../ressources/ai4realnet_small"
EXPERT_CONFIG_PATH=os.path.join(get_package_root(), "explainability_panel", "config_expert.ini")
# env, obs = load_environment(ENV_PATH)
my_env = MyEnv(path_env=ENV_PATH, expert_config_path=EXPERT_CONFIG_PATH)
my_agent = MyAgent()
# my_env.reset()

