# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import warnings
import logging
import numpy as np
from tqdm import tqdm
import tensorflow as tf

try:
    from grid2op.Chronics import MultifolderWithCache
    _CACHE_AVAILABLE_DEEPQAGENT = True
except ImportError:
    _CACHE_AVAILABLE_DEEPQAGENT = False

from DeepQAgent import DeepQAgent
from trainingParam import TrainingParam

from heuristics import RecoPowerlineModule, RecoverInitTopoModule
from heuristics import TopoNNTopKModule
from convex_optim import OptimModule

logging.basicConfig(level=logging.INFO, filename="agents.log", filemode="w")
logger = logging.getLogger(__name__)

DEFAULT_NAME = "DeepQSimple"

class DeepQSimple(DeepQAgent):
    """
    A simple deep q learning algorithm. It does nothing different thant its base class.
    """
    def __init__(self,
                 env,
                 action_space,
                 nn_archi,
                 name="DeepQAgent",
                 store_action=True,
                 istraining=False,
                 filter_action_fun=None,
                 verbose=False,
                 observation_space=None,
                 rho_danger: float = 0.99,
                 rho_safe: float = 0.9,
                 **kwargs_converters):
        super().__init__(action_space, nn_archi, name, store_action, istraining, filter_action_fun, verbose, observation_space, **kwargs_converters)
        self.kwargs_converters = kwargs_converters
        # Heuristic
        self.reconnect = RecoPowerlineModule(self.action_space)
        self.recover_topo = RecoverInitTopoModule(self.action_space)
        
        # TOPNN
        self.topo_agent = TopoNNTopKModule(
            action_space=self.action_space,
            agent=self,
            top_k=10
        )
        
        # Continuous control
        self.optim = OptimModule(env, self.action_space)
        
        self.rho_danger = rho_danger
        self.rho_safe = rho_safe
        self.action_list = []
        # self.epsilon2 = 1.0
        # self._explore_expert_num = 0
        
    def act(self, observation, reward, done=False):
        
        # Init action with DoNothing
        act = self.action_space({})
        
        # Try to perform reconnection if necessary
        reconnect_act = self.reconnect.get_act(observation, act, reward)
        _obs, _rew, _done, _info = observation.simulate(reconnect_act, time_step=1)
        
        if reconnect_act is not None:  
            if (reconnect_act is not None
                and not _done
                and reconnect_act != self.action_space({})
                and 0. < _obs.rho.max() < 2.
                and (len(_info["exception"]) == 0)
            ):
                logger.info("calling reconnection module")
                act += reconnect_act
                
        if observation.rho.max() > self.rho_danger:
            recovery_act = self.recover_topo.get_act(
                observation, act, reward, rho_threshold=self.rho_danger
            )
            if recovery_act is not None:
                logger.info("Calling recovery action")
                act += recovery_act
            else:
                if all(observation.line_status):
                    # TODO : this if condition could be useful for implementation of domains shift KPI
                    # TODO : To see if a separate agent should be trained for n-1 cases
                    # gym_obs = self.env_gym.observation_space.to_gym(observation)
                    # topo_act = self.topo_agent.get_act(gym_obs, act, reward)
                    # topo_act = self.env_gym.action_space.from_gym(topo_act)
                    # topo_act = self.topo_agent.get_act(observation, act, reward)
                    # Action taken by the agent
                    transformed_observation = self.convert_obs(observation)
                    encoded_act = self.my_act(transformed_observation, reward, done)
                    topo_act = self.convert_act(encoded_act)
                    # topo_act = self.topo_agent.get_act(observation, act, reward)
                
                    if topo_act is not None:
                        logger.info("Calling topo agent")
                        _obs, *_ = observation.simulate(act + topo_act)
                        logger.info(topo_act)
                        act += topo_act
                    
            # # CALL OPTIM Module for dispatching actions 
            # _obs, _rew, _done, _info = observation.simulate(act, time_step=1)
            # if _obs.rho.max() > self.rho_safe or (len(_info["exception"]) != 0):
            #     logger.info("calling optim module")
            #     # logger.warning(f"EXCEPTION : {_info['exception']}")
            #     # logger.warning(f"Done: {_done}")
            #     # act = self.action_space({})
            #     act = self.optim.get_act(observation, act, reward)
            
        elif _obs.rho.max() < self.rho_safe:
            # Try to find a recovery action when the grid is safe
            recovery_act = self.recover_topo.get_act(
                observation, act, reward, rho_threshold=0.8
            )
            if recovery_act is not None:
                logger.info("Calling recovery action (grid is safe)")
                act += recovery_act
        if act != self.action_space({}):
            self.action_list.append(act)
        
        return act
    
    def train(self,
              env,
              iterations,
              save_path,
              logdir,
              training_param=None):
        """
        Part of the public l2rpn-baselines interface, this function allows to train the baseline.

        If `save_path` is not None, the the model is saved regularly, and also at the end of training.

        TODO explain a bit more how you can train it.

        Parameters
        ----------
        env: :class:`grid2op.Environment.Environment` or :class:`grid2op.Environment.MultiEnvironment`
            The environment used to train your model.

        iterations: ``int``
            The number of training iteration. NB when reloading a model, this is **NOT** the training steps that will
            be used when re training. Indeed, if `iterations` is 1000 and the model was already trained for 750 time
            steps, then when reloaded, the training will occur on 250 (=1000 - 750) time steps only.

        save_path: ``str``
            Location at which to save the model

        logdir: ``str``
            Location at which tensorboard related information will be kept.

        training_param: :class:`l2rpn_baselines.utils.TrainingParam`
            The meta parameters for the training procedure. This is currently ignored if the model is reloaded (in that
            case the parameters used when first created will be used)

        """        
        if training_param is None:
            training_param = TrainingParam()

        self._train_lr = training_param.lr

        if self._training_param is None:
            self._training_param = training_param
        else:
            training_param = self._training_param
        self._init_deep_q(self._training_param, env)
        self._fill_vectors(self._training_param)
        
        # create an index table to retreive the action ids for specific substations
        # action_size = self.get_action_size(env.action_space, self.filter_action_fun, self.kwargs_converters)
        # for act_id in range(action_size):
        #     action = self.convert_act(act_id)
        #     if action.impact_on_objects()["topology"]["changed"] is True:
        #         self.deep_q._action_dict[act_id] = action.impact_on_objects()["topology"]["bus_switch"][0]["substation"]
        #     else:
        #         self.deep_q._action_dict[act_id] = None

        self._init_replay_buffer()

        # efficient reading of the data (read them by chunk of roughly 1 day
        nb_ts_one_day = 24 * 60 / 5  # number of time steps per day
        self._set_chunk(env, nb_ts_one_day)

        # Create file system related vars
        if save_path is not None:
            save_path = os.path.abspath(save_path)
            os.makedirs(save_path, exist_ok=True)

        if logdir is not None:
            logpath = os.path.join(logdir, self.name)
            self._tf_writer = tf.summary.create_file_writer(logpath, name=self.name)
        else:
            logpath = None
            self._tf_writer = None
        UPDATE_FREQ = training_param.update_tensorboard_freq  # update tensorboard every "UPDATE_FREQ" steps
        SAVING_NUM = training_param.save_model_each

    
        if hasattr(env, "nb_env"):
            nb_env = env.nb_env
            warnings.warn("Training using {} environments".format(nb_env))
            self.__nb_env = nb_env
            self._set_nb_env(nb_env)
        else:
            self.__nb_env = 1
            self._set_nb_env(1)
            
        # if isinstance(env, grid2op.Environment.Environment):
        #     self.__nb_env = 1
        # else:
        #     import warnings
        #     nb_env = env.nb_env
        #     warnings.warn("Training using {} environments".format(nb_env))
        #     self.__nb_env = nb_env

        self.init_obs_extraction(env.observation_space)

        training_step = self._training_param.last_step

        # some parameters have been move to a class named "training_param" for convenience
        self.epsilon = self._training_param.initial_epsilon
        # self.epsilon2 = self._training_param.initial_epsilon2

        # now the number of alive frames and total reward depends on the "underlying environment". It is vector instead
        # of scalar
        alive_frame, total_reward = self._init_global_train_loop()
        reward, done = self._init_local_train_loop()
        epoch_num = 0
        self._losses = np.zeros(iterations)
        alive_frames = np.zeros(iterations)
        total_rewards = np.zeros(iterations)
        new_state = None
        self._reset_num = 0
        self._curr_iter_env = 0
        self._max_reward = env.reward_range[1]

        # action types
        # injection, voltage, topology, line, redispatching = action.get_types()
        self.nb_injection = 0
        self.nb_voltage = 0
        self.nb_topology = 0
        self.nb_line = 0
        self.nb_redispatching = 0
        self.nb_do_nothing = 0

        # for non uniform random sampling of the scenarios
        th_size = None
        self._prev_obs_num = 0
        if self.__nb_env == 1:
            # TODO make this available for multi env too
            if _CACHE_AVAILABLE_DEEPQAGENT:
                if isinstance(env.chronics_handler.real_data, MultifolderWithCache):
                    th_size = env.chronics_handler.real_data.cache_size
            if th_size is None:
                th_size = len(env.chronics_handler.real_data.subpaths)

            # number of time step lived per possible scenarios
            if self._time_step_lived is None or self._time_step_lived.shape[0] != th_size:
                self._time_step_lived = np.zeros(th_size, dtype=np.uint64)
            # number of time a given scenario has been played
            if self._nb_chosen is None or self._nb_chosen.shape[0] != th_size:
                self._nb_chosen = np.zeros(th_size, dtype=np.uint)
            # number of time a given scenario has been played
            if self._proba is None or self._proba.shape[0] != th_size:
                self._proba = np.ones(th_size, dtype=np.float64)

        self._prev_id = 0
        # this is for the "limit the episode length" depending on your previous success
        self._total_sucesses = 0

        # it's the first ever loop
        # self._curr_iter_env += 1
        # obs = env.reset()
        # if self.__nb_env == 1:
        #     # still hack to have same program interface between multi env and not multi env
        #     obs = [obs]
        # initial_state = self._convert_obs_train(obs)
            
        # initial_state = self._need_reset(env, training_step, epoch_num, done, None)
        
        with tqdm(total=iterations - training_step, disable=not self.verbose) as pbar:
            while training_step < iterations:
                # reset or build the environment
                # TODO: Milad : source of the problem
                initial_state = self._need_reset(env, training_step, epoch_num, done, new_state) 
                initial_state_cpy = initial_state.copy()
                

                # code added to get the right observation format
                if self.__nb_env == 1:
                    obs = env.current_obs
                else:
                    obs = [env.current_obs]
                # if obs is not None:
                #     if self.__nb_env == 1:
                #         obs = env.current_obs
                # else:
                #     obs = env.get_obs()
                    
                # Slowly decay the exploration parameter epsilon
                # if self.epsilon > training_param.FINAL_EPSILON:
                self.epsilon = self._training_param.get_next_epsilon(current_step=training_step)
                # self.epsilon2 = self._training_param.get_next_epsilon2(current_step=training_step)

                # taking an expert action with respect to epsilon 2
                # rand_val = np.random.random()
                # if rand_val < self.epsilon2:
                #     expert_subs = self._get_expert_knowledge(env, obs) # this should match the id of extracted actions to the ids of the whole action space (normally it should be done automatically without any specific operation)
                #     if expert_subs is not None:
                #         expert_actions = [act_id for act_id, sub_id in action_dict.items() if sub_id in expert_subs]
                #         expert_action_index = np.array([np.random.choice(expert_actions)])
                #     else:
                #         # TODO: We should take an action in random if it is None and not DoNothing
                #         expert_action_index = [0] # DoNothing
                #     expert_action = self._convert_all_act(expert_action_index)
                #     self._explore_expert_num += 1
                # else:
                #     expert_action_index = None
                #     expert_action = None
                
                # then we need to predict the next moves. Agents have been adapted to predict a batch of data
                # pm_i, pq_v, act = self._next_move(initial_state, self.epsilon, expert_action_index)
                pm_i, pq_v, act = self._next_move(initial_state, self.epsilon, training_step, env , obs)

                # todo store the illegal / ambiguous / ... actions
                reward, done = self._init_local_train_loop()
                if self.__nb_env == 1:
                    # still the "hack" to have same interface between multi env and env...
                    # yeah it's a pain
                    act = act[0]

                temp_observation_obj, temp_reward, temp_done, info = env.step(act)

                if self.__nb_env == 1:
                    # dirty hack to wrap them into list
                    temp_observation_obj = [temp_observation_obj]
                    temp_reward = np.array([temp_reward], dtype=np.float32)
                    temp_done = np.array([temp_done], dtype=bool)
                    info = [info]

                new_state = self._convert_obs_train(temp_observation_obj) # TODO: It overwrite the initial_state using the new_state values
                self._updage_illegal_ambiguous(training_step, info)
                done, reward, total_reward, alive_frame, epoch_num \
                    = self._update_loop(done, temp_reward, temp_done, alive_frame, total_reward, reward, epoch_num)

                # update the replay buffer
                self._store_new_state(initial_state_cpy, pm_i, reward, done, new_state)

                # now train the model
                if not self._train_model(training_step):
                    # infinite loss in this case
                    raise RuntimeError("ERROR INFINITE LOSS")

                # Save the network every 1000 iterations
                if training_step % SAVING_NUM == 0 or training_step == iterations - 1:
                    self.save(save_path)

                # save some information to tensorboard
                alive_frames[epoch_num] = np.mean(alive_frame)
                total_rewards[epoch_num] = np.mean(total_reward)
                self._store_action_played_train(training_step, pm_i)
                self._save_tensorboard(training_step, epoch_num, UPDATE_FREQ, total_rewards, alive_frames)
                training_step += 1
                pbar.update(1)

        self.save(save_path)

    def _next_move(self, curr_state, epsilon, training_step, env, obs):
        pm_i, pq_v, q_actions = self.deep_q.predict_movement(curr_state, epsilon, env, obs, training=True)
        pm_i, pq_v = self._short_circuit_actions(training_step, pm_i, pq_v, q_actions)
        act = self._convert_all_act(pm_i)
        return pm_i, pq_v, act
    
    def get_nb_env(self):
        return self.__nb_env
    