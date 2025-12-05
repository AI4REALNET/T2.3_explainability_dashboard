# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

# tf2.0 friendly
import warnings
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Activation, Dense
    from tensorflow.keras.layers import Input

from l2rpn_baselines.utils import BaseDeepQ
from trainingParam import TrainingParam


class DeepQ_NN(BaseDeepQ):
    """
    Constructs the desired deep q learning network

    Attributes
    ----------
    schedule_lr_model:
        The schedule for the learning rate.
    """

    def __init__(self,
                 nn_params,
                 training_param=None):
        if training_param is None:
            training_param = TrainingParam()
        BaseDeepQ.__init__(self,
                           nn_params,
                           training_param)
        self.schedule_lr_model = None
        self._explore_total_num = 0
        self._explore_expert_num = 0
        self._exploit_num = 0
        self._action_dict = {}
        
        self.construct_q_network()

    def construct_q_network(self):
        """
        The network architecture can be changed with the :attr:`l2rpn_baselines.BaseDeepQ.nn_archi`

        This function will make 2 identical models, one will serve as a target model, the other one will be trained
        regurlarly.
        """
        self._model = Sequential()
        input_layer = Input(shape=(self._nn_archi.observation_size,),
                            name="state")
        lay = input_layer
        for lay_num, (size, act) in enumerate(zip(self._nn_archi.sizes, self._nn_archi.activs)):
            lay = Dense(size, name="layer_{}".format(lay_num))(lay)  # put at self.action_size
            lay = Activation(act)(lay)

        output = Dense(self._action_size, name="output")(lay)

        # self._model = Model(inputs=[input_layer], outputs=[output])
        self._model = Model(inputs=input_layer, outputs=output)
        self._schedule_lr_model, self._optimizer_model = self.make_optimiser()
        self._model.compile(loss='mse', optimizer=self._optimizer_model)

        self._target_model = Model(inputs=[input_layer], outputs=[output])
        self._target_model.set_weights(self._model.get_weights())
        
    # def predict_movement(self, data, epsilon, expert_action=None, batch_size=None, training=False):
    def predict_movement(self, data, epsilon, env=None, obs=None, overload=False, batch_size=None, training=False):
        """
        Predict movement of game controler where is epsilon probability randomly move."""
        if batch_size is None:
            batch_size = data.shape[0]
        
        # exploit
        q_actions = self._model(data, training=training).numpy()
        opt_policy = np.argmax(np.abs(q_actions), axis=-1)
        # a heuristic to take no action if rho is less than 1 (no overloading)
            
        if not(overload) and (training is False):
            opt_policy = 0
            
        self._exploit_num += 1
            
        if epsilon > 0:
            rand_val = np.random.random()
            if (rand_val < epsilon):
                # Explore
                
                # Explore the whole action space in the case where there is no topology based actions
                # expert_action_index = [0] # DoNothing
                opt_policy = np.random.randint(0, self._action_size, size=1)
                self._explore_total_num += 1
                self._exploit_num -= 1
    
        return opt_policy, q_actions[0, opt_policy], q_actions
    
    def save_tensorboard(self, step_tb):
        pass
