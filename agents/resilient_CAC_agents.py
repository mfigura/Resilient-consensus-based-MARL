import numpy as np
import tensorflow as tf
from tensorflow import keras

class RPBCAC_agent():
    '''
    RESILIENT PROJECTION-BASED CONSENSUS ACTOR-CRITIC AGENT
    This is an implementation of the resilient projection-based consensus actor-critic (RPBCAC) algorithm from Figura et al. (2021).
    The algorithm is a realization of temporal difference learning with one-step lookahead. It is an instance of decentralized learning,
    where each agent receives its own reward and observes the global state and action. The RPBCAC agent seeks to maximize a team-average
    objective function of the cooperative agents in the presence of adversaries. The RPBCAC agent employs neural networks to approximate
    the actor, critic, and team-average reward function.

    The updates are divided into four parts.

    1) The agent performs a batch local stochastic update of the critic and team-average reward through methods critic_update_local()
    and TR_update_local().

    2) The agent receives parameters from its neighbors and computes an element-wise trimmed mean over hidden layer parameters.

    3) The agent estimates the neighbors' estimation errors via projection and applies the resilient projection-based consensus update.

    4) The agent performs a stochastic update using the mean estimation errors. This is executed by methods critic_update_team() and TR_update_team().

    The code is applicable for both online and batch training. The RCAC agent further includes method get_action() to sample actions
    from the policy approximated by the actor network.

    ARGUMENTS: NN models for actor, critic, and team_reward
               slow learning rate (for the actor network)
               fast learning rate (for the critic and team reward networks)
               discount factor gamma
               max number of adversaries among the agent's neighbors
    '''
    def __init__(self,actor,critic,team_reward,slow_lr,fast_lr,gamma=0.95, H=0):
        self.actor = actor
        self.critic = critic
        self.TR = team_reward
        self.gamma = gamma
        self.H = H
        self.n_actions=self.actor.output_shape[1]

        self.fast_lr = fast_lr
        self.optimizer_fast=keras.optimizers.SGD(learning_rate=fast_lr)
        self.mse = keras.losses.MeanSquaredError()
        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=slow_lr),loss=keras.losses.SparseCategoricalCrossentropy())
        self.critic_feature_extractor = keras.Model(inputs=self.critic.inputs,outputs=self.critic.layers[-2].output)
        self.TR_feature_extractor = keras.Model(inputs=self.TR.inputs,outputs=self.TR.layers[-2].output)

    def _transpose_list(self,weights):
        '''Transpose list and stack agents weights'''
        arr = []
        for node in zip(*weights):
            temp = []
            for layer in node:
                temp.append(layer)
            arr.append(np.array(temp))

        return arr

    def _trimmed_mean_error(self,vec):
        '''
        Sorts a vector by value, eliminates H values strictly larger or smaller than the agent's value,
        and computes an average of the remaining values
        Arguments: np array with values
        Returns: average accepted value
        '''
        vec = vec.ravel()
        own_val = vec[0:1]                      #get own value
        sorted_vals = np.sort(vec[1:])        #sort neighbors' values
        n_innodes = sorted_vals.shape[0]
        H_small = sorted_vals[:self.H]             #get H smallest values
        H_small_valid = H_small[H_small>=own_val[0]]   #get H smallest values that are greater than own value
        H_large = sorted_vals[n_innodes-self.H:]            #repeat for large values
        H_large_valid = H_large[H_large<=own_val[0]]
        H_mid = sorted_vals[self.H:n_innodes-self.H]               #get values in the mid arange

        valid_values = np.concatenate((own_val,H_small_valid,H_mid,H_large_valid))

        return np.mean(valid_values)

    def _trimmed_mean_hidden(self,hidden_weights_innodes,model):
        '''
        Updates hidden layer parameters according to the element-wise trimmed-mean rule
        - truncates H smallest and largest parameter values and computes a simple average of the accepted values
        ARGUMENTS: lists of np arrays with hidden layer parameters from neighbors
        '''

        for i,layer_weights in enumerate(model.trainable_weights[:-2]):
            sorted_weights = np.sort(hidden_weights_innodes[i],axis=0)
            truncated_weights = sorted_weights[self.H:sorted_weights.shape[0]-self.H]
            mean_weights = np.mean(truncated_weights,axis=0)
            layer_weights.assign(tf.Variable(mean_weights))

    def critic_update_team(self,states,new_states,TD_errors):
        '''
        Stochastic update of the critic using the estimated average TD error of the neighbors
        ARGUMENTS: visited consecutive states, neighbors' TD errors
        RETURNS: updated critic parameters
        '''
        team_TD_target = TD_errors + self.critic(states).numpy()
        with tf.GradientTape() as tape:
            V = self.critic(states)
            critic_loss = self.mse(team_TD_target,V)
        critic_grad = tape.gradient(critic_loss,self.critic.trainable_weights[-2:])

        self.optimizer_fast.apply_gradients(zip(critic_grad, self.critic.trainable_weights[-2:]))
        critic_weights = self.critic.get_weights()

        return critic_weights,critic_loss

    def TR_update_team(self,states,team_actions,team_errors):
        '''
        Stochastic update of the team-average reward using the estimated average estimation error of the neighbors
        ARGUMENTS: visited states, team actions, neighbors' estimation errors
        RETURNS: updated team reward parameters
        '''
        sa = np.concatenate((states,team_actions),axis=1)
        tr = team_errors + self.TR(sa).numpy()
        with tf.GradientTape() as tape:
            team_r = self.TR(sa)
            TR_loss = self.mse(tr,team_r)
        TR_grad = tape.gradient(TR_loss,self.TR.trainable_weights[-2:])

        self.optimizer_fast.apply_gradients(zip(TR_grad, self.TR.trainable_weights[-2:]))
        TR_weights = self.TR.get_weights()

        return TR_weights,TR_loss

    def actor_update(self,states,new_states,team_actions,local_actions):
        '''
        Stochastic update of the actor network
        - performs a single update of the actor
        - estimates team-average TD errors with a one-step lookahead
        - applies the estimated team-average TD errors as sample weights
          to the cross-entropy gradient
        ARGUMENTS: visited states, team actions, agent's actions
        RETURNS: training loss
        '''
        sa = np.concatenate((states,team_actions),axis=1)
        team_rewards = self.TR(sa).numpy()

        V = self.critic(states).numpy()
        nV = self.critic(new_states).numpy()
        global_TD_error=team_rewards+self.gamma*nV-V
        actor_loss = self.actor.train_on_batch(states,local_actions,sample_weight=global_TD_error)

        return actor_loss

    def critic_update_local(self,state,new_state,local_reward):
        '''
        Local stochastic update of the critic network
        - performs a stochastic update of the critic network using local rewards
        - evaluates a local TD target with a one-step lookahead
        - applies an MSE gradient with the local TD target as a target value
        - resets the internal critic parameters to the value prior to the stochastic update
        ARGUMENTS: visited consecutive states, local rewards
        RETURNS: updated critic parameters
        '''
        critic_weights_temp = self.critic.get_weights()
        self.critic_features = self.critic_feature_extractor(state)

        nV = self.critic(new_state).numpy()
        local_TD_target=local_reward+self.gamma*nV

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.critic.trainable_weights)
            V = self.critic(state)
            critic_loss = self.mse(local_TD_target,V)
        critic_grad = tape.gradient(critic_loss,self.critic.trainable_weights)

        self.optimizer_fast.apply_gradients(zip(critic_grad, self.critic.trainable_weights))
        critic_weights = self.critic.get_weights()
        self.critic.set_weights(critic_weights_temp)

        return critic_weights, critic_loss

    def TR_update_local(self,state,team_action,local_reward):
        '''
        Local stochastic update of the team reward network
        - performs a stochastic update of the team-average reward network
        - applies an MSE gradient with a local reward as a target value
        - resets the internal team-average reward parameters to the prior value
        ARGUMENTS: visited states, team actions, local rewards
        RETURNS: updated team reward parameters
        '''
        TR_weights_temp = self.TR.get_weights()
        sa = np.concatenate((state,team_action),axis=1)
        self.TR_features = self.TR_feature_extractor(sa)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.TR.trainable_weights)
            team_r = self.TR(sa)
            TR_loss = self.mse(local_reward,team_r)
        TR_grad = tape.gradient(TR_loss,self.TR.trainable_weights)

        self.optimizer_fast.apply_gradients(zip(TR_grad, self.TR.trainable_weights))
        TR_weights = self.TR.get_weights()
        self.TR.set_weights(TR_weights_temp)

        return TR_weights, TR_loss

    def resilient_consensus_critic(self,state,critic_weights_innodes):
        '''
        Resilient consensus update over the critic parameters
        - projects the received updated parameters into the feature vector to estimate the neighbors' errors
        - removes H values larger and smaller than the agent's error
        - computes a simple average of the accepted estimated errors

        ARGUMENTS: list of critic parameters received from neighbors
                   (the agent's parameters must appear first in the list followed by its neighbors)
        RETURNS: trimmed mean critic estimation errors
        '''
        critic_weights_temp = self.critic.get_weights()
        critic_errors = np.zeros((len(critic_weights_innodes),len(state)))

        'Estimation of neighbors estimation errors via projection'
        V = self.critic(state)
        features_dot = tf.norm(self.critic_features).numpy()**2 + 1
        for node,weights in enumerate(critic_weights_innodes):
            self.critic.set_weights(weights)
            critic_errors[node] = ((self.critic(state) - V).numpy()/(2*self.fast_lr*features_dot)).ravel()
        self.critic.set_weights(critic_weights_temp)

        '''Trim extreme values and average accepted values'''
        critic_mean_error = self._trimmed_mean_error(critic_errors)

        return critic_mean_error

    def resilient_consensus_TR(self,state,team_action,TR_weights_innodes):
        '''
        Resilient consensus update over the team reward parameters
        - projects the received updated parameters into the feature subspace to estimate the neighbors' errors
        - removes H values larger and smaller than the agent's error
        - computes a simple average of the accepted estimated errors

        ARGUMENTS: list of TR parameters received from neighbors
                   (the agent's parameters must appear first in the list followed by its neighbors)
        RETURNS: trimmed mean TR estimation errors
        '''
        TR_weights_temp = self.TR.get_weights()
        TR_errors = np.zeros((len(TR_weights_innodes),len(state)))
        sa = np.concatenate((state,team_action),axis=1)

        'Estimation of neighbors estimation errors via projection'
        TR = self.TR(sa)
        features_dot = tf.norm(self.TR_features).numpy()**2 + 1
        for node,weights in enumerate(TR_weights_innodes):
            self.TR.set_weights(weights)
            TR_errors[node] = ((self.TR(sa) - TR).numpy()/(self.fast_lr*features_dot)).ravel()

        self.TR.set_weights(TR_weights_temp)

        '''Trim extreme values and average accepted values'''
        TR_mean_error = self._trimmed_mean_error(TR_errors)

        return TR_mean_error

    def resilient_consensus_hidden(self,critic_hidden_innodes,TR_hidden_innodes):
        '''
        Resilient consensus update over the hidden parameters of the critic and team-average reward
        - removes H largest and smallest received parameter values
        - assigns a simple average of the accepted values to the hidden layers
        ARGUMENTS: list of parameters for the critic and team reward received from neighbors
        RETURNS: estimated team-average error in the critic and team-average reward estimation (scalars)
        '''

        critic_hidden_innodes = self._transpose_list(critic_hidden_innodes)
        TR_hidden_innodes = self._transpose_list(TR_hidden_innodes)

        self._trimmed_mean_hidden(critic_hidden_innodes,self.critic)
        self._trimmed_mean_hidden(TR_hidden_innodes,self.TR)

    def get_action(self,state,from_policy=False,mu=0.1):
        '''Choose an action at the current state
            - set from_policy to True to sample from the actor
            - set from_policy to False to sample from the random uniform distribution over actions
            - set mu to [0,1] to control probability of choosing a random action
        '''
        random_action = np.random.choice(self.n_actions)
        if from_policy==True:
            state = np.array(state).reshape(1,-1)
            action_prob = self.actor.predict(state)
            action_from_policy = np.random.choice(self.n_actions, p = action_prob[0])
            self.action = np.random.choice([action_from_policy,random_action], p = [1-mu,mu])
        else:
            self.action = random_action

        return self.action

class RTMCAC_agent():
    '''
    RESILIENT CONSENSUS ACTOR-CRITIC AGENT
    This is an implementation of the resilient trimmed-mean consensus actor-critic (RTMCAC) algorithm with the idea
    adopted from Wu et al.(2021). The algorithm is a realization of temporal difference learning with one-step lookahead.
    It is an instance of decentralized learning, where each agent receives its own reward and observes the global state
    and action. The RCAC agent seeks to maximize a team-average objective function of the cooperative agents in the presence
    of adversaries. The RCAC agent employs neural networks to approximate the actor, critic, and team-average reward function.

    The updates are divided into two parts.

    1) The agent performs stochastic updates of the critic and team-average reward through methods critic_update_local()
    and TR_update_local(). These methods return the updated parameters but the agent internally keeps its parameters unchanged
    after the execution.

    2) The updated parameters are exchanged between agents that apply projection to estimate the neighbors' rewards.
    The most extreme estimated rewards are eliminated, while the remaining values are averaged. These actions are executed
    through method resilient_consensus().

    The code is applicable for both online and batch training, but the projection step must be executed individually for each step
    in training. The RCAC agent further includes method get_action() to sample actions from the policy approximated by the actor network.

    ARGUMENTS: NN models for actor, critic, and team_reward
               slow learning rate (for the actor network)
               fast learning rate (for the critic and team reward networks)
               discount factor gamma
               max number of adversaries among the agent's neighbors
    '''
    def __init__(self,actor,critic,team_reward,slow_lr,fast_lr,gamma=0.95, H=0):
        self.actor = actor
        self.critic = critic
        self.TR = team_reward
        self.gamma = gamma
        self.H = H
        self.n_actions=self.actor.output_shape[1]

        self.fast_lr = fast_lr
        self.optimizer_fast=keras.optimizers.SGD(learning_rate=fast_lr)
        self.mse = keras.losses.MeanSquaredError()
        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=slow_lr),loss=keras.losses.SparseCategoricalCrossentropy())

    def _transpose_list(self,weights):
        '''Transpose list and stack agents weights'''
        arr = []
        for node in zip(*weights):
            temp = []
            for layer in node:
                temp.append(layer)
            arr.append(np.array(temp))

        return arr

    def _trimmed_mean(self,weights_innodes,model):
        '''
        Updates layer parameters according to the elementwise trimmed-mean rule
        - truncates H smallest and largest parameter values and computes a simple average of the accepted values
        ARGUMENTS: lists of np arrays with hidden layer parameters from neighbors
        '''
        for i,layer_weights in enumerate(model.trainable_weights):
            sorted_weights = np.sort(weights_innodes[i],axis=0)
            truncated_weights = sorted_weights[self.H:sorted_weights.shape[0]-self.H]
            mean_weights = np.mean(truncated_weights,axis=0)
            layer_weights.assign(tf.Variable(mean_weights))

    def actor_update(self,states,new_states,team_actions,local_actions):
        '''
        Stochastic update of the actor network
        - performs a single update of the actor
        - estimates team-average TD errors with a one-step lookahead
        - applies the estimated team-average TD errors as sample weights to the cross-entropy gradient
        ARGUMENTS: visited states, team actions, agent's actions
        RETURNS: training loss
        '''
        sa = np.concatenate((states,team_actions),axis=1)
        team_rewards = self.TR(sa).numpy()

        V = self.critic(states).numpy()
        nV = self.critic(new_states).numpy()
        global_TD_error=team_rewards+self.gamma*nV-V
        actor_loss = self.actor.train_on_batch(states,local_actions,sample_weight=global_TD_error)

        return actor_loss

    def critic_update_local(self,state,new_state,local_reward):
        '''
        Local stochastic update of the critic network
        - performs a stochastic update of the critic network using local rewards
        - evaluates a local TD target with a one-step lookahead
        - applies an MSE gradient with the local TD target as a target value
        - resets the internal critic parameters to the value prior to the stochastic update
        ARGUMENTS: visited consecutive states, local rewards
        RETURNS: updated critic output layer parameters
        '''
        critic_weights_temp = self.critic.get_weights()
        nV = self.critic(new_state).numpy()
        local_TD_target=local_reward+self.gamma*nV
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.critic.trainable_weights)
            V = self.critic(state)
            critic_loss = self.mse(local_TD_target,V)
        critic_grad = tape.gradient(critic_loss,self.critic.trainable_weights)

        self.optimizer_fast.apply_gradients(zip(critic_grad, self.critic.trainable_weights))
        critic_vars = self.critic.get_weights()
        self.critic.set_weights(critic_weights_temp)

        return critic_vars, critic_loss

    def TR_update_local(self,state,team_action,local_reward):
        '''
        Local stochastic update of the team reward network
        - performs a stochastic update of the team-average reward network
        - applies an MSE gradient with a local reward as a target value
        - resets the internal team-average reward parameters to the prior value
        ARGUMENTS: visited states, team actions, local rewards
        RETURNS: updated team reward output layer parameters
        '''
        TR_weights_temp = self.TR.get_weights()
        sa = np.concatenate((state,team_action),axis=1)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.TR.trainable_weights)
            team_r = self.TR(sa)
            TR_loss = self.mse(local_reward,team_r)
        TR_grad = tape.gradient(TR_loss,self.TR.trainable_weights)

        self.optimizer_fast.apply_gradients(zip(TR_grad, self.TR.trainable_weights))
        TR_vars = self.TR.get_weights()
        self.TR.set_weights(TR_weights_temp)

        return TR_vars, TR_loss

    def resilient_consensus(self,critic_innodes,TR_innodes):
        '''
        Resilient consensus update over the critic and team-average reward parameters
        - removes H largest and smallest values from each entry of the parameter vector
        - computes an element-wise trimmed mean

        ARGUMENTS: list of model parameters for the critic and team reward received from neighbors
                   (the agent's parameters must appear first in the list followed by its neighbors)
        RETURNS: estimated team-average error in the critic and team-average reward estimation (scalars)
        '''
        critic_innodes = self._transpose_list(critic_innodes)
        TR_innodes = self._transpose_list(TR_innodes)

        self._trimmed_mean(critic_innodes,self.critic)
        self._trimmed_mean(TR_innodes,self.TR)

    def get_action(self,state,from_policy=False,mu=0.1):
        '''Choose an action at the current state
            - set from_policy to True to sample from the actor
            - set from_policy to False to sample from the random uniform distribution over actions
            - set mu to [0,1] to control probability of choosing a random action
        '''
        random_action = np.random.choice(self.n_actions)
        if from_policy==True:
            state = np.array(state).reshape(1,-1)
            action_prob = self.actor.predict(state)
            action_from_policy = np.random.choice(self.n_actions, p = action_prob[0])
            self.action = np.random.choice([action_from_policy,random_action], p = [1-mu,mu])
        else:
            self.action = random_action

        return self.action
