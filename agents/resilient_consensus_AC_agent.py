import numpy as np
import tensorflow as tf
from tensorflow import keras

class Resilient_CAC_agent():
    '''
    RESILIENT ACTOR-CRITIC AGENT
    This is an implementation of the resilient consensus actor-critic (RCAC) algorithm from Figura et al. (2021). The algorithm
    is a realization of temporal difference learning with one-step lookahead, also known as TD(0). It is an instance
    of decentralized learning, where each agent receives its own reward and observes the global state and action. The RCAC agent
    seeks to maximize a team-average objective function of the cooperative agents in the presence of adversaries. The RCAC agent
    employs neural networks to approximate the actor, critic, and team-average reward function. The current implementation assumes
    that the hidden layer parameters of the critic and team-average reward function are fixed and the same for all agents.

    The updates are divided into four parts.

    1) The agent performs stochastic updates of the critic and team-average reward through methods critic_update_local()
    and TR_update_local(). These methods return the updated parameters but the agent internally keeps its parameters unchanged
    after the execution.

    2) The updated output layer parameters are exchanged between agents that apply projection to estimate the neighbors' rewards.
    The most extreme estimated rewards are eliminated, while the remaining values are averaged. These actions are executed
    through method resilient_consensus().

    3) The agent performs another stochastic update of the critic and team-average reward, this time using the mean estimated
    rewards after truncation. The agent also updates the actor network using the team-average reward and critic in the evaluation
    of the TD error. These actions are covered by methods critic_update_team() and TR_update_team().

    4) The agents exchange the hidden layer parameters and truncate the most extreme parameter values elementwise. The remaining values
    are averaged. These actions are executed by method truncate_and_average_hidden().

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

    def _tf_to_np(self,list_tf):
        '''Converts list of tf variables to a list of np arrays'''
        arr = []
        for node in zip(*list_tf):
            temp = []
            for layer in node:
                temp.append(layer.numpy())
            arr.append(np.array(temp))

        return arr

    def _estimate_error(self,pars_innodes_np,pars_old_np,gradient_np):
        '''
        Estimates the estimation error applied in neighbors' updates based on the received parameter values, its own parameter values
        prior to the stochastic update, and the gradient wrt to the output layer parameters (features). If the passed arguments
        are critic parameters, the agent estimates the TD error. If the arguments are team-average reward parameters, the agent
        estimates a simple error.
        ARGUMENTS: list of np arrays with output layer parameters of neighbors after the stochastic update
                   list of np arrays with output layer parameters of the agent prior to the stochastic update
                   list of np arrays that contain a gradient wrt prior agent's output layer parameters
        RETURNS: np array with estimated TD error of the neighbors
        '''
        grad_norm = 0
        n_agents = pars_innodes_np[0].shape[0]
        magnitude = np.zeros(n_agents)
        for i in range(len(gradient_np)):
            grad_norm += np.sum(gradient_np[i]**2)
            for j in range(n_agents):
                pars_diff = pars_innodes_np[i][j].ravel()-pars_old_np[i].ravel()
                features = gradient_np[i].ravel()
                magnitude[j] += np.dot(pars_diff,features)
        magnitude /= (self.fast_lr*grad_norm)

        return magnitude

    def _truncate_and_average(self,vec):
        '''
        Sorts a vector by value, eliminates H values larger or smaller than the agent's value,
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

    def resilient_consensus_hidden(self,hidden_variables_innodes,hidden_variables_local):
        '''
        Updates hidden layer parameters according to the elementwise resilient averaging rule
        - truncates H smallest and largest parameter values and computes a simple average of the accepted values
        ARGUMENTS: list of np arrays with hidden layer parameters from neighbors,
                   hidden layer parameters of the approximated NN
        '''
        hidden_vars_np = self._tf_to_np(hidden_variables_innodes)

        for i,layer_pars in enumerate(hidden_variables_local):
            sorted_pars = np.sort(hidden_vars_np[i],axis=0)
            truncated_pars = sorted_pars[self.H:sorted_pars.shape[0]-self.H]
            average_pars = np.mean(truncated_pars,axis=0)
            layer_pars.assign(tf.Variable(average_pars))

    def critic_update_team(self,states,new_states,TD_errors):
        '''Stochastic update of the critic using the estimated team-average TD error'''
        team_TD_target = TD_errors + self.critic(states)
        with tf.GradientTape() as tape:
            V = self.critic(states)
            critic_loss = self.mse(team_TD_target,V)
        critic_grad = tape.gradient(critic_loss,self.critic.trainable_variables)

        self.optimizer_fast.apply_gradients(zip(critic_grad, self.critic.trainable_variables))
        critic_hidden_vars = [tf.identity(item) for item in self.critic.trainable_variables[:-2]]

        return critic_hidden_vars,critic_loss

    def TR_update_team(self,states,team_actions,team_errors):
        '''Stochastic update of the team-average reward using the estimated team-average error'''
        sa = np.concatenate((states,team_actions),axis=1)
        tr = team_errors + self.TR(sa)
        with tf.GradientTape() as tape:
            team_r = self.TR(sa)
            TR_loss = self.mse(tr,team_r)
        TR_grad = tape.gradient(TR_loss,self.TR.trainable_variables)

        self.optimizer_fast.apply_gradients(zip(TR_grad, self.TR.trainable_variables))
        TR_hidden_vars = [tf.identity(item) for item in self.TR.trainable_variables[:-2]]

        return TR_hidden_vars,TR_loss

    def actor_update(self,states,new_states,team_actions,local_actions):
        '''
        Stochastic update of the actor network
        - performs a single update of the actor
        - estimates team-average TD errors with a one-step lookahead
        - applies the estimated team-average TD errors as sample weights
          for the cross-entropy gradient
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
        - further computes the critic gradient (later used in the consensus updates)
        - resets the internal critic parameters to the value prior to the stochastic update
        ARGUMENTS: visited consecutive states, local rewards
        RETURNS: updated critic output layer parameters
        '''
        critic_weights_temp = self.critic.get_weights()
        nV = self.critic(new_state).numpy()
        local_TD_target=local_reward+self.gamma*nV
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.critic.trainable_variables)
            V = self.critic(state)
            critic_loss = self.mse(local_TD_target,V)
        self.critic_grad = tape.gradient(V,self.critic.trainable_variables)
        critic_mse_grad = tape.gradient(critic_loss,self.critic.trainable_variables)

        self.optimizer_fast.apply_gradients(zip(critic_mse_grad, self.critic.trainable_variables))
        critic_output_vars = [tf.identity(item) for item in self.critic.trainable_variables[-2:]]
        self.critic.set_weights(critic_weights_temp)

        return critic_output_vars

    def TR_update_local(self,state,team_action,local_reward):
        '''
        Local stochastic update of the team reward network
        - performs a stochastic update of the team-average reward network
        - applies an MSE gradient with a local reward as a target value
        - further computes a gradient of the team reward (later used in the consensus updates)
        - resets the internal team-average reward parameters to the prior value
        ARGUMENTS: visited states, team actions, local rewards
        RETURNS: updated team reward output layer parameters
        '''
        TR_weights_temp = self.TR.get_weights()
        sa = np.concatenate((state,team_action),axis=1)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.TR.trainable_variables)
            team_r = self.TR(sa)
            TR_loss = self.mse(local_reward,team_r)
        self.TR_grad = tape.gradient(team_r,self.TR.trainable_variables)
        TR_mse_grad = tape.gradient(TR_loss,self.TR.trainable_variables)

        self.optimizer_fast.apply_gradients(zip(TR_mse_grad, self.TR.trainable_variables))
        TR_output_vars = [tf.identity(item) for item in self.TR.trainable_variables[-2:]]
        self.TR.set_weights(TR_weights_temp)

        return TR_output_vars

    def resilient_consensus(self,critic_innodes,TR_innodes):
        '''
        Resilient consensus update over the critic and team-average reward parameters
        - projects the received updated parameters into the subspace spanned by the gradient
          to estimate the neighbors' errors
        - removes H values larger and smaller than the agent's error
        - computes a simple average of the accepted estimated errors

        ARGUMENTS: list of model parameters for the critic and team reward received from neighbors
                   (the agent's parameters must appear first in the list followed by its neighbors)
        RETURNS: estimated team-average error in the critic and team-average reward estimation (scalars)
        '''

        '''Convert list of tf variables (model parameters in each layer) to list of numpy arrays'''
        critic_innodes_arr = self._tf_to_np(critic_innodes)
        TR_innodes_arr = self._tf_to_np(TR_innodes)
        critic_old_np = self._tf_to_np([self.critic.trainable_variables[-2:]])
        TR_old_np = self._tf_to_np([self.TR.trainable_variables[-2:]])
        critic_grad_np = self._tf_to_np([self.critic_grad[-2:]])
        TR_grad_np = self._tf_to_np([self.TR_grad[-2:]])

        '''Estimate the estimation error in the neighbors' updates'''
        critic_mag = self._estimate_error(critic_innodes_arr,critic_old_np,critic_grad_np)
        TR_mag = self._estimate_error(TR_innodes_arr,TR_old_np,TR_grad_np)

        '''Truncate and average over accepted values'''
        critic_error_team = self._truncate_and_average(critic_mag)
        TR_error_team = self._truncate_and_average(TR_mag)

        return critic_error_team,TR_error_team

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
