import numpy as np
import tensorflow as tf
from tensorflow import keras

class Faulty_CAC_agent():
    '''
    FAULTY CONSENSUS ACTOR-CRITIC AGENT
    This is an implementation of the faulty consensus actor-critic (FCAC) agent that is trained simultaneously with the resilient
    consensus actor-critic (RCAC) agents. The algorithm is a realization of temporal difference learning with one-step lookahead,
    also known as TD(0). The FCAC agent employs neural networks to approximate the actor, critic, and team-average reward.
    It updates its actor network but does not update the critic and team-average reward parameters. Furthermore, it transmits
    fixed parameter values to the other agents in the network. The FCAC agent does not apply consensus updates. It samples actions
    from the policy approximated by the actor network.

    ARGUMENTS: NN models for actor and critic, and team-average reward
               slow learning rate (for the actor network)
               discount factor gamma
    '''
    def __init__(self,actor,critic,team_reward,slow_lr,gamma=0.95):
        self.actor = actor
        self.critic = critic
        self.TR = team_reward
        self.gamma = gamma
        self.n_actions=self.actor.output_shape[1]

        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=slow_lr),loss=keras.losses.SparseCategoricalCrossentropy())

    def actor_update(self,states,new_states,local_rewards,local_actions):
        '''
        Stochastic update of the actor network
        - performs a single batch update of the actor
        - computes TD errors with a one-step lookahead
        - applies the TD errors as sample weights for the cross-entropy gradient
        ARGUMENTS: visited states, agent's rewards and actions
        RETURNS: training loss
        '''
        V = self.critic(states).numpy()
        nV = self.critic(new_states).numpy()
        local_TD_error=local_rewards+self.gamma*nV-V
        actor_loss = self.actor.train_on_batch(states,local_actions,sample_weight=local_TD_error)

        return actor_loss

    def get_fixed_critic(self):
        '''
        Returns critic hidden and output layer parameters and average loss
        '''
        critic_vars = [tf.identity(item) for item in self.critic.trainable_variables]
        critic_loss = 0

        return critic_vars[:-2], critic_vars[-2:], critic_loss

    def get_fixed_TR(self):
        '''
        Returns team-average reward hidden and output layer parameters and average loss
        '''
        TR_vars = [tf.identity(item) for item in self.TR.trainable_variables]
        TR_loss = 0

        return TR_vars[:-2], TR_vars[-2:], TR_loss

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


class Malicious_CAC_agent():
    '''
    MALICIOUS CONSENSUS ACTOR-CRITIC AGENT
    This is an implementation of the malicious consensus actor-critic (MCAC) agent that is trained simultaneously with the resilient
    consensus actor-critic (RCAC) agents. The algorithm is a realization of temporal difference learning with one-step lookahead,
    also known as TD(0). The MCAC agent receives both local and compromised team reward, and observes the global state and action.
    The adversary seeks to maximize its own objective function and minimize the average objective function of the remaining agents.
    The MCAC agent employs neural networks to approximate the actor and critic. It trains the actor, local critic, compromised team
    critic, and compromised team reward. For the actor updates, the agents uses local rewards and critic. The MCAC agent does not
    apply consensus updates but transmits the compromised critic and team reward parameters.

    ARGUMENTS: NN models for actor and critic, and team reward
               slow learning rate (for the actor network)
               fast learning rate (for the critic and team reward networks)
               discount factor gamma
    '''
    def __init__(self,actor,critic_local,critic,team_reward,slow_lr,fast_lr,gamma=0.95):
        self.actor = actor
        self.critic_local = critic_local
        self.critic = critic
        self.TR = team_reward
        self.gamma = gamma
        self.n_actions=self.actor.output_shape[1]

        self.fast_lr = fast_lr
        self.optimizer_fast=keras.optimizers.SGD(learning_rate=fast_lr)
        self.mse = keras.losses.MeanSquaredError()
        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=slow_lr),loss=keras.losses.SparseCategoricalCrossentropy())

    def actor_update(self,states,new_states,local_rewards,local_actions):
        '''
        Stochastic update of the actor network
        - performs a single batch update of the actor
        - computes TD errors with a one-step lookahead
        - applies the TD errors as sample weights for the cross-entropy gradient
        ARGUMENTS: visited states, agent's rewards and actions
        RETURNS: training loss
        '''
        V = self.critic_local(states).numpy()
        nV = self.critic_local(new_states).numpy()
        local_TD_error=local_rewards+self.gamma*nV-V
        actor_loss = self.actor.train_on_batch(states,local_actions,sample_weight=local_TD_error)

        return actor_loss

    def critic_update_compromised(self,states,new_states,compromised_rewards,reset=False):
        '''
        Stochastic update of the team critic network
        - performs an update of the team critic network
        - evaluates compromised TD targets with a one-step lookahead
        - applies MSE gradients with TD targets as target values
        ARGUMENTS: visited consecutive states, compromised_rewards
                    boolean to reset parameters to prior values
        RETURNS: updated compromised critic hidden and output layer parameters, training loss
        '''
        critic_weights_temp = self.critic.get_weights()
        nV_team = self.critic(new_states).numpy()
        TD_targets_team = compromised_rewards+self.gamma*nV_team
        with tf.GradientTape() as tape:
            V_team = self.critic(states)
            critic_loss_team = self.mse(TD_targets_team,V_team)
        critic_grad = tape.gradient(critic_loss_team,self.critic.trainable_variables)
        self.optimizer_fast.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        critic_vars = [tf.identity(item) for item in self.critic.trainable_variables]
        if reset == True:
            self.critic.set_weights(critic_weights_temp)

        return critic_vars[:-2], critic_vars[-2:], critic_loss_team

    def critic_update(self,states,new_states,local_rewards):
        '''
        Local stochastic update of the critic network
        - performs a stochastic update of the critic network using local rewards
        - evaluates a local TD target with a one-step lookahead
        - applies an MSE gradient with the local TD target as a target value
        ARGUMENTS: visited consecutive states, local rewards
        '''
        nV_local = self.critic_local(new_states).numpy()
        TD_targets_local = local_rewards + self.gamma*nV_local
        with tf.GradientTape() as tape:
            V_local = self.critic_local(states)
            critic_loss_local = self.mse(TD_targets_local,V_local)
        critic_grad_local = tape.gradient(critic_loss_local,self.critic_local.trainable_variables)
        self.optimizer_fast.apply_gradients(zip(critic_grad_local, self.critic_local.trainable_variables))

    def TR_update_compromised(self,states,team_actions,compromised_rewards,reset=False):
        '''
        Stochastic update of the team reward network
        - performs a single batch update of the team reward network
        - applies an MSE gradient with compromised rewards as target values
        ARGUMENTS: visited states, team actions, compromised rewards,
                    boolean to reset parameters to prior values
        RETURNS: updated compromised team reward hidden and output layer parameters, training loss
        '''
        TR_weights_temp = self.TR.get_weights()
        sa = np.concatenate((states,team_actions),axis=1)
        with tf.GradientTape() as tape:
            team_r = self.TR(sa)
            TR_loss = self.mse(compromised_rewards,team_r)
        TR_grad = tape.gradient(TR_loss,self.TR.trainable_variables)
        self.optimizer_fast.apply_gradients(zip(TR_grad, self.TR.trainable_variables))

        TR_train_vars = [tf.identity(item) for item in self.TR.trainable_variables]
        if reset == True:
            self.TR.set_weights(TR_weights_temp)

        return TR_train_vars[:-2], TR_train_vars[-2:], TR_loss

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


class Greedy_CAC_agent():
    '''
    GREEDY CONSENSUS ACTOR-CRITIC AGENT
    This is an implementation of the greedy consensus actor-critic (GCAC) agent that is trained simultaneously with the resilient
    consensus actor-critic (RCAC) agents. The algorithm is a realization of temporal difference learning with one-step lookahead,
    also known as TD(0). The GCAC agent receives a local reward, and observes the global state and action. The GCAC agent seeks
    to maximize its own objective function and is oblivious to the remaining agents' objectives. It employs neural networks
    to approximate the actor, critic, and estimated reward function. For the actor updates, the agents uses the local rewards
    and critic. The GCAC agent does not apply consensus updates but transmits its critic and reward function parameters.

    ARGUMENTS: NN models for actor and critic, and team reward
               slow learning rate (for the actor network)
               fast learning rate (for the critic and team reward networks)
               discount factor gamma
    '''

    def __init__(self,actor,critic,team_reward,slow_lr,fast_lr,gamma=0.95):
        self.actor = actor
        self.critic = critic
        self.TR = team_reward
        self.gamma = gamma
        self.n_actions=self.actor.output_shape[1]

        self.fast_lr = fast_lr
        self.optimizer_fast=keras.optimizers.SGD(learning_rate=fast_lr)
        self.mse = keras.losses.MeanSquaredError()
        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=slow_lr),loss=keras.losses.SparseCategoricalCrossentropy())

    def actor_update(self,states,new_states,local_rewards,local_actions):
        '''
        Stochastic update of the actor network
        - performs a single update of the actor
        - computes TD errors with a one-step lookahead
        - applies the TD errors as sample weights for the cross-entropy gradient
        ARGUMENTS: visited states, local rewards and actions
        RETURNS: training loss
        '''

        V = self.critic(states).numpy()
        nV = self.critic(new_states).numpy()
        global_TD_error=local_rewards+self.gamma*nV-V
        actor_loss = self.actor.train_on_batch(states,local_actions,sample_weight=global_TD_error)

        return actor_loss

    def critic_update_local(self,state,new_state,local_reward,reset=False):
        '''
        Local stochastic update of the critic network
        - performs a stochastic update of the critic network using local rewards
        - evaluates a local TD target with a one-step lookahead
        - applies an MSE gradient with the local TD target as a target value
        - resets the internal critic parameters to the value prior to the stochastic update
        ARGUMENTS: visited consecutive states, local rewards,
                    boolean to reset parameters to prior values
        RETURNS: updated critic parameters (hidden and output) and training loss
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
        critic_vars = [tf.identity(item) for item in self.critic.trainable_variables]
        if reset == True:
            self.critic.set_weights(critic_weights_temp)

        return critic_vars[:-2],critic_vars[-2:],critic_loss

    def TR_update_local(self,state,team_action,local_reward,reset=False):
        '''
        Local stochastic update of the team reward network
        - performs a stochastic update of the team-average reward network
        - applies an MSE gradient with a local reward as a target value
        - further computes a gradient of the team reward (later used in the consensus updates)
        - resets the internal team-average reward parameters to the prior value
        ARGUMENTS: visited states, team actions, local rewards,
                    boolean to reset parameters to prior values
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
        TR_vars = [tf.identity(item) for item in self.TR.trainable_variables]
        if reset == True:
            self.TR.set_weights(TR_weights_temp)

        return TR_vars[:-2],TR_vars[-2:],TR_loss

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
