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

    def actor_update(self,s,ns,r_local,a_local):
        '''
        Stochastic update of the actor network
        - performs a single update of the actor
        - estimates team-average TD errors with a one-step lookahead
        - applies the estimated team-average TD errors as sample weights to the cross-entropy gradient
        ARGUMENTS: observations, new observations, state-actions pairs, local actions
        RETURNS: training loss
        '''
        V = self.critic(s)
        nV = self.critic(ns)
        global_TD_error = r_local + self.gamma * nV - V
        actor_loss = self.actor.train_on_batch(s,a_local,sample_weight=global_TD_error)

        return actor_loss

    def get_critic_weights(self):
        '''
        Returns critic parameters and average loss
        '''
        return self.critic.get_weights()

    def get_TR_weights(self):
        '''
        Returns team-average reward parameters
        '''
        return self.TR.get_weights()

    def get_action(self,state,from_policy=False,mu=0.1):
        '''Choose an action at the current state
            - set from_policy to True to sample from the actor
            - set from_policy to False to sample from the random uniform distribution over actions
            - set mu to [0,1] to control probability of choosing a random action
        '''
        random_action = np.random.choice(self.n_actions)
        action_prob = self.actor.predict(state).ravel()
        action_from_policy = np.random.choice(self.n_actions, p = action_prob)
        self.action = np.random.choice([action_from_policy,random_action], p = [1-mu,mu])

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

        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=slow_lr),loss=keras.losses.SparseCategoricalCrossentropy())
        self.critic_local.compile(optimizer=keras.optimizers.SGD(learning_rate=fast_lr),loss=keras.losses.MeanSquaredError())
        self.critic.compile(optimizer=keras.optimizers.SGD(learning_rate=fast_lr),loss=keras.losses.MeanSquaredError())
        self.TR.compile(optimizer=keras.optimizers.SGD(learning_rate=fast_lr),loss=keras.losses.MeanSquaredError())

    def actor_update(self,s,ns,r_local,a_local):
        '''
        Stochastic update of the actor network
        - performs a single update of the actor
        - estimates team-average TD errors with a one-step lookahead
        - applies the estimated team-average TD errors as sample weights to the cross-entropy gradient
        ARGUMENTS: observations, new observations, state-actions pairs, local actions
        RETURNS: training loss
        '''
        V = self.critic(s)
        nV = self.critic(ns)
        global_TD_error = r_local + self.gamma * nV - V
        actor_loss = self.actor.train_on_batch(s,a_local,sample_weight=global_TD_error)

        return actor_loss

    def critic_update_compromised(self,s,ns,r_compromised):
        '''
        Stochastic update of the team critic network
        - performs an update of the team critic network
        - evaluates compromised TD targets with a one-step lookahead
        - applies MSE gradients with TD targets as target values
        ARGUMENTS: visited consecutive states, compromised_rewards
                    boolean to reset parameters to prior values
        RETURNS: updated compromised critic hidden and output layer parameters, training loss
        '''
        nV = self.critic(ns)
        TD_target_compromised = r_compromised + self.gamma * nV
        loss = self.critic.train_on_batch(s,TD_target_compromised)

        return self.critic.get_weights(), loss

    def critic_update_local(self,s,ns,r_local):
        '''
        Local stochastic update of the critic network
        - performs a stochastic update of the critic network using local rewards
        - evaluates a local TD target with a one-step lookahead
        - applies an MSE gradient with the local TD target as a target value
        ARGUMENTS: visited consecutive states, local rewards
        RETURNS: updated critic parameters
        '''
        nV = self.critic(ns)
        local_TD_target = r_local + self.gamma * nV
        self.critic_local.train_on_batch(s,local_TD_target)

    def TR_update_compromised(self,sa,r_compromised):
        '''
        Stochastic update of the team reward network
        - performs a single batch update of the team reward network
        - applies an MSE gradient with compromised rewards as target values
        ARGUMENTS: visited states, team actions, compromised rewards,
                    boolean to reset parameters to prior values
        RETURNS: updated compromised team reward hidden and output layer parameters, training loss
        '''
        loss = self.TR.train_on_batch(sa,r_compromised)

        return self.TR.get_weights(), loss

    def get_action(self,state,from_policy=False,mu=0.1):
        '''Choose an action at the current state
            - set from_policy to True to sample from the actor
            - set from_policy to False to sample from the random uniform distribution over actions
            - set mu to [0,1] to control probability of choosing a random action
        '''
        random_action = np.random.choice(self.n_actions)
        action_prob = self.actor.predict(state).ravel()
        action_from_policy = np.random.choice(self.n_actions, p = action_prob)
        self.action = np.random.choice([action_from_policy,random_action], p = [1-mu,mu])

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

        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=slow_lr),loss=keras.losses.SparseCategoricalCrossentropy())
        self.TR.compile(optimizer=keras.optimizers.SGD(learning_rate=fast_lr),loss=keras.losses.MeanSquaredError())
        self.critic.compile(optimizer=keras.optimizers.SGD(learning_rate=fast_lr),loss=keras.losses.MeanSquaredError())

    def actor_update(self,s,ns,r_local,a_local):
        '''
        Stochastic update of the actor network
        - performs a single update of the actor
        - computes TD errors with a one-step lookahead
        - applies the TD errors as sample weights for the cross-entropy gradient
        ARGUMENTS: visited states, local rewards and actions
        RETURNS: training loss
        '''

        V = self.critic(s)
        nV = self.critic(ns)
        print(r_local.shape,V.shape)
        TD_error = r_local + self.gamma * nV - V
        actor_loss = self.actor.train_on_batch(s,a_local,sample_weight=TD_error)

        return actor_loss

    def critic_update_local(self,s,ns,r_local):
        '''
        Local stochastic update of the critic network
        - performs a stochastic update of the critic network using local rewards
        - evaluates a local TD target with a one-step lookahead
        - applies an MSE gradient with the local TD target as a target value
        ARGUMENTS: visited consecutive states, local rewards
        RETURNS: updated critic parameters
        '''
        nV = self.critic(ns)
        local_TD_target = r_local + self.gamma * nV
        critic_loss = self.critic.train_on_batch(s,local_TD_target)

        return self.critic.get_weights(), critic_loss

    def TR_update_local(self,sa,r_local):
        '''
        Local stochastic update of the team reward network
        - performs a stochastic update of the team-average reward network
        - applies an MSE gradient with a local reward as a target value
        ARGUMENTS: state-action pairs, local rewards
        RETURNS: updated team reward parameters
        '''
        TR_loss = self.TR.train_on_batch(sa,r_local)

        return self.TR.get_weights(), TR_loss

    def get_action(self,state,from_policy=False,mu=0.1):
        '''Choose an action at the current state
            - set from_policy to True to sample from the actor
            - set from_policy to False to sample from the random uniform distribution over actions
            - set mu to [0,1] to control probability of choosing a random action
        '''
        '''Choose an action at the current state
            - set from_policy to True to sample from the actor
            - set from_policy to False to sample from the random uniform distribution over actions
            - set mu to [0,1] to control probability of choosing a random action
        '''
        random_action = np.random.choice(self.n_actions)
        action_prob = self.actor.predict(state).ravel()
        action_from_policy = np.random.choice(self.n_actions, p = action_prob)
        self.action = np.random.choice([action_from_policy,random_action], p = [1-mu,mu])

        return self.action
