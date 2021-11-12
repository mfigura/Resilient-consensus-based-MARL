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

    def critic_update_compromised(self,states,new_states,compromised_rewards):
        '''
        Stochastic update of the team critic network
        - performs an update of the team critic network
        - evaluates compromised TD targets with a one-step lookahead
        - applies MSE gradients with TD targets as target values
        ARGUMENTS: visited consecutive states, compromised_rewards
                    boolean to reset parameters to prior values
        RETURNS: updated compromised critic hidden and output layer parameters, training loss
        '''
        nV_team = self.critic(new_states).numpy()
        TD_targets_team = compromised_rewards+self.gamma*nV_team
        with tf.GradientTape() as tape:
            V_team = self.critic(states)
            critic_loss_team = self.mse(TD_targets_team,V_team)
        critic_grad = tape.gradient(critic_loss_team,self.critic.trainable_weights)
        self.optimizer_fast.apply_gradients(zip(critic_grad, self.critic.trainable_weights))

        return self.critic.get_weights()

    def critic_update_local(self,states,new_states,local_rewards):
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
        critic_grad_local = tape.gradient(critic_loss_local,self.critic_local.trainable_weights)
        self.optimizer_fast.apply_gradients(zip(critic_grad_local, self.critic_local.trainable_weights))

    def TR_update_compromised(self,states,team_actions,compromised_rewards):
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
        TR_grad = tape.gradient(TR_loss,self.TR.trainable_weights)
        self.optimizer_fast.apply_gradients(zip(TR_grad, self.TR.trainable_weights))

        return self.TR.get_weights()

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
            tape.watch(self.critic.trainable_weights)
            V = self.critic(state)
            critic_loss = self.mse(local_TD_target,V)
        self.critic_grad = tape.gradient(V,self.critic.trainable_weights)
        critic_mse_grad = tape.gradient(critic_loss,self.critic.trainable_weights)

        self.optimizer_fast.apply_gradients(zip(critic_mse_grad, self.critic.trainable_weights))

        return self.critic.get_weights()

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
            tape.watch(self.TR.trainable_weights)
            team_r = self.TR(sa)
            TR_loss = self.mse(local_reward,team_r)
        self.TR_grad = tape.gradient(team_r,self.TR.trainable_weights)
        TR_mse_grad = tape.gradient(TR_loss,self.TR.trainable_weights)

        self.optimizer_fast.apply_gradients(zip(TR_mse_grad, self.TR.trainable_weights))

        return self.TR.get_weights()

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

class Byzantine_CAC_agent():
    '''
    BYZANTINE CONSENSUS ACTOR-CRITIC AGENT
    This is an implementation of the Byzantine consensus actor-critic (BCAC) agent that is trained simultaneously with the resilient
    consensus actor-critic (RCAC) agents. The algorithm is a realization of temporal difference learning with one-step lookahead.
    The BCAC agent receives a local reward, and observes the global state and action. The BCAC agent seeks to maximize its own
    objective function and prevent learning of cooperative agents by attacking individual parameters in the output layers of their
    team-average function approximators. The attacks are causal as the agent has full knowledge of the updated values of the reliable
    agents. The BCAC agents employs neural networks to approximate the actor and critic for its individual training goals. For the
    actor updates, the agents uses the local rewards and critic. The BCAC agent does not apply consensus updates.

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
        self.critic_feature_extractor = keras.Model(inputs=self.critic.inputs,outputs=self.critic.layers[-1].output)
        self.TR_feature_extractor = keras.Model(inputs=self.TR.inputs,outputs=self.TR.layers[-1].output)

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

    def critic_attack(self,state,critic_innodes):
        '''
        A causal attack on the critic parameters
        - selects a neighbor at random and assigns the neighbor's values to its own hidden layer parameters
        - chooses output layer parameters that will be accepted by neighbors and yield the highest possible critic estimate
        ARGUMENTS: critic parameters from neighbors
        RETURNS: critic parameters
        '''
        random_neighbor = np.random.choice(len(critic_innodes))
        self.critic.set_weights(critic_innodes[random_neighbor])
        mean_feature_vector = [np.mean(self.critic_feature_extractor(state).numpy(),axis=0),np.array(1)]
        critic_max =[]

        for i,layer in enumerate(mean_feature_vector):
            critic_vector = np.stack([item[-2+i]*layer for item in critic_innodes])
            critic_max.append(np.amax(critic_vector, axis=0))
        self.critic.layers[-1].set_weights(critic_max)

        return self.critic.get_weights()

    def TR_attack(self,state,team_action,TR_innodes):
        '''
        A causal attack on the team reward parameters
        - selects a neighbor at random and assigns the neighbor's values to its own hidden layer parameters
        - chooses output layer parameters that will be accepted by neighbors and yield the lowest TR estimate
        ARGUMENTS: TR parameters from neighbors
        RETURNS: TR parameters
        '''
        random_neighbor = np.random.choice(len(TR_innodes))
        self.TR.set_weights(TR_innodes[random_neighbor])
        sa = np.concatenate((state,team_action),axis=1)
        mean_feature_vector = [np.mean(self.TR_feature_extractor(sa).numpy(),axis=0),np.array(1)]
        TR_min =[]

        for i,layer in enumerate(mean_feature_vector):
            TR_vector = np.stack([item[-2+i]*layer for item in TR_innodes])
            print(TR_vector)
            TR_min.append(np.amin(TR_vector, axis=0))
        self.TR.layers[-1].set_weights(TR_min)

        return self.TR.get_weights()

    def critic_set_weights(self,critic_weights_coop):
        'Assigns some neighbors parameters to the critic and returns the hidden parameters'
        random_neighbor = np.random.choice(len(critic_weights_coop))
        self.critic.set_weights(critic_weights_coop[random_neighbor])

        return self.critic.get_weights()

    def TR_set_weights(self,TR_weights_coop):
        'Assigns some neighbors parameters to the TR and returns the hidden parameters'
        random_neighbor = np.random.choice(len(TR_weights_coop))
        self.TR.set_weights(TR_weights_coop[random_neighbor])

        return self.TR.get_weights()

    def critic_update_local(self,states,new_states,local_rewards):
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
        critic_grad_local = tape.gradient(critic_loss_local,self.critic_local.trainable_weights)
        self.optimizer_fast.apply_gradients(zip(critic_grad_local, self.critic_local.trainable_weights))

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
