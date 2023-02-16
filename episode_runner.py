from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import pickle
from pandemic_env.environment import PandemicEnv
from pandemic_env.metrics import Metrics
import matplotlib.pylab as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.legacy import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint , TrainEpisodeLogger

import mlflow, mlflow.keras
import json
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

class ep_run():
  def __init__(self, 
              m=50,
              n=50,
              lr = 3e-4,
              weight_vac=0.05, 
              weight_inf=0.1, 
              weight_recov=0.5, 
              seed_strategy=4, 
              cost_vaccine=10, 
              cost_infection=1000, 
              cost_recover=0.1, 
              lockdown_cost=10000, 
              transmission_rate=0.5,
              sensitivity=3,
              reward_factor= 2,
              plot_title=None,
              train=True):
    
    self.env = PandemicEnv(m=m, 
                  n=n, 
                  weight_vac=weight_vac, 
                  weight_inf=weight_inf, 
                  weight_recov=weight_recov, 
                  seed_strategy=seed_strategy, 
                  cost_vaccine=cost_vaccine, 
                  cost_infection=cost_infection, 
                  cost_recover=cost_recover, 
                  lockdown_cost=lockdown_cost, 
                  transmission_rate=transmission_rate,
                  sensitivity=sensitivity,
                  reward_factor=reward_factor,
                  plot_title=plot_title,
                  train=train)

    self.states = self.env.observation_space.shape

    # if action_space is not None:
    #   self.env.action_space = action_space

    self.actions = self.env.action_space.n
    self.EXPERIMENT_NAME = "rl-training"

    try:
        self.EXPERIMENT_ID = mlflow.create_experiment(self.EXPERIMENT_NAME)

    except:
        experiment = mlflow.get_experiment_by_name(self.EXPERIMENT_NAME)
        self.EXPERIMENT_ID = experiment.experiment_id   

  def build_model(self, states, actions):
      """
      Build a sequential model.
      """
      model = Sequential()
      model.add(Flatten(input_shape=(1,) + self.states))
      model.add(Dense(32, activation='relu'))
      model.add(Dense(2, activation='linear'))
      return model

  def build_agent(self, model, actions):
      """
      Build a DQN Agent.
      """
      policy = BoltzmannQPolicy()

      memory = SequentialMemory(limit=50000, window_length=1)
      dqn = DQNAgent(model=model, 
                    nb_actions=actions, 
                    memory=memory, 
                    target_model_update=1e-2, 
                    policy=policy,
                    nb_steps_warmup=35,
                    enable_double_dqn=True)
      return dqn

  def build_model_agent(self, lr, num_steps, verbose =2, save_weights=True):
    """
    Build DQN model and agent.
    """
    model = self.build_model(self.states, self.actions)
    print(model.summary())

    dqn = self.build_agent(model, self.actions)
    dqn.compile(Adam(learning_rate=lr), metrics=['mae'])

    metrics = Metrics(dqn)

    log_filename = 'dqn_logs.json'
    checkpoint_weights_filename = 'dqn_weights_ckpt.h5f'
    # callbacks = [TrainEpisodeLogger()]
    # tensorboard_callback = [TensorBoard(log_dir="./logs")]

    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=500)]
    callbacks += [FileLogger(log_filename, interval=1)]
    reward = dqn.fit(self.env, callbacks=callbacks, nb_steps=num_steps,  visualize=False, verbose=verbose)

    if save_weights:
      weights_filename = 'dqn_weights.h5f'
      dqn.save_weights(weights_filename, overwrite=True)

    return reward, dqn

  def run_experimentation(self, run_name, lr = 1e-3, num_steps=1000):
    """
    Run experimentation to be logged in MLFlow.
    """
    # start mlflow experiment run
    with mlflow.start_run(experiment_id=self.EXPERIMENT_ID, run_name=run_name):
      reward, dqn = self.build_model_agent(lr = lr, num_steps=num_steps)
      pickle.dump( reward.history, open( "reward_history.p", "wb" ) )

      mlflow.log_artifact("reward_history.p")
      mlflow.keras.log_model(dqn.model, "model")
      
      # Plot rewards
      fig, ax = plt.subplots()
      ax.plot(reward.history['episode_reward'])
      plt.xlabel("Episode")
      plt.ylabel("Reward")
      plt.title("Training rewards")
    
      # log the plot and log it as a figure
      plt.savefig("train_rewards-plot.png", dpi=400)
      mlflow.log_figure(fig, "train_rewards-plot.png") 
      
      return reward, dqn

  def test(self, env, experiment_type, num_episodes):
    """
    Test the model manually.
    """

    rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps=7
        counter=0
        while not done:
          if experiment_type == "lockdown":
            action = 1
          elif experiment_type == "no lockdown":
            action = 0
          else:
            # for every weekly
            if counter % (2 * steps) < steps:
              action = 0
            else:
              action = 1
          counter += 1
          obs, reward, done, info = env.step(action)
          total_reward += reward
        rewards.append(total_reward)

    return rewards

# # training
# reward, dqn = ep_run(plot_title="Training", train=True).run_experimentation(run_name = "RUN_NAME", lr=1e-3, num_steps=20000)

# # testing
# test_ep_runner = ep_run(action_space=Discrete(1), plot_title="continuous restrictions", train=False) # lockdown
# test_reward = test_ep_runner.test(test_ep_runner.env, experiment_type="lockdown", num_episodes=1)