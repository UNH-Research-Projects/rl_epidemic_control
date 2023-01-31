from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
from pandemic_env.environment import PandemicEnv
from pandemic_env.metrics import Metrics
import matplotlib.pylab as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Activation
from tensorflow.keras.optimizers import Adam, RMSprop

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

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
              cost_recover=0, 
              lockdown_cost=10000, 
              transmission_rate=0.5,
              sensitivity=3,
              reward_type= 2):
    
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
                  reward_type=reward_type)

    self.states = self.env.observation_space.shape
    self.actions = self.env.action_space.n


  def build_model(self, states, actions):
      model = Sequential()
      #model.add(Dense(24, activation='relu', input_dim=states[0]))
      # model.add(Reshape((2500,), input_shape=(1,m,n)))
      # model.add(Dense(2500, activation='relu'))
      # model.add(Dense(2500, activation='relu'))
      # model.add(Dense(actions, activation='linear'))

      model.add(Flatten(input_shape=(1,) + self.states))
      model.add(Dense(64))
      model.add(Activation('relu'))
      model.add(Dense(64))
      model.add(Activation('relu'))
      model.add(Dense(64))
      model.add(Activation('relu'))
      model.add(Dense(actions, activation='linear'))
      return model

  def build_agent(self, model, actions):
      policy = BoltzmannQPolicy()
      memory = SequentialMemory(limit=50000, window_length=1)
      dqn = DQNAgent(model=model, 
                  nb_actions=actions, 
                  memory=memory, 
              #    enable_dueling_network=True, 
              #    dueling_type='avg', 
                target_model_update=1e-2, 
                policy=policy,
                nb_steps_warmup=35,
                # train_interval=4,
                # delta_clip=1.,
                enable_double_dqn=True)
      dqn.compile(Adam(3e-4), metrics=['mae'])
      return dqn

  def build_model_agent(self, lr, num_steps, verbose =2, save_weights=True):
    model = self.build_model(self.states, self.actions)
    print(model.summary())

    dqn = self.build_agent(model, self.actions)
    dqn.compile(Adam(lr=lr), metrics=['mae'])

    metrics = Metrics(dqn)
    reward = dqn.fit(self.env, callbacks=[metrics], nb_steps=num_steps,  visualize=False, verbose=verbose)

    if save_weights:
      weights_filename = 'dqn_weights.h5f'
      dqn.save_weights(weights_filename, overwrite=True)

    return reward

  def run_experimentation(self, model_name, lr = 1e-4, num_steps=1000,):
    with mlflow.start_run():
      reward = self.build_model_agent(lr = lr, num_steps=num_steps)
      mlflow.log_param("reward_history", reward.history)
      # mlflow.log_param("strategy_history", l1_ratio)
      # mlflow.log_metric("metrics", metrics)

      tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

      # Model registry does not work with file store
      if tracking_url_type_store != "file":

          # Register the model
          # There are other ways to use the Model Registry, which depends on the use case,
          # please refer to the doc for more information:
          # https://mlflow.org/docs/latest/model-registry.html#api-workflow
          mlflow.sklearn.log_model(lr, "model", registered_model_name=model_name)
      else:
          mlflow.sklearn.log_model(lr, "model")

# checkpoint_weights_filename = 'dqn_weights_ckpt.h5f'
# log_filename = 'dqn_log.json'
# callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=2500)]
# callbacks += [FileLogger(log_filename, interval=100)]
