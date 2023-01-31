from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
from pandemic_env.environment import PandemicEnv
from pandemic_env.metrics import Metrics
import matplotlib.pylab as plt

# lattice size
m=50
n=50
lr = 3e-4

env = PandemicEnv(m=m, 
                  n=n, 
                  weight_vac=0.05, 
                  weight_inf=0.1, 
                  weight_recov=0.5, 
                  seed_strategy=4, 
                  cost_vaccine=10, 
                  cost_infection=1000, 
                  cost_recover=0, 
                  lockdown_cost=10000, 
                  transmission_rate=0.5,
                  reward_type= 2)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Activation
from tensorflow.keras.optimizers import Adam

states = env.observation_space.shape
actions = env.action_space.n
print(states)
print(actions)

def build_model(states, actions):
    model = Sequential()
    #model.add(Dense(24, activation='relu', input_dim=states[0]))
    # model.add(Reshape((2500,), input_shape=(1,m,n)))
    # model.add(Dense(2500, activation='relu'))
    # model.add(Dense(2500, activation='relu'))
    # model.add(Dense(actions, activation='linear'))

    model.add(Flatten(input_shape=(1,) + states))
    model.add(Dense(24))
    model.add(Activation('relu'))
    model.add(Dense(24))
    model.add(Activation('relu'))
    model.add(Dense(24))
    model.add(Activation('relu'))
    model.add(Dense(actions, activation=''))
    return model

model = build_model(states, actions)
model.summary()


del model

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

model = build_model(states, actions)

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    # dqn = DQNAgent(model=model, memory=memory, policy=policy,
    #               nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    dqn = DQNAgent(model=model, 
                nb_actions=actions, 
                memory=memory, 
            #    enable_dueling_network=True, 
            #    dueling_type='avg', 
               target_model_update=1e-2, 
               policy=policy,
               train_interval=4,
               delta_clip=1.,
               enable_double_dqn=True)
    dqn.compile(Adam(learning_rate=lr), metrics=['mae'])
    return dqn

dqn = build_agent(model, actions)
# dqn.compile(Adam(lr=1e-3), metrics=['mae'])

metrics = Metrics(dqn)
weights_filename = 'dqn_weights.h5f'
# checkpoint_weights_filename = 'dqn_weights_ckpt.h5f'
# log_filename = 'dqn_log.json'
# callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=2500)]
# callbacks += [FileLogger(log_filename, interval=100)]

reward = dqn.fit(env, callbacks=[metrics], nb_steps=50000,  visualize=False, verbose=2)

train_rewards = reward.history['episode_reward']
print(train_rewards)
plt.plot(train_rewards)
plt.ylabel('Train rewards')
plt.show()
plt.savefig('Result2.png')

np.savetxt('reward_history.txt', reward.history, fmt="%s")
dqn.save_weights(weights_filename, overwrite=True)

scores = dqn.test(env, nb_episodes=10, visualize=False)
print(np.mean(scores.history['episode_reward']))


