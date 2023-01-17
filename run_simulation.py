from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
from pandemic_env.environment import PandemicEnv
import matplotlib.pylab as plt

m = 100
n = 100

env = PandemicEnv(m=m, 
                  n=n, 
                  weight_vac=0.05, 
                  weight_inf=0.1, 
                  weight_recov=0.5, 
                  seed_strategy=4, 
                  cost_vaccine=10, 
                  cost_infection=1000, 
                  cost_recover=10, 
                  lockdown_cost=10000, 
                  transmission_rate=0.5)

# print(env.action_space.sample())

# episodes = 1
# for episode in range(1, episodes + 1):
#     state = env.reset()
#     done = False
#     score = 0
#
#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         n_state, reward, done, info = env.step(action)
#         score += reward
#     print('Episode:{} Score:{}'.format(episode, score))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.optimizers import Adam

states = env.observation_space.shape
actions = env.action_space.n
print(states)
print(actions)

def build_model(states, actions):
    model = Sequential()
    #model.add(Dense(24, activation='relu', input_dim=states[0]))
    model.add(Reshape((m*n,), input_shape=(1,m,n)))
    model.add(Dense(2500, activation='relu'))
    model.add(Dense(2500, activation='relu'))
    model.add(Dense(actions, activation='linear'))
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
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  nb_actions=actions, nb_steps_warmup=30, target_model_update=1e-2)
    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Save weights
weights_filename = 'dqn_weights.h5f'
checkpoint_weights_filename = 'dqn_weights_ckpt.h5f'
log_filename = 'dqn_log.json'
callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=2500)]
callbacks += [FileLogger(log_filename, interval=100)]
reward = dqn.fit(env, callbacks=callbacks, nb_steps=5000,  visualize=False, verbose=1, log_interval=1000)

# dqn.fit(env, nb_steps=50000, br=best_reward, visualize=False, verbose=1)
train_rewards = reward.history['episode_reward']
print(train_rewards)
plt.plot(train_rewards)
plt.title("Episode rewards")
plt.show()
plt.savefig('Result.png')

np.savetxt('traindata.txt', train_rewards, fmt="%s")

    # After training is done, we save the final weights one more time.
wts = dqn.save_weights(weights_filename, overwrite=True)

scores = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history['episode_reward']))
#train_rewards = scores.history['episode_reward']
#np.savetxt('traindata.txt', train_rewards)

#dqn.save_weights('dqn_weights.h5f', overwrite=True)

