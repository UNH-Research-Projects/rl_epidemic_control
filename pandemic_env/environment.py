import gym
from gym import spaces
import player.create_players as cp
import numpy as np
import matplotlib.pyplot as plt


class PandemicEnv(gym.Env):
    """Custom environment for simulating a pandemic.

    This environment follows the gym interface and allows the user to
    control the spread of a pandemic by adjusting the contact rate.

    Parameters:
    m (int): The number of rows in the lattice.
    n (int): The number of columns in the lattice.
    weight_vac (float): The weight for the vaccine strategy.
    weight_inf (float): The weight for the infection strategy.
    weight_recov (float): The weight for the recovery strategy.
    seed_strategy (int): The initial strategy for the seed player.
    cost_vaccine (float): The cost of using the vaccine strategy.
    cost_infection (float): The cost of using the infection strategy.
    cost_recover (float): The cost of using the recovery strategy.
    lockdown_cost (float): The cost of implementing a lockdown.

    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        m,
        n,
        weight_vac,
        weight_inf,
        weight_recov,
        seed_strategy,
        cost_vaccine,
        cost_infection,
        cost_recover,
        lockdown_cost,
    ):
        # super(PandemicEnv, self).__init__()
        self.m = m
        self.n = n
        # Define action and observation space
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(self.m, self.n), dtype=np.int8
        )
        self.players_lattice = cp.CreatePlayers(
            self.m,
            self.n,
            weight_vac,
            weight_inf,
            weight_recov,
            seed_strategy,
            cost_vaccine,
            cost_infection,
            cost_recover,
            lockdown_cost,
        )
        self.players_lattice.get_strategy()

        # self.players_lattice.get_sensitivity_transmission_rate()
        self.pandemic_length = 0

        self.infected_num_list = []
        self.reward_list = []

    def step(self, action):
        """Execute one time step within the environment.

        Parameters:
        action (int): The action to be taken.
            0: Increase contact rate.
            1: Decrease contact rate.

        Returns:
        tuple: A tuple containing the observation, reward, done flag, and metadata.
        """

        iteration = 1
        contact_rate = self.take_action(action)
        pandemic_time = self.pandemic_length
        self.players_lattice.update_lattice(iteration, contact_rate, pandemic_time)
        state = self.players_lattice.build_matrix_strategy(pandemic_time)
        self.pandemic_length += iteration

        num_infected = self.players_lattice.count_num_strategy(2)
        self.infected_num_list.append(num_infected)

        print("Infections for step {}: {} ".format(self.pandemic_length, num_infected))
        reward = self.players_lattice.calc_reward(contact_rate)

        self.reward_list.append(reward)
        # if self.pandemic_length >= 100:
        # if self.players_lattice.count_num_strategy(2) <= 0.01*(self.m*self.n):
        if num_infected <= 0:
            done = True
        else:
            done = False
        # obs = np.append(state)
        # obs = np.append(state, axis=0)
        obs = state

        return obs, reward, done, {}

    def take_action(self, action):
        """Adjust the contact rate based on the given action.

        Parameters:
        action (int): The action to be taken.
            0: Increase contact rate.
            1: Decrease contact rate.

        Returns:
        float: The new contact rate.
        """
        print(action)
        if action == 0:
            contact_rate = 1
        elif action == 1:
            contact_rate = 0.5
        return contact_rate

    def reset(self):
        """Reset the environment and return the initial state.

        Returns:
        np.ndarray: An m x n array representing the initial state of the lattice.
        """
        plt.plot(self.infected_num_list)
        plt.show()

        plt.plot(self.reward_list)
        plt.title("Rewards")
        plt.show()

        self.players_lattice.state_zero()
        state = self.players_lattice.build_matrix_strategy(0)
        self.pandemic_length = 0
        return state

    def render(self, mode="human", close=False):
        """
        Render the current state of the lattice.
        """
        self.players_lattice.draw_matrix_strategy(-1)
