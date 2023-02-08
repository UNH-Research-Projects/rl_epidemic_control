import gym
from gym import spaces
import players.create_players as cp
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
        transmission_rate,
        sensitivity,
        reward_type,
        plot_strategies = False
    ):
        # super(PandemicEnv, self).__init__()
        self.m = m
        self.n = n
        self.reward_type = reward_type
        self.plot_strategies = plot_strategies
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
            transmission_rate,
            sensitivity
        )
        self.players_lattice.get_strategy()
        self.pandemic_length = 0

        self.infected_num_list, self.vaccinated_num_list, self.recovered_num_list = [], [], []
        self.reward_list = []
        self.avg_infected_epi, self.avg_vaccinated_epi, self.avg_recovered_epi = [], [], []

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

        print("Infections for step {}: {} ".format(self.pandemic_length, num_infected))
        self.infected_num_list.append(num_infected)
        self.vaccinated_num_list.append(self.players_lattice.count_num_strategy(1))
        self.recovered_num_list.append(self.players_lattice.count_num_strategy(3))

        reward = self.players_lattice.calc_reward(contact_rate, self.pandemic_length, self.reward_type)


        # if self.pandemic_length >= 100:
        # if self.players_lattice.count_num_strategy(2) <= 0.01*(self.m*self.n):
        if num_infected <= 0:
            self.avg_infected_epi.append(sum(self.infected_num_list)) #/len(self.infected_num_list))
            self.avg_vaccinated_epi.append(sum(self.vaccinated_num_list)) #/len(self.vaccinated_num_list))
            self.avg_recovered_epi.append(sum(self.recovered_num_list)) #/len(self.recovered_num_list))
            # self.avg_infected_epi.append(self.players_lattice.count_num_strategy(2))
            # self.avg_vaccinated_epi.append(self.players_lattice.count_num_strategy(1))
            # self.avg_recovered_epi.append(self.players_lattice.count_num_strategy(3))
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
        if self.plot_strategies:
            fig, ax = plt.subplots()
            ax.plot(self.avg_infected_epi, color="red")
            ax.plot(self.avg_vaccinated_epi, color="blue")
            ax.plot(self.avg_recovered_epi, color="green")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Number of individuals")
            ax.legend(['Infected', 'Vaccinated', 'Recovered'])
            ax.set_title("Change in total number of individuals for each strategy")

            try:
                fig.savefig(fig, "states_"+ self.pandemic_length +".png")
            except:
                plt.show()
                
            self.avg_infected_epi, self.avg_vaccinated_epi, self.avg_recovered_epi = [], [], []

        self.players_lattice.state_zero()
        state = self.players_lattice.build_matrix_strategy(0)
        self.pandemic_length = 0
        return state

    def render(self, mode="human", close=False):
        """
        Render the current state of the lattice.
        """
        self.players_lattice.draw_matrix_strategy(-1)