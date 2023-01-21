import players.players as pl
import numpy as np
import matplotlib.pylab as plt
import math


class CreatePlayers(object):
    """A class for generating and managing a lattice of players.

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
        transmission_rate
    ):
        self.m = m
        self.n = n
        self.lattice_size = m * n
        self.weight_vac = weight_vac
        self.weight_inf = weight_inf
        self.weight_recov = weight_recov
        self.seed_strategy = seed_strategy
        self.cost_vaccine = cost_vaccine
        self.cost_infection = cost_infection
        self.cost_recover = cost_recover
        self.lockdown_cost = lockdown_cost
        self.transmission_rate = transmission_rate
        self.media_affect=3
        self.dict_players = {}
        self.generate_players()
        self.get_neighbors()
        # self.get_sensitivity_transmission_rate()

    # Functions for defining players and their attributes
    def generate_players(self):
        """Generate players and add them to the lattice."""
        for p in range(0, (self.m * self.n)):
            self.dict_players[p] = pl.Players()

    def find_key(self, i, j):
        """Find the key for the player at the given lattice coordinates.

        Parameters:
        i (int): The row coordinate.
        j (int): The column coordinate.

        Returns:
        int: The key for the player at the given coordinates.
        """
        
        key = (i * self.m) + (j)
        return key

    def get_neighbors(self):
        """Determine the neighbors for each player in the lattice.

        This method calculates the row and column coordinates for each player
        based on their key, and stores the coordinates in the player's attributes.
        """
    
        for p in range(0, self.lattice_size):
            i = int(p / self.n)
            j = p % self.n

            if i > 0 and j > 0 and i < self.m - 1 and j < self.n - 1:
                self.dict_players[p].von_neighbors = [
                    self.find_key(i - 1, j),
                    self.find_key(i, j - 1),
                    self.find_key(i, j + 1),
                    self.find_key(i + 1, j),
                ]
                self.dict_players[p].cross_neighbors = [
                    self.find_key(i - 1, j - 1),
                    self.find_key(i - 1, j + 1),
                    self.find_key(i + 1, j - 1),
                    self.find_key(i + 1, j + 1),
                ]

            elif i == 0 and j == 0:
                self.dict_players[p].von_neighbors = [
                    self.find_key(i, j + 1),
                    self.find_key(i + 1, j),
                ]
                self.dict_players[p].cross_neighbors = [self.find_key(i + 1, j + 1)]

            elif i == 0 and j > 0 and j < self.n - 1:
                self.dict_players[p].von_neighbors = [
                    self.find_key(i, j - 1),
                    self.find_key(i, j + 1),
                    self.find_key(i + 1, j),
                ]
                self.dict_players[p].cross_neighbors = [
                    self.find_key(i + 1, j - 1),
                    self.find_key(i + 1, j + 1),
                ]

            elif i == 0 and j == self.n - 1:
                self.dict_players[p].von_neighbors = [
                    self.find_key(i, j - 1),
                    self.find_key(i + 1, j),
                ]
                self.dict_players[p].cross_neighbors = [self.find_key(i + 1, j - 1)]

            elif j == self.n - 1 and i > 0 and i < self.m - 1:
                self.dict_players[p].von_neighbors = [
                    self.find_key(i - 1, j),
                    self.find_key(i, j - 1),
                    self.find_key(i + 1, j),
                ]
                self.dict_players[p].cross_neighbors = [
                    self.find_key(i - 1, j - 1),
                    self.find_key(i + 1, j - 1),
                ]

            elif i == self.m - 1 and j == self.n - 1:
                self.dict_players[p].von_neighbors = [
                    self.find_key(i - 1, j),
                    self.find_key(i, j - 1),
                ]
                self.dict_players[p].cross_neighbors = [self.find_key(i - 1, j - 1)]

            elif i == self.m - 1 and j > 0 and j < self.n - 1:
                self.dict_players[p].von_neighbors = [
                    self.find_key(i - 1, j),
                    self.find_key(i, j - 1),
                    self.find_key(i, j + 1),
                ]
                self.dict_players[p].cross_neighbors = [
                    self.find_key(i - 1, j - 1),
                    self.find_key(i - 1, j + 1),
                ]

            elif i == self.m - 1 and j == 0:
                self.dict_players[p].von_neighbors = [
                    self.find_key(i - 1, j),
                    self.find_key(i, j + 1),
                ]
                self.dict_players[p].cross_neighbors = [self.find_key(i - 1, j + 1)]

            elif j == 0 and i > 0 and i < self.m - 1:
                self.dict_players[p].von_neighbors = [
                    self.find_key(i - 1, j),
                    self.find_key(i, j + 1),
                    self.find_key(i + 1, j),
                ]
                self.dict_players[p].cross_neighbors = [
                    self.find_key(i - 1, j + 1),
                    self.find_key(i + 1, j + 1),
                ]

            self.dict_players[p].neighbors = (
                self.dict_players[p].von_neighbors
                + self.dict_players[p].cross_neighbors
            )
            self.dict_players[p].num_neighbors = len(self.dict_players[p].neighbors)

    def get_strategy(self):
        """Set the initial strategies for each player in the lattice.

        The initial strategies are chosen randomly based on the weight values
        for each strategy. The strategies and their corresponding weights are:
        vaccine (weight_vac), infection (weight_inf), and recovery (weight_recov).
        The resulting strategy for each player is stored in their attributes.
        """        
        # np.random.seed(self.seed_strategy)
        for p in range(0, self.lattice_size):
            rand_number = np.random.random()
            if rand_number < self.weight_vac:
                self.dict_players[p].strategy = 1  # 1 means vaccinated
            elif rand_number < self.weight_inf:
                self.dict_players[p].strategy = 2  # 1 means infected
                self.dict_players[p].infection_time = 0
            else:
                self.dict_players[p].strategy = 0  # 0 means susceptible

            self.dict_players[p].strategy_history.append(self.dict_players[p].strategy)

    def get_age(self):
        """Set the age for each player in the lattice.
        """        
        # np.random.seed(self.seed_strategy)
        for p in range(0, self.lattice_size):            
            self.dict_players[p].age = np.random.randint(1, 80)  

    def state_zero(self):
        """Reset the strategies and histories for each player in the lattice.

        This method sets the current strategy of each player to their initial strategy
        and clears their strategy and payoff histories. It also resets the infection
        time for infected players.
        """        
        for p in range(0, self.lattice_size):
            self.dict_players[p].strategy = self.dict_players[p].strategy_history[0]
            self.dict_players[p].strategy_history = []
            self.dict_players[p].payoff_history = []
            if self.dict_players[p].strategy == 2:
                self.dict_players[p].infection_time = 0
            else:
                self.dict_players[p].infection_time = 1000000
            self.dict_players[p].strategy_history.append(self.dict_players[p].strategy)

    # Functions for playing game
    def get_neighbor_strategy(self, key_player):
        """Get the strategies of the neighbors for a given player.

        Parameters:
        key_player (int): The key for the player whose neighbors to consider.

        Returns:
        list: A list of the strategies of the neighbors.
        int: The number of vaccinated neighbors.
        int: The number of infected neighbors.
        int: The number of recovered neighbors.
        """
        neighbor_strategy = []
        for i in range(0, self.dict_players[key_player].num_neighbors):
            strategy = self.dict_players[
                self.dict_players[key_player].neighbors[i]
            ].strategy_history[-1]
            neighbor_strategy.append(strategy)

        num_vac = neighbor_strategy.count(1)
        num_inf = neighbor_strategy.count(2)
        num_rec = neighbor_strategy.count(3)
        return neighbor_strategy, num_vac, num_inf, num_rec

    def count_num_strategy(self, strategy):
        """Count the number of players with a given strategy.

        Parameters:
        strategy (int): The strategy to count.

        Returns:
        int: The number of players with the given strategy.
        """
        num_strategy = 0
        for p in range(0, self.lattice_size):
            if self.dict_players[p].strategy == strategy:
                num_strategy = num_strategy + 1
        return num_strategy

    def get_sensitivity_transmission_rate(self):
        """Set the sensitivity and transmission rate for each player in the lattice.

        The sensitivity is chosen randomly from the values 1 to 7, and is stored in
        the player's attributes. The transmission rate is calculated based on the
        sensitivity, and is also stored in the player's attributes.
        """
        # np.random.seed(self.seed_strategy)
        for p in range(0, self.lattice_size):
            sensitivity = np.random.randint(1, 8)
            self.dict_players[p].sensitivity = sensitivity
            self.dict_players[p].sensitivity_history.append(
                self.dict_players[p].sensitivity
            )
            if sensitivity in range(1, 3):  # 1 to 2
                self.dict_players[p].transmission_rate = 0.33
            elif sensitivity in range(3, 6):  # 3 to 5
                self.dict_players[p].transmission_rate = 0.66
            elif sensitivity in range(6, 9):  # 6 to 8
                self.dict_players[p].transmission_rate = 1

    def calc_shared_cost_player(self, key_player, contact_rate):
        """Calculate the shared cost for a given player.

        Parameters:
        key_player (int): The key for the player whose shared cost to calculate.
        contact_rate (float): The current contact rate in the game.

        Returns:
        float: The shared cost for the player.
        """    
        num_infected_neighbor = self.get_neighbor_strategy(key_player)[2]
        num_vaccinated_neighbor = self.get_neighbor_strategy(key_player)[1]
        num_recovered_neighbor = self.get_neighbor_strategy(key_player)[3]
        prob_susceptible_infection = 0
        neighbor = self.get_neighbor_strategy(key_player)[0]
        i = 0
        for p in neighbor:
            if p == 0:
                prob_susceptible_infection = (
                    prob_susceptible_infection
                    + self.calc_as_player(
                        self.dict_players[key_player].neighbors[i], contact_rate
                    )
                )
            i = i + 1
        shared_cost = (
            (
                num_infected_neighbor
                / (self.dict_players[key_player].num_neighbors + 1)
                * self.cost_infection
            )
            + (
                num_vaccinated_neighbor
                / (self.dict_players[key_player].num_neighbors + 1)
                * self.cost_vaccine
            )
            + (
                contact_rate
                * prob_susceptible_infection
                * self.cost_infection
                / (self.dict_players[key_player].num_neighbors + 1)
            )
            + (
                num_recovered_neighbor
                / (self.dict_players[key_player].num_neighbors + 1)
                * self.cost_recover
            )
        )
        player_strategy = self.dict_players[key_player].strategy
        if player_strategy == 0:
            shared_cost = shared_cost + (
                self.calc_as_player(key_player, contact_rate)
                * self.cost_infection
                / (self.dict_players[key_player].num_neighbors + 1)
            )
        elif player_strategy == 1:
            shared_cost = shared_cost + (
                self.cost_vaccine / (self.dict_players[key_player].num_neighbors + 1)
            )
        elif player_strategy == 2:
            shared_cost = shared_cost + (
                self.cost_infection / (self.dict_players[key_player].num_neighbors + 1)
            )
        elif player_strategy == 3:
            shared_cost = shared_cost + (
                self.cost_recover / (self.dict_players[key_player].num_neighbors + 1)
            )
        return shared_cost

    def calc_as_player(self, key_player, contact_rate):
        """
        Calculate the actual susceptibility for a given player.

        Parameters:
        key_player (int): The key for the player whose actual susceptibility to calculate.
        contact_rate (float): The current contact rate in the game.

        Returns:
        float: The actual susceptibility for the player.
        """

        num_infected_neighbor = self.get_neighbor_strategy(key_player)[2]
        # actual_susceptibility = contact_rate * self.dict_players[key_player].transmission_rate * (num_infected_neighbor/self.dict_players[key_player].num_neighbors)
        actual_susceptibility = (
            contact_rate
            * self.transmission_rate
            * (num_infected_neighbor / self.dict_players[key_player].num_neighbors)
        )
        return actual_susceptibility

    def calc_payoff_player(self, key_player, contact_rate):
        """Calculate the payoff for a given player.

        Parameters:
        key_player (int): The key for the player whose payoff to calculate.
        contact_rate (float): The current contact rate in the game.

        Returns:
        float: The payoff for the player.
        """
        player_strategy = self.dict_players[key_player].strategy
        payoff_value = 0
        if player_strategy == 0:
            payoff_value = -(self.calc_shared_cost_player(key_player, contact_rate))
        elif player_strategy == 1:
            payoff_value = -(
                self.calc_shared_cost_player(key_player, contact_rate)
                + self.cost_vaccine
            )
        elif player_strategy == 2:
            payoff_value = -(
                self.calc_shared_cost_player(key_player, contact_rate)
                + self.cost_infection
            )
        elif player_strategy == 3:
            payoff_value = -(
                self.calc_shared_cost_player(key_player, contact_rate)
                + self.cost_recover
            )
        return payoff_value

    def calc_payoff(self, contact_rate):
        """Calculate the payoff for all players in the game.

        Parameters:
        contact_rate (float): The current contact rate in the game.
        """
        for p in range(0, self.lattice_size):
            self.dict_players[p].payoff_history.append(
                self.calc_payoff_player(p, contact_rate)
            )

    def calc_reward(self, contact_rate, iterate):
        """Calculate the reward for the game.

        Parameters:
        contact_rate (float): The current contact rate in the game.

        Returns:
        float: The reward for the game.
        """
        
        reward = 0

        newly_vaccinated = 0
        newly_infected = 0
        newly_recovered = 0

        self.lockdown_cost = self.lattice_size/10

        if iterate != 0:
            prev_strategy = self.count_num_strategy_result(iterate-1) # num strategy, % strategy
            current_strategy = self.count_num_strategy_result(iterate)

            newly_vaccinated = current_strategy[1][1] - prev_strategy[1][1]
            newly_infected = current_strategy[1][2] - prev_strategy[1][2]
            newly_recovered = current_strategy[1][3] - prev_strategy[1][3]

        else:
            newly_vaccinated = current_strategy[1][1] 
            newly_infected = current_strategy[1][2] 
            newly_recovered = current_strategy[1][3] 

        # reward = -(newly_vaccinated * self.cost_vaccine + newly_infected * self.cost_infection + newly_recovered * self.cost_recover)
        
        reward = self.weight_inf * (1- newly_infected) + self.weight_recov * (1- newly_recovered) + self.weight_vac * (1- newly_vaccinated)
        # for p in range(0, self.lattice_size):

        #     player_strategy = self.dict_players[p].strategy

        #     if player_strategy == 0:
        #         reward = reward
        #     elif player_strategy == 1:
        #         reward = reward- self.cost_vaccine
        #     elif player_strategy == 2:
        #         reward = reward- self.cost_infection
        #     elif player_strategy == 3:
        #         reward = reward- self.cost_recover
        if contact_rate == 0.5:
            reward = reward - self.lockdown_cost

        # else:
        #     reward = reward
        return reward

    # *****Main update*****
    def update_strategy_player(
        self, key_player, iteration, contact_rate, pandemic_time
    ):
        """
        Updates the strategy of a single player based on their neighbor's payoffs and the player's infection rate.

        Parameters:
        key_player (int): The key of the player whose strategy is being updated.
        iteration (int): The current iteration of the simulation.
        contact_rate (float): The current contact rate of the simulation.
        pandemic_time (int): The length of time the pandemic is expected to last.

        Returns:
        int: The updated strategy of the player.
        """

        sensitivity_factor = self.media_affect

        # sensitivity_factor = self.dict_players[key_player].sensitivity # comment this out if not using variable sensitivity

        neighbor_payoff = []
        for i in range(0, self.dict_players[key_player].num_neighbors):
            payoff = self.dict_players[
                self.dict_players[key_player].neighbors[i]
            ].payoff_history[-1]
            neighbor_payoff.append(payoff)
        sorted_neighbor_payoff = sorted(neighbor_payoff, reverse=True)
        referred_payoff = sorted_neighbor_payoff[0:sensitivity_factor]
        referred_payoff_sensitivity = sorted_neighbor_payoff[0:sensitivity_factor]
        referred_payoff_greater = sorted(
            i
            for i in neighbor_payoff
            if i > self.dict_players[key_player].payoff_history[-1]
        )
        candidate_neighbor = [
            i for i, j in enumerate(neighbor_payoff) if j in referred_payoff
        ]
        candidate_neighbor_strategy = [
            self.dict_players[self.dict_players[key_player].neighbors[i]].strategy
            for i in (candidate_neighbor)
        ]
        strategy = 0
        if 1 in (candidate_neighbor_strategy):
            strategy = 1
        else:
            rand_number = np.random.random()
            if rand_number < self.calc_as_player(key_player, contact_rate):
                strategy = 2
                self.dict_players[key_player].infection_time = iteration + pandemic_time
            else:
                strategy = 0
        return strategy

    def update_strategy(self, iterate, contact_rate, pandemic_time):
        """
        Updates the strategy of each player in the lattice for a given iteration. The new strategy is determined based on the player's current strategy, the state of their neighbors, and the given contact rate and pandemic time.

        Parameters:
        iterate (int): The current iteration of the simulation.
        contact_rate (float): The probability that a player will adopt the strategy of their neighbors if they have a higher payoff.
        pandemic_time (int): The number of iterations that a player will remain infected before recovering.

        Returns:
        None
        """
        new_strategy = []
        for p in range(0, self.lattice_size):
            rand_number = np.random.random()
            if self.dict_players[p].strategy == 0:
                new_strategy.append(
                    self.update_strategy_player(p, iterate, contact_rate, pandemic_time)
                )
            elif (
                self.dict_players[p].strategy == 2
                and (pandemic_time + iterate - self.dict_players[p].infection_time)
                == self.dict_players[p].disease_duration
            ):
                # elif self.dict_players[p].strategy == 2 and ((iterate - self.dict_players[p].infection_time) == self.dict_players[p].disease_duration or rand_number<self.weight_recov):
                new_strategy.append(3)
            else:
                new_strategy.append(self.dict_players[p].strategy)
        for p in range(0, self.lattice_size):
            self.dict_players[p].strategy = new_strategy[p]
            self.dict_players[p].strategy_history.append(new_strategy[p])

    def update_lattice(self, iteration, contact_rate, pandemic_time):
        """Update the strategies of players for a given number of iterations.
    
        Parameters
        ----------
        iteration : int
            The number of iterations to update the strategies.
        contact_rate : float
            The rate of contact between players.
        pandemic_time : int
            The duration of the pandemic.
        """
        # np.random.seed(100)
        for i in range(0, iteration):
            self.calc_payoff(contact_rate)
            self.update_strategy(i, contact_rate, pandemic_time)
            # self.draw_matrix_strategy(i)
        self.calc_payoff(contact_rate)

    # Functions for seeing result
    def build_matrix_strategy(self, iterate):
        """Build a matrix representing the strategies of players at a given iteration.
    
        Parameters
        ----------
        iterate : int
            The iteration for which to build the matrix of strategies.
            
        Returns
        -------
        matrix_strategy : list of lists
            A matrix representing the strategies of players at the given iteration.
        """
        matrix_strategy = [[0 for i in range(0, self.m)] for j in range(0, self.n)]
        for p in range(0, self.lattice_size):
            i = int(p / self.n)
            j = p % self.n
            matrix_strategy[i][j] = self.dict_players[p].strategy_history[iterate]
        return matrix_strategy

    def build_matrix_payoff(self, iterate):
        """Build a matrix representing the payoffs of players at a given iteration.
    
        Parameters
        ----------
        iterate : int
            The iteration for which to build the matrix of payoffs.
            
        Returns
        -------
        matrix_payoff : list of lists
            A matrix representing the payoffs of players at the given iteration.
        """
        matrix_payoff = [[0 for i in range(0, self.m)] for j in range(0, self.n)]
        for p in range(0, self.lattice_size):
            i = int(p / self.n)
            j = p % self.n
            matrix_payoff[i][j] = self.dict_players[p].payoff_history[iterate]
        return matrix_payoff

    def draw_matrix_strategy(self, iterate):
        """Build a matrix representing the payoffs of players at a given iteration.
    
        Parameters
        ----------
        iterate : int
            The iteration for which to build the matrix of payoffs.
            
        Returns
        -------
        matrix_payoff : list of lists
            A matrix representing the payoffs of players at the given iteration.
        """
        plt.imshow(
            self.build_matrix_strategy(iterate),
            vmin=0,
            vmax=3,
            interpolation="nearest",
            cmap=plt.cm.rainbow,
        )
        plt.show()
        # plt.savefig('plot.png', dpi=300, bbox_inches='tight')

    def count_num_strategy_result(self, iterate):
        """
        Counts the number of players in each strategy for a given iteration.

        Parameters
        ----------
        iterate : int
            The iteration for which the counts will be calculated.

        Returns
        -------
        num_strategy : list
            A list of integers, representing the number of players in each strategy (0, 1, 2, 3).
        percent_strategy : list
            A list of floats, representing the percentage of players in each strategy (0, 1, 2, 3).
        """
        num_strategy = [0 for i in range(0, 4)]
        percent_strategy = [0 for i in range(0, 4)]
        for p in range(0, self.lattice_size):
            if self.dict_players[p].strategy_history[iterate] == 0:
                num_strategy[0] = num_strategy[0] + 1
            elif self.dict_players[p].strategy_history[iterate] == 1:
                num_strategy[1] = num_strategy[1] + 1
            elif self.dict_players[p].strategy_history[iterate] == 2:
                num_strategy[2] = num_strategy[2] + 1
            elif self.dict_players[p].strategy_history[iterate] == 3:
                num_strategy[3] = num_strategy[3] + 1
        for i in range(0, len(num_strategy)):
            percent_strategy[i] = num_strategy[i] / (self.m * self.n)
        return num_strategy, percent_strategy

    def draw_num_strategy_result(self, iteration):
        """
        Plots the percentage of players in each strategy over time.

        Parameters
        ----------
        iteration : int
            The number of iterations to plot.
        """
        percent_result = []
        for i in range(0, iteration):
            percent_result.append(self.count_num_strategy_result(i)[1])
        plt.plot(percent_result)
        plt.show()

    def creat_list_image(self, iteration):
        """
        Creates a list of images representing the matrix of strategies at each iteration.

        Parameters
        ----------
        iteration : int
            The number of iterations to include in the list of images.

        Returns
        -------
        list_image : list
            A list of images representing the matrix of strategies at each iteration.
        """
        list_image = []
        for i in range(0, iteration):
            im = plt.imshow(
                self.build_matrix_strategy(i),
                vmin=0,
                vmax=3,
                interpolation="nearest",
                cmap=plt.cm.rainbow,
                animated=True,
            )
            # bim = plt.imshow(self.build_matrix_strategy(i), animated=True)
            list_image.append([im])
        return list_image

    def calc_epidemic_season_length(self, iteration):
        """Calculate the length of the epidemic season.
        
        Args:
            iteration (int): The maximum number of iterations.
            
        Returns:
            int: The length of the epidemic season.
        """
        epidemic_length = iteration + 1
        for i in range(0, iteration):
            if self.count_num_strategy_result(i)[0][2] == 0:
                epidemic_length = i
                break
        return epidemic_length
