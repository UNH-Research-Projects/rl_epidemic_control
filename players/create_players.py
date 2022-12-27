import players.players as pl
import numpy as np
import matplotlib.pylab as plt
import math


class CreatePlayers(object):
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
        self.dict_players = {}
        self.generate_players()
        self.get_neighbors()

    # Functions for defining players and their attributes
    def generate_players(self):
        for p in range(0, (self.m * self.n)):
            self.dict_players[p] = pl.Players()

    def find_key(self, i, j):
        key = (i * self.m) + (j)
        return key

    def get_neighbors(self):
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

    def state_zero(self):
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
        num_strategy = 0
        for p in range(0, self.lattice_size):
            if self.dict_players[p].strategy == strategy:
                num_strategy = num_strategy + 1
        return num_strategy

    def get_sensitivity_transmission_rate(self):
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
        num_infected_neighbor = self.get_neighbor_strategy(key_player)[2]
        # actual_susceptibility = contact_rate * self.dict_players[key_player].transmission_rate * (num_infected_neighbor/self.dict_players[key_player].num_neighbors)
        actual_susceptibility = (
            contact_rate
            * self.transmission_rate
            * (num_infected_neighbor / self.dict_players[key_player].num_neighbors)
        )
        return actual_susceptibility

    def calc_payoff_player(self, key_player, contact_rate):
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
        for p in range(0, self.lattice_size):
            self.dict_players[p].payoff_history.append(
                self.calc_payoff_player(p, contact_rate)
            )

    def calc_reward(self, contact_rate):
        reward = 0

        total_cost = self.cost_vaccine + self.cost_infection + self.cost_recover
        if contact_rate == 0.5:
            total_cost += self.lockdown_cost

        for p in range(0, self.lattice_size):
            player_strategy = self.dict_players[p].strategy

            if player_strategy == 0:
                weight = 0

            elif player_strategy == 1:
                weight = self.cost_vaccine / total_cost

            elif player_strategy == 2:
                weight = self.cost_infection / total_cost

            elif player_strategy == 3:
                weight = self.cost_recover / total_cost

            reward -= weight

        # number of infected /total population (Compute in every step)
        score = self.count_num_strategy(2) / self.lattice_size

        reward -= score
        # if contact_rate == 0.5:
        #     # add a lockdown cost
        #     cost += self.lockdown_cost

        # print("Calculated reward for contact_rate {} is: {}".format(contact_rate, log_reward))
        return np.exp(reward / self.lattice_size)

    # def calc_reward(self, contact_rate):
    #     reward = 0
    #     for p in range(0, self.lattice_size):
    #         player_strategy = self.dict_players[p].strategy

    #     # score = np.exp((self.count_num_strategy(player_strategy))
    #         if player_strategy == 0:
    #             reward = reward
    #         elif player_strategy == 1:
    #             reward = reward- self.cost_vaccine
    #         elif player_strategy == 2:
    #             reward = reward- self.cost_infection
    #         elif player_strategy == 3:
    #             reward = reward- self.cost_recover
    #     if contact_rate == 0.5:
    #         reward = reward - self.lockdown_cost
    #     else:
    #         reward = reward
    #     return reward

    # *****Main update*****
    def update_strategy_player(
        self, key_player, iteration, contact_rate, pandemic_time
    ):
        sensitivity_factor = self.media_affect

        # sensitivity_factor = self.dict_players[key_player].sensitivity
        # print(f"Updating strategy of the player for sensitivity factor of ", sensitivity_factor)

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
        # np.random.seed(100)
        for i in range(0, iteration):
            self.calc_payoff(contact_rate)
            self.update_strategy(i, contact_rate, pandemic_time)
            # self.draw_matrix_strategy(i)
        self.calc_payoff(contact_rate)

    # Functions for seeing result
    def build_matrix_strategy(self, iterate):
        matrix_strategy = [[0 for i in range(0, self.m)] for j in range(0, self.n)]
        for p in range(0, self.lattice_size):
            i = int(p / self.n)
            j = p % self.n
            matrix_strategy[i][j] = self.dict_players[p].strategy_history[iterate]
        return matrix_strategy

    def build_matrix_payoff(self, iterate):
        matrix_payoff = [[0 for i in range(0, self.m)] for j in range(0, self.n)]
        for p in range(0, self.lattice_size):
            i = int(p / self.n)
            j = p % self.n
            matrix_payoff[i][j] = self.dict_players[p].payoff_history[iterate]
        return matrix_payoff

    def draw_matrix_strategy(self, iterate):
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
        percent_result = []
        for i in range(0, iteration):
            percent_result.append(self.count_num_strategy_result(i)[1])
        plt.plot(percent_result)
        plt.show()

    def creat_list_image(self, iteration):
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
        epidemic_length = iteration + 1
        for i in range(0, iteration):
            if self.count_num_strategy_result(i)[0][2] == 0:
                epidemic_length = i
                break
        return epidemic_length
