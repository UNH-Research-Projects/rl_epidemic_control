import players.create_players as cp
import matplotlib.pylab as plt
import matplotlib.animation as animation


class Game(object):
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
        iteration,
    ):
        self.m = m
        self.n = n
        self.iteration = iteration
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
            transmission_rate
        )

    def synchronous_game(self):
        self.players_lattice.get_strategy()
        self.players_lattice.update_lattice(self.iteration)
        self.players_lattice.draw_matrix_strategy(0)
        self.players_lattice.draw_matrix_strategy(-1)

        count_strategy = self.players_lattice.count_num_strategy_result(-1)
        count_strategy[0].append(
            self.players_lattice.calc_epidemic_season_length(self.iteration)
        )
        print(count_strategy)
        fig = plt.figure()
        list_image = self.players_lattice.creat_list_image(self.iteration)
        ani = animation.ArtistAnimation(
            fig, list_image, interval=200, blit=True, repeat_delay=1000
        )
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=10, metadata=dict(artist="Me"), bitrate=2000)
        ani.save("mymovie.mp4", writer=writer)
        plt.show()
        return count_strategy
