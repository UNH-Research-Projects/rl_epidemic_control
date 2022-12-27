class Players(object):
    def __init__(self):
        self.strategy = 0
        self.neighbors = []
        self.von_neighbors = []
        self.cross_neighbors = []
        self.num_neighbors = 0
        self.strategy_history = []
        self.payoff_history = []
        self.infection_time = 1000000
        self.disease_duration = 5
