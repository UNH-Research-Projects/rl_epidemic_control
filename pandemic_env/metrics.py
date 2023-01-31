import keras

class Metrics(keras.callbacks.Callback):
    def __init__(self, agent):
        keras.callbacks.Callback.__init__
        self.agent = agent
        
    def on_train_begin(self, logs={}):
        self.metrics = {key : [] for key in self.agent.metrics_names}

    def on_step_end(self, episode_step, logs):
        for ordinal, key in enumerate(self.agent.metrics_names, 0):
            self.metrics[key].append(logs.get('metrics')[ordinal])