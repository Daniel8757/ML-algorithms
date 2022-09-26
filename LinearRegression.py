import numpy as np
import yaml

with open('config.yaml') as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)

LEARNING_RATE = configs['learning_rate']

class LinearRegression:
    def __init__(self, learning_rate=LEARNING_RATE, log=False):
        self.learning_rate = learning_rate
        self.log = log

    def log_print(self, to_print):
        if self.log:
            print(to_print)

    def cost_function(self, pred, actual):
        n = len(pred)
        squared_dist = sum([(pred[i]-actual[i])**2 for i in range(len(pred))])
        return squared_dist/n

    def predict_values(self, params, datapoints):
        pred = []
        for point in datapoints:
            prediction = sum([point[i]*params[i+1] for i in range(len(point))]) + params[0]
            pred.append(prediction)
        return pred

    def update_values(self, params, datapoints, pred, actual):
        # first param is always the "b" value
        n = len(datapoints)
        for i, param in enumerate(params):
            if i == 0:
                new_param = param - self.learning_rate * (2/n) * sum([(pred[j]-actual[j]) for j in range(len(pred))])
            else:
                new_param = param - self.learning_rate * (2/n) * sum([(pred[j]-actual[j])*datapoints[j][i-1] for j in range(len(pred))])
            params[i] = new_param
    
    # datapoints represents x values, actual represents y values
    def run_regression(self, datapoints, actual, runs=1000):
        params = [0 for _ in range(len(datapoints[0])+1)]

        for _ in range(runs):
            pred = self.predict_values(params, datapoints)
            self.log_print("Current squared error is {}".format(self.cost_function(pred, actual)))
            self.update_values(params, datapoints, pred, actual)
        
        self.log_print("Final squared error is {}".format(self.cost_function(pred, actual)))
