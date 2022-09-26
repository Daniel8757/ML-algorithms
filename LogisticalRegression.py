import math

LEARNING_RATE = 0.001

class LogisticalRegression:
    def __init__(self, learning_rate=LEARNING_RATE, log=False):
        self.learning_rate = learning_rate
        self.log = log

    def log_print(self, to_print):
        if self.log:
            print(to_print)
    
    def get_probability(self, params, datapoint):
        return 1 / (1 + math.e**(params[0] + sum([params[i+1]*datapoint[i] for i in range(len(datapoint))])))

    def predict_values(self, params, datapoints):
        pred = []
        for point in datapoints:
            pred.append(self.get_probability(params, point))
        return pred

    # not sure if you should just add loss of each to get total loss
    def loss(self, pred, actual):
        loss_one = lambda a, p : -(p*math.log(a) + (1-p)*math.log(1-a)) if (0 < a and a < 1) else int(a == p)
        return sum([loss_one(pred[i], actual[i]) for i in range(len(pred))])

    def update_values(self, params, datapoints, pred, actual):
        n = len(datapoints)
        for i, param in enumerate(params):
            for j in range(len(pred)):
                print(actual[j], pred[j])
            if i == 0:
                new_param = param - self.learning_rate * sum([actual[j]/pred[j] + (actual[j]-pred[j])/(pred[j]-1) for j in range(len(pred))])
            else:
                new_param = param - self.learning_rate * sum([(actual[j]/pred[j] + (actual[j]-pred[j])/(pred[j]-1))*datapoints[j][i-1] for j in range(len(pred))])
            params[i] = new_param
        
    def run_regression(self, datapoints, actual, runs=1000):
        # first param is always the "b" value
        params = [0 for _ in range(len(datapoints[0])+1)]

        for _ in range(runs):
            pred = self.predict_values(params, datapoints)
            self.log_print("Current loss is {}".format(self.loss(pred, actual)))
            self.update_values(params, datapoints, pred, actual)

        self.log_print("Final loss is {}".format(self.cost_function(pred, actual)))