from sklearn.svm import LinearSVC

class SVM:
    def run_linear(self, datapoints, actual):
        model = LinearSVC()
        model.fit(datapoints, actual)
        return model.get_params

lol = SVM()
