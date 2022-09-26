import LinearRegression
import LogisticalRegression
import matplotlib.pyplot as plt

import yaml
import pandas as pd

def test_linear_regression(df):
    datapoints = [(i,) for i in df['x'].tolist()]
    actual = df['y'].tolist()

    a = LinearRegression.LinearRegression(0.0001, log=True)
    a.run_regression(datapoints, actual)

def test_logictial_regression(df):
    datapoints = [i for i in df['x'].tolist()]
    actual = df['y'].tolist()

    a = LogisticalRegression.LogisticalRegression(0.0001, log=True)
    a.run_regression(datapoints, actual)


#test_linear_regression(pd.read_csv(r'test.csv'))
df = pd.read_csv(r'logtest.csv')

datapoints = []
col = []

for index, row in df.iterrows():
    datapoints.append((row['credit_score']/100, row['age']/10))
    if row['churn'] == 0:
        col.append(0) 
    else:
        col.append(1)

a = LogisticalRegression.LogisticalRegression(0.0001, log=True)
a.run_regression(datapoints, col)
#datapoints = [(i,) for i in df['credit_score'].tolist()]


"""
plt.plot(datapoints, marker="o", c = col[i])
plt.title('Histogram of IQ')
plt.xlabel('Mutation Rate')
plt.ylabel('Number of Runs')
plt.show()
"""