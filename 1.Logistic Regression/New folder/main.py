import numpy as np
from datapoints import set_seed, labeled_two_classPoints
from plot2d import Plot2D

set_seed(135)
datapoints = labeled_two_classPoints(Aweight=5, Bweight=3, 
                                     Aamount=500, Bamount=500, 
                                     scale=10)
plot = Plot2D((10,10))

rows    = datapoints.shape[0]
columns = datapoints.shape[1]

X = np.c_[np.ones(rows),datapoints[:,:2]]
y = datapoints[:,2]
theta = np.zeros(3)

sigmoid = lambda x: np.reciprocal(1+np.exp(-1*x))

def predict(theta, X):
    return sigmoid(X.dot(theta))

def CostFunction(theta, X, y):
    m = X.shape[0]
    h = predict(theta, X) # hypothesis vector   
    return ( np.dot(-1*y.transpose(),np.log(h)) - np.dot((1 - y).transpose(),np.log(1 - h)) ) / m

def grad(theta, X,y):
    m = X.shape[0]
    h = predict(theta, X) # hypothesis vector          
    return (X.transpose().dot((h - y))) / m

a = 0.1
predictor_func = lambda X: predict(theta, np.c_[np.ones(X.shape[0]), X])


plot.plot_labeled_data(datapoints)

for i in range(1000000):
    theta = theta - a*grad(theta, X,y)

    if not i % 1000:
        plot.plot_decision_surface(predictor_func)
        cost = CostFunction(theta, X, y)
        print(f'[{i:07}] Cost is: {cost}')
        plot.add_line_cost_func(cost,1)
        plot.pause(0.01)


    



