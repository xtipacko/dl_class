import warnings
warnings.filterwarnings("ignore")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import csv
from datapoints import labeled_two_classPoints, set_seed




class Plot2D:
    def __init__(self, XY):
        self.xy = XY
        self.last_point_func1 = None
        self.lines_func0 = None
        self.lines_func1 = None
        self.linspace0 = np.linspace(-1*self.xy[0], self.xy[0], num=100)
        mpl.rcParams['toolbar'] = 'None'
        self.fig = plt.figure(figsize=(20, 10), dpi=80)
        self.fig.set_facecolor((0,0,0))
        self.fig.suptitle('Logistic regression with toy data', fontsize=14, color=(0,0.6,0))
        self.ax0 = self.fig.add_subplot(121)
        self.ax1 = self.fig.add_subplot(122)
        self.fig.subplots_adjust(top=0.85)
        
        
        self.ax0.set_facecolor('black')
        
        self.ax0.spines['bottom'].set_color((1,1,1))
        self.ax0.spines['left'].set_color((1,1,1))
        self.ax0.spines['bottom'].set_position('center')
        self.ax0.spines['left'].set_position('center')
        self.ax0.tick_params(axis='x', colors='white')
        self.ax0.tick_params(axis='y', colors='white')
        self.ax0.axis([-1*self.xy[0],
                        self.xy[0],
                       -1*self.xy[1],
                        self.xy[0]])
        
        #hide 0 ticks
        xticks = self.ax0.xaxis.get_major_ticks()
        xticks[3].label1.set_visible(False)
        
        yticks = self.ax0.yaxis.get_major_ticks()
        yticks[3].label1.set_visible(False)
        
        
        self.ax1.set_facecolor('black')
        self.ax1.spines['bottom'].set_color((1,1,1))
        self.ax1.spines['left'].set_color((1,1,1))
        
        self.ax1.tick_params(axis='x', colors='white')
        self.ax1.tick_params(axis='y', colors='white')
        plt.legend(loc='upper left', scatterpoints=1, numpoints=1)
        self.contours = None
        
        self.show()


    def show(self):
        plt.ion()


    def plot_decision_surface(self, predictor_func):
        self.clear(contours=True)

        x1_min = -self.xy[0]
        x1_max =  self.xy[0]
        x2_min = -self.xy[1]
        x2_max =  self.xy[1]
        x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                             np.arange(x2_min, x2_max, 0.1))
        Y = predictor_func(np.c_[x1.ravel(),x2.ravel()])
        Y = Y.reshape(x1.shape)
        self.contours = self.ax0.contourf(x1, x2, Y, cmap='RdBu', alpha=0.55)
        

    def clear(self, contours=False):
        if contours and self.contours:
            for s in self.contours.collections:
                s.remove()





    def plot_labeled_data(self, datapoints):
        # where datapoints is a list of numbers (x1, x2,  y) 
        # y is label (1 or 0)
        # x1, x2 is features
        assert type(datapoints) is np.ndarray, 'datapoint should be numpy.array'
        assert datapoints.shape[1] == 3, 'datapoint format should be (x1, x2 ,y)'
        classA = np.array([ row[0:2] for row in datapoints if row[2] == 1.0])
        self.ax0.scatter(classA[:,0],classA[:,1], s=5, c=(0.53,  0.90 ,  0.99), marker='o', label='ClassA')
        classB = np.array([ row[0:2] for row in datapoints if row[2] == 0.0])
        self.ax0.scatter(classB[:,0],classB[:,1], s=5, c=(0.81,  0.28 ,  0.29), marker='o', label='ClassB')


    def pause(self, sec):
        plt.pause(sec)

    def add_line_cost_func(self, val, step):
        # from self.last_point_func1 = 0
        if not self.last_point_func1:
            self.last_point_func1 = (0,val)
        p1 = self.last_point_func1
        p2 = (self.last_point_func1[0]+step, val)

        self.ax1.plot( (p1[0],p2[0]), (p1[1],p2[1]), color='white')

        # moving x by step and y making = val
        self.last_point_func1 = p2
        


if __name__ == '__main__':
    plot = Plot2D((10,10))
    set_seed(135)
    datapoints = labeled_two_classPoints(Aweight=5, Bweight=3, Aamount=500, Bamount=500, scale=10)
    plot.plot_labeled_data(datapoints)
    
    from sklearn.svm import SVC


    clf =  SVC(kernel='rbf', probability=True)
    clf2 =  SVC(kernel='poly', degree=3, probability=True)
    X = datapoints[:,:2]
    y = datapoints[:,2]
    clf.fit(X,y)
    clf2.fit(X,y)
    
    plot.plot_decision_surface(clf2.predict)
    plot.pause(3)
    plot.plot_decision_surface(clf.predict)

    #plot.clear(contours=True)



    from math import log
    for i in range(1,10000):    
        print(i)
        f = lambda x: 300/x**(1/2)
        plot.add_line_cost_func(f(i),0.1)
        plot.pause(0.0001)

    plot.pause(100)









