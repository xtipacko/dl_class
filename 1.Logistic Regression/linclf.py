import numpy as np

class LR(object):
    def __init__(self, X, Y, init_w, init_b, alpha=1, l=0.01):
        self.w = init_w 
        self.b = init_b        
        self.X_unnorm = X
        self.X, self.X_mean, self.X_std = self.normalize(self.X_unnorm, gen_stats=True)
        self.Y = Y
        self.m = X.shape[1]
        self.alpha = alpha
        self.l = l # lambda

    
    def normalize(self, X_unnorm, gen_stats=False):
        if gen_stats:
            X_mean = np.mean(X_unnorm, axis=1)[:,np.newaxis] # column vector
            X_std = np.std(X_unnorm, axis=1)[:,np.newaxis]
        else:
            X_mean = self.X_mean
            X_std = self.X_std
        X = (X_unnorm - X_mean) / X_std
        return X, X_mean, X_std


    def J(self, w, b, X=None, Y=None, regularize=False):
        if not X or not Y:
            X, Y = self.X, self.Y
            m = self.m
        else:
            m = X.shape[1]
        A, _ = self.__predict(X, w, b)
        if regularize:
            # print(f'w shape: {w.shape}, b is: {b}, X.shape: {X.shape}, y.shape: {y.shape}, m is: {m}, a.shape is: {a.shape}')
            l = self.l
            reg_term = l*np.dot(w.T, w) / (2*m)
            result = -(np.dot(Y, np.log(A).T) + np.dot((1-Y), np.log(1-A).T)) / m + reg_term
        else:
            result = -(np.dot(Y, np.log(A).T) + np.dot((1-Y), np.log(1-A).T)) / m
        # print(type(result))
        # print(result.shape)
        return np.asscalar(result)


    def sigmoid(self, Z):
        return np.reciprocal((1 + np.exp(-Z)))


    def __predict(self, X, w, b):
        # w = self.w
        # b = self.b
        Z = np.dot(w.T, X) + b
        A = self.sigmoid(Z)
        return A, Z


    def predict(self, X, normalize=True):
        if normalize:
            X, X_mean, X_std = self.normalize(X)
        return self.__predict(X, self.w, self.b)


    def grad(self):
        A, Z = self.__predict(self.X, self.w, self.b)
        dZ = A - self.Y
        reg_term = self.l*self.w
        dw = (np.dot(self.X, dZ.T) + reg_term) / self.m 
        db = np.sum(dZ) / self.m 
        return dw, db


    def backprop_numcheck(self, iterations=1):        
        init_w = self.w
        init_b = self.b
        for i in range(iterations):
            print(f'[{i}] ', end='')            
            dw, db = self.grad()
            print(f'Gradient calculated')
            ndw, ndb = self.num_grad()
            diff_dw = dw - ndw
            diff_db = db - ndb
            max_diff = max(np.max(diff_dw), diff_db)
            # print(db, ndb)
            # print(np.c_[dw, ndw])
            print(f'[{i:03}] Numerical Check, max dw/db diff  = {max_diff}')
            self.w -=  self.alpha*ndw
            self.b -=  self.alpha*ndb
        self.w = init_w
        self.b = init_b


    def backprop(self, iterations=1):        
        for i in range(iterations):
            dw, db = self.grad()
            self.w -=  self.alpha*dw
            self.b -=  self.alpha*db


    def num_grad(self):
        w = self.w
        b = self.b
        # print(f'w.shape is {w.shape}')
        epsilon = 1e-4
        perturb_w = np.zeros((w.shape[0], 1))
        # print(perturb_w.shape)
        dw = np.zeros((w.shape[0],1))
        b = 0
        for i in range(w.shape[0]):
            perturb_w[i] = epsilon
            # print(f'w.shape is {w.shape}')
            # print(f'perturb_w.shape is {perturb_w.shape}')
            # print(f'(w - perturb_w).shape is {(w - perturb_w).shape}')
            loss1 = self.J(w - perturb_w, b, regularize=True)
            loss2 = self.J(w + perturb_w, b, regularize=True)
            # print(f'perturb w [{i}]')
            # print(f'loss1 {loss1}')
            # print(f'loss2 {loss2}')
            dw[i] = (loss2 - loss1) / (2*epsilon)
            perturb_w[i] = 0
        b_loss1 = self.J(w, b - epsilon)
        b_loss2 = self.J(w, b + epsilon)
        db = (b_loss2 - b_loss1) / (2*epsilon)
        return dw, db

    
    def accuracy(self, X, y):
        a, z = self.predict(X)
        h = np.array(a >= 0.5, dtype=np.int)
        accuracy_vector = np.array(h == y, dtype=np.int)
        #show = np.column_stack((h,y,accuracy_vector))
        #print(show, show.shape)
        return np.mean(accuracy_vector)



    def train(self, iterations=10000, min_J_diff=0.1):
        pass

if __name__ == '__main__':
    from testdatapoints import set_seed, labeled_two_classPoints
    from plot2d import Plot2D
    set_seed(142)
    datapoints = labeled_two_classPoints(Aweight=2, Bweight=1, 
                                         Aamount=500, Bamount=500, 
                                         scale=5)
    plot = Plot2D((5,5))
    
    rows    = datapoints.shape[0]
    columns = datapoints.shape[1]
    
    X = datapoints[:,:2]
    X = X.T
    Y = datapoints[:,2]
    Y = Y.reshape(1,np.size(Y))
    print(f'Y shape: {Y.shape}')
    w = np.zeros((X.shape[0],1)) #np.random.randn(X.shape[0],1)        #(3072, 1)
    b = 0
    model = LR(X,Y,w,b, alpha=3)

    predictor_func = lambda X: model.predict(X)[0]
    
    plot.plot_labeled_data(datapoints)

    model.backprop_numcheck(iterations=10)
    accuracy = model.accuracy(model.X, model.Y)
    print(f'Training accuracy: {accuracy}')

    # for i in range(1000):
        # model.backprop(iterations=1)
        # cost = model.J(model.w, model.b)
        # accuracy = model.accuracy(model.X, model.y)
        # print(f'[{i:04}] Cost is: {cost:.6f}, Training accuracy: {accuracy}')
        # plot.plot_decision_surface(predictor_func)
        # plot.add_line_cost_func(cost,1)
        # 
        # plot.pause(0.01)