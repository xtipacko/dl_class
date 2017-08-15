import numpy as np



class LR(object):
    def __init__(self, X, y, init_w, init_b, alpha=1, l=0.01):
        self.w = init_w 
        self.b = init_b        
        self.X_unnorm = X
        self.X, self.X_mean, self.X_std = self.normalize(self.X_unnorm, gen_stats=True)
        self.y = y[:,np.newaxis]
        self.m = y.shape[0]
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


    def J(self, w, b):
        X, y = self.X, self.y
        a, _ = self.__predict(X)        
        result = ((-y.T).dot(np.log(a)) - (1-y.T).dot(np.log(1-a))) / self.m
        return np.asscalar(result)


    def sigmoid(self, z):
        return (1 / (1 + np.exp(-z)))


    def __predict(self, X):
        w = self.w
        b = self.b
        z = w.T.dot(X) + b
        z = z.T  # workaround?
        a = self.sigmoid(z)
        return a, z


    def predict(self, X, normalize=True):
        if normalize:
            X = self.normalize(X)
        return self.__predict(X)


    def grad(self):
        a, z = self.__predict(self.X)
        dz = a - self.y
        dw = self.X.dot(dz) / self.m
        db = np.sum(a - self.y) / self.m
        return dw, db


    def backprop_numcheck(self, iterations=1):        
        for i in range(iterations):
            print(f'[{i}] ', end='')            
            dw, db = self.grad()
            # print(f'dw shape {dw.shape}, db shape {db.shape}')
            print(f'Gradient calculated')
            ndw, ndb = self.num_grad()
            # print(f'ndw shape {ndw.shape}, ndb shape {ndb.shape}')
            diff_dw = dw - ndw
            diff_db = db - ndb
            max_diff = max(np.max(diff_dw), diff_db)
            print(f'[{i:03}] Numerical Check, max dw/db diff  = {max_diff}')
            # print(f'dw shape {dw.shape}, db shape {db.shape}')
            self.w -=  self.alpha*dw
            self.b -=  self.alpha*db


    def backprop(self, iterations=1):        
        for i in range(iterations):
            dw, db = self.grad()
            self.w -=  self.alpha*dw
            self.b -=  self.alpha*db


    def num_grad(self):
        w = self.w
        b = self.b
        epsilon = 1e-4
        perturb_w = np.zeros(w.shape[0])
        dw = np.zeros(w.shape[0])
        b = 0
        for i in range(w.shape[0]):
            perturb_w[i] = epsilon
            loss1 = self.J(w - perturb_w, b)
            loss2 = self.J(w + perturb_w, b)
            dw[i] = (loss2 - loss1) / (2*epsilon)
            perturb_w[i] = 0
        b_loss1 = self.J(w, b - epsilon)
        b_loss2 = self.J(w, b + epsilon)
        db = (b_loss2 - b_loss1) / (2*epsilon)
        return dw[:,np.newaxis], db

    
    def missclferr(self, X):
        a, z = self.__predict(X)
        h = np.array(a >= 0.5, dtype=np.int)
        return np.mean(h == y)



    def train(self, iterations=10000, min_J_diff=0.1):
        pass

if __name__ == '__main__':
    