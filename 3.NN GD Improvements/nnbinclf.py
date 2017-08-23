import numpy as np

class NN(object):
    def __init__(self, X, Y, alpha=1, l=0.01, layers_list=[2, 3, 1]):
        self.layers_list = layers_list
        self.num_layers = len(layers_list)
        self.W_dims = self.__calc_dims(self.layers_list)
        # print(self.W_dims)
        self.WW, self.BB = self.__init_weights(self.W_dims) #list of weights and biases for each layer
        
        self.VdWW = [None] + [np.zeros(W_dims) for W_dims in self.W_dims]
        self.VdBB = [None] + [np.zeros((W_dims[0],1)) for W_dims in self.W_dims]

        self.SdWW = [None] + [np.zeros(W_dims) for W_dims in self.W_dims]
        self.SdBB = [None] + [np.zeros((W_dims[0],1)) for W_dims in self.W_dims]

        self.X_unnorm = X
        self.X, self.X_mean, self.X_std = self.normalize(self.X_unnorm, gen_stats=True)
        self.Y = Y
        self.m = X.shape[1]
        self.alpha = alpha
        self.l = l # lambda


    def __calc_dims(self, layers_list):
        '''layers_list - list with number of units in each layer'''
        '''u - is units'''
        assert(layers_list[-1] == 1, 'for binary classifier where should be only 1 output layer')
        layers = range(len(layers_list)-1)
        return [ (layers_list[l+1], layers_list[l]) for l in layers]


    def __init_weights(self, weght_mx_dims_list):
        '''weght_mx_dims_list in form [(m,n), (m,n), ..., (m,n)] '''
        WW = [None] 
        BB = [None] # W0, B0 element - doesn't exist
        for mx_dims in weght_mx_dims_list:
            inp_num = mx_dims[0]
            W = np.random.randn(*mx_dims) / np.sqrt(inp_num/2)
            B = np.zeros((mx_dims[0], 1))
            WW.append(W)
            BB.append(B)
        return WW, BB

    
    def normalize(self, X_unnorm, gen_stats=False):
        if gen_stats:
            X_mean = np.mean(X_unnorm, axis=1)[:,np.newaxis] # column vector
            X_std = np.std(X_unnorm, axis=1)[:,np.newaxis]
        else:
            X_mean = self.X_mean
            X_std = self.X_std
        X = (X_unnorm - X_mean) / X_std
        return X, X_mean, X_std


    def J(self, WW, BB, X=None, Y=None, regularize=False):
        if X is None or Y is None:
            X, Y = self.X, self.Y
            m = self.m
        else:
            m = X.shape[1]
        AA, _ = self.__predict(X, WW, BB)
        A = AA[-1]
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


    def relu(self, Z):
        return np.maximum(0,Z)


    def drelu(self, Z):
        return np.array(Z >= 0, dtype=np.int)


    def __predict(self, X, WW, BB):
        # w = self.w
        # b = self.b
        AA = [X]
        # print(f'X shape {X.shape}') 
        # print(f'AA[0] shape {AA[0].shape}') 
        ZZ = [None]
        # for l in range(1,self.num_layers):
        #     print(f'W{l} shape {WW[l].shape}') 
        #     print(f'B{l} shape {BB[l].shape}')
        # iterating through layers from 1 (starting from 0) to penultimate        
        for l in range(1,self.num_layers-1):
            Z = np.dot(WW[l], AA[l-1]) + BB[l]
            A = self.relu(Z)
            ZZ.append(Z)
            AA.append(A)
        # everything is ok, if last WW has number l, then last AA at this point has number l-1
        Z = np.dot(WW[-1], AA[-1]) + BB[-1]  
        A = self.sigmoid(Z)
        ZZ.append(Z)
        AA.append(A)
        # for l in range(self.num_layers):
        #     if ZZ[l] is not None:
        #         print(f'Z{l} len {ZZ[l].shape}')
        #     if AA[l] is not None:
        #         print(f'A{l} len {AA[l].shape}')
        return AA, ZZ


    def predict(self, X, normalize=True):
        if normalize:
            X, X_mean, X_std = self.normalize(X)
        AA, ZZ = self.__predict(X, self.WW, self.BB)
        return AA[-1]


    def grad(self):
        AA, ZZ = self.__predict(self.X, self.WW, self.BB)
        # dZ has backwards numeration
        dZZ = [None]*self.num_layers
        dAA = [None]*self.num_layers
        dWW = [None]*self.num_layers
        dBB = [None]*self.num_layers

        dZZ[-1] = AA[-1] - self.Y
        R = self.l*self.WW[-1] # regularization term
        dWW[-1] = (np.dot(dZZ[-1], AA[-2].T) + R) / self.m
        dBB[-1] = np.sum(dZZ[-1], axis=1, keepdims=True) / self.m
        for l in range(self.num_layers-2,0,-1):

            dAA[l] = np.dot(self.WW[l+1].T, dZZ[l+1])  
            dZZ[l] = dAA[l]*self.drelu(ZZ[l])
            R = self.l*self.WW[l]
            dWW[l] = (np.dot(dZZ[l], AA[l-1].T) + R) / self.m 
            dBB[l] = np.sum(dZZ[l], axis=1, keepdims=True) / self.m 
        return dWW, dBB


    def random_batch(self, batch_size):
        idx = np.random.randint(self.m, size=batch_size)
        X = self.X[:,idx]
        Y = self.Y[:,idx]
        return X, Y


    def grad_minibatch(self, batch_size):
        X, Y = self.random_batch(batch_size)
        AA, ZZ = self.__predict(X, self.WW, self.BB)
        # dZ has backwards numeration
        dZZ = [None]*self.num_layers
        dAA = [None]*self.num_layers
        dWW = [None]*self.num_layers
        dBB = [None]*self.num_layers
        dZZ[-1] = AA[-1] - Y

        R = self.l*self.WW[-1] # regularization term
        dWW[-1] = (np.dot(dZZ[-1], AA[-2].T) + R) / self.m
        dBB[-1] = np.sum(dZZ[-1], axis=1, keepdims=True) / self.m
        for l in range(self.num_layers-2,0,-1):
            dAA[l] = np.dot(self.WW[l+1].T, dZZ[l+1])  
            dZZ[l] = dAA[l]*self.drelu(ZZ[l])
            R = self.l*self.WW[l]
            dWW[l] = (np.dot(dZZ[l], AA[l-1].T) + R) / self.m 
            dBB[l] = np.sum(dZZ[l], axis=1, keepdims=True) / self.m 
        return dWW, dBB

    def backprop_numcheck(self, iterations=1):        
        pass


    def backprop(self, iterations=1):
        for i in range(iterations):
            dWW, dBB = self.grad()
            for l in range(1,self.num_layers):
                self.WW[l] -= self.alpha*dWW[l]
                self.BB[l] -= self.alpha*dBB[l]

    def backprop_minibatch_momentum(self, iterations=1, batch_size=256): 
        for i in range(iterations):
            dWW, dBB = self.grad_minibatch(batch_size)
            for l in range(1,self.num_layers):
                self.WW[l] -= self.alpha*dWW[l]
                self.BB[l] -= self.alpha*dBB[l]

    def backprop_minibatch_momentum(self, iterations=1, batch_size=256, b=0.9):   
        nb = 1 - b     
        for i in range(iterations):
            dWW, dBB = self.grad_minibatch(batch_size)
            for l in range(1,self.num_layers):
                self.VdWW[l] = b*self.VdWW[l] + nb*dWW[l]
                self.VdBB[l] = b*self.VdBB[l] + nb*dBB[l]
                self.WW[l] -= self.alpha*self.VdWW[l]
                self.BB[l] -= self.alpha*self.VdBB[l]

    def backprop_minibatch_rmsprop(self, iterations=1, batch_size=256, b2=0.9):   
        epsilon = 1e-8
        for i in range(iterations):
            dWW, dBB = self.grad_minibatch(batch_size)
            for l in range(1,self.num_layers):
                self.SdWW[l] = b2*self.SdWW[l] + np.square(dWW[l])
                self.SdBB[l] = b2*self.SdBB[l] + np.square(dBB[l])
                self.WW[l] -= self.alpha*(dWW[l] / np.sqrt(self.SdWW[l] + epsilon))
                self.BB[l] -= self.alpha*(dBB[l] / np.sqrt(self.SdBB[l] + epsilon))

    def num_grad(self):
        pass

    
    def accuracy(self, X, Y):
        A = self.predict(X)
        H = np.array(A >= 0.5, dtype=np.int)
        # tp_vector = np.array(H == Y and Y == 1, dtype=np.int)
        # tn_vector = np.array(H == Y and Y == 0, dtype=np.int)
        # fp_vector = np.array(H != Y and Y == 0, dtype=np.int)
        # fn_vector = np.array(H != Y and Y == 1, dtype=np.int)

        tt_vector = np.array(H == Y)
        tf_vector = np.array(H != Y)
        yp_vector = np.array(Y == 0)
        yn_vector = np.array(Y == 1)

        tp_vector = np.array(np.logical_and(tt_vector, yp_vector), dtype=np.int)
        tn_vector = np.array(np.logical_and(tt_vector, yn_vector), dtype=np.int)
        fp_vector = np.array(np.logical_and(tf_vector, yn_vector), dtype=np.int)
        fn_vector = np.array(np.logical_and(tf_vector, yp_vector), dtype=np.int)

        tp = np.mean(tp_vector)
        tn = np.mean(tn_vector)
        fp = np.mean(fn_vector)
        fn = np.mean(fp_vector)
        
        return tp, tn, fp, fn



    def train(self, iterations=10000, min_J_diff=0.1):
        pass

if __name__ == '__main__':
    from testdatapoints import set_seed, labeled_two_classPoints
    from plot2d import Plot2D
    set_seed(144)
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


    model = NN(X,Y, alpha=0.03, l=-0.5, layers_list=[2, 32, 32, 1])

    predictor_func = lambda X: model.predict(X)
    
    plot.plot_labeled_data(datapoints)

    # model.backprop_numcheck(iterations=10)
    accuracy = model.accuracy(model.X, model.Y)
    print(f'Training accuracy: {accuracy}')

    for i in range(1000):
        model.backprop(iterations=1000)
        cost = model.J(model.WW, model.BB)
        accuracy = model.accuracy(model.X, model.Y)
        print(f'[{i:04}] Cost is: {cost:.6f}, Training accuracy: {accuracy}')
        plot.plot_decision_surface(predictor_func)
        plot.add_line_cost_func(cost,1)
        
        plot.pause(0.01)