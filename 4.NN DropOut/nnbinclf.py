import numpy as np

class NN(object):
    def __init__(self, X, Y, alpha=1, keep_prob=None, layers_list=[2, 3, 1]):
        self.layers_list = layers_list
        self.num_layers = len(layers_list)
        self.W_dims = self.__calc_dims(self.layers_list)
        # print(self.W_dims)
        self.WW, self.BB = self.__init_weights(self.W_dims) #list of weights and biases for each layer

        self.VdWW = [None] + [np.zeros(W_dims) for W_dims in self.W_dims]
        self.VdBB = [None] + [np.zeros((W_dims[0],1)) for W_dims in self.W_dims]

        self.SdWW = [None] + [np.zeros(W_dims) for W_dims in self.W_dims]
        self.SdBB = [None] + [np.zeros((W_dims[0],1)) for W_dims in self.W_dims]

        self.adagradcacheWW = [None] + [np.zeros(W_dims) for W_dims in self.W_dims]
        self.adagradcacheBB = [None] + [np.zeros((W_dims[0],1)) for W_dims in self.W_dims]
        
        self.X_unnorm = X
        self.X, self.X_mean, self.X_std = self.normalize(self.X_unnorm, gen_stats=True)
        self.Y = Y
        self.m = X.shape[1]
        self.alpha = alpha

        if keep_prob:
            assert isinstance(keep_prob, list), 'keep_probe is a list of probabilities of keeping neurons during dropout in each layer'
            assert len(keep_prob) == self.num_layers, 'keep_prob list should have the same number of layers as the model itself'      
            assert keep_prob[-1] == 1, 'we don\'t drop from output layer'
            keep_prob = [ float(i) for i in keep_prob]
            self.keep_prob = keep_prob # lambda
        else:
            self.keep_prob = [1.0]*self.num_layers


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
            W = np.array(np.random.randn(*mx_dims)) / np.sqrt(inp_num/2)
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


    def J(self, WW, BB, X=None, Y=None):
        if X is None or Y is None:
            X, Y = self.X, self.Y
            m = self.m
        else:
            m = X.shape[1]
        AA, _, _ = self.__predict(X, WW, BB)
        A = AA[-1]
        result = -(np.dot(Y, np.log(A).T) + np.dot((1-Y), np.log(1-A).T)) / m
        return np.asscalar(result)


    def sigmoid(self, Z):
        return np.reciprocal((1 + np.exp(-Z)))


    def relu(self, Z):
        A = np.maximum(0,Z)
        return A


    def drelu(self, Z):
        dZ = np.array(Z >= 0, dtype=np.int)
        return dZ


    def __predict(self, X, WW, BB, dropout=False):
        if not dropout:
            KP = [1.0]*(self.num_layers)
        else:
            KP = self.keep_prob
            

        AA = [X]
        ZZ = [None]  
        DD  = [ np.random.rand(*X.shape) < KP[0] ]
        
        AA[0] *= DD[0]
        AA[0] = AA[0] / KP[0]


        for l in range(1,self.num_layers-1):
            Z = np.dot(WW[l], AA[l-1]) + BB[l]            
            A = self.relu(Z)            
            D = np.random.rand(*A.shape) < KP[l]            
            A *= D            
            A = A / KP[l]
            ZZ.append(Z)
            AA.append(A)
            DD.append(D)

        Z = np.dot(WW[-1], AA[-1]) + BB[-1]                          
        A = self.sigmoid(Z)
        D = np.random.rand(*A.shape) < KP[-1]
        A *= D
        A = A / KP[l]

        ZZ.append(Z)
        AA.append(A)
        DD.append(D)
        return AA, ZZ, DD


    def predict(self, X, normalize=True):
        if normalize:
            X, X_mean, X_std = self.normalize(X)
        AA, ZZ, DD = self.__predict(X, self.WW, self.BB)
        return AA[-1]


    def random_batch(self, batch_size):
        idx = np.random.randint(self.m, size=batch_size)
        X = self.X[:,idx]
        Y = np.array(self.Y[:,idx])
        return X, Y


    def grad_minibatch(self, batch_size):
        KP = self.keep_prob
        X, Y = self.random_batch(batch_size)
        AA, ZZ, DD = self.__predict(X, self.WW, self.BB, dropout=True)

        # dZ has backwards numeration
        dZZ = [None]*self.num_layers
        dAA = [None]*self.num_layers
        dWW = [None]*self.num_layers
        dBB = [None]*self.num_layers

        dZZ[-1] = (AA[-1] - Y) # we do not drop from output layer       


        dWW[-1] = np.dot(dZZ[-1], AA[-2].T) / self.m
        dBB[-1] = np.sum(dZZ[-1], axis=1, keepdims=True) / self.m
        for l in range(self.num_layers-2,0,-1):
            dAA[l] = np.dot(self.WW[l+1].T, dZZ[l+1]) * DD[l]
            dAA[l] = dAA[l] / KP[l]
            dZZ[l] = dAA[l]*self.drelu(ZZ[l])
            dWW[l] = np.dot(dZZ[l], AA[l-1].T) / self.m 
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

    def backprop_minibatch(self, iterations=1, batch_size=256): 
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

    def backprop_minibatch_adagrad(self, iterations=1, batch_size=256):   
        epsilon = 1e-7
        for i in range(iterations):
            dWW, dBB = self.grad_minibatch(batch_size)
            for l in range(1,self.num_layers):
                self.adagradcacheWW[l] += dWW[l]**2
                self.adagradcacheBB[l] += dBB[l]**2
                self.WW[l] -= self.alpha * dWW[l] / (np.sqrt(self.adagradcacheWW[l])  + epsilon)
                self.BB[l] -= self.alpha * dBB[l] / (np.sqrt(self.adagradcacheBB[l])  + epsilon)


    def backprop_minibatch_rmsprop(self, iterations=1, batch_size=256, decay_rate=0.99):   
        epsilon = 1e-8
        un_decay_rate = 1 - decay_rate   
        for i in range(iterations):
            dWW, dBB = self.grad_minibatch(batch_size)
            for l in range(1,self.num_layers):
                self.SdWW[l] = decay_rate * self.SdWW[l] + un_decay_rate * dWW[l]**2
                self.SdBB[l] = decay_rate * self.SdBB[l] + un_decay_rate * dBB[l]**2
                self.WW[l] -= self.alpha * dWW[l] / (np.sqrt(self.SdWW[l]) + epsilon)
                self.BB[l] -= self.alpha * dBB[l] / (np.sqrt(self.SdBB[l]) + epsilon)


    def backprop_minibatch_adam(self, iterations=1, realiter = 0, batch_size=256, b1=0.9, b2=0.999): 
        nb1 = 1 - b1  
        nb2 = 1 - b2
        epsilon = 1e-7
        for i in range(iterations):
            t = i + realiter
            b1_bias_correction = 1 - b1**t
            b2_bias_correction = 1 - b2**t
            dWW, dBB = self.grad_minibatch(batch_size)
            for l in range(1,self.num_layers):
                self.VdWW[l] = (b1 * self.VdWW[l] + nb1 * dWW[l])    / b1_bias_correction
                self.VdBB[l] = (b1 * self.VdBB[l] + nb1 * dBB[l])    / b1_bias_correction
                self.SdWW[l] = (b2 * self.SdWW[l] + nb2 * dWW[l]**2) / b2_bias_correction
                self.SdBB[l] = (b2 * self.SdBB[l] + nb2 * dBB[l]**2) / b2_bias_correction
                self.WW[l] -= self.alpha * self.VdWW[l] / (np.sqrt(self.SdWW[l]) + epsilon)
                self.BB[l] -= self.alpha * self.VdBB[l] / (np.sqrt(self.SdBB[l]) + epsilon)



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