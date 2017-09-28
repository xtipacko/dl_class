import numpy as np
from timeit import default_timer as now

def GCD(a, b):
    while b:
        a, b = b, a % b
    return a


def LCM(a, b):
    return a*b // GCD(a, b)


def one_hot_encoding(Yraw, K):
    H = np.arange(0, K).reshape(-1,1)
    Y = (H == Yraw).astype(np.int32)
    return Y


def one_hot_decoding(Y):
    return np.argmax(Y, axis=0)


def sigmoid(Z):
    return np.reciprocal(1 + np.exp(-Z))


def desigmoid(Z):
    sigm = sigmoid(Z)
    return (1-sigm)*sigm


def relu(Z):
    A = np.maximum(0, Z)
    return A


def derelu(Z):
    dZ = (Z >= 0).astype(np.int32)
    return dZ


def softmax(Z):
    '''softmax with numerical stabilization'''
    D = -np.max(Z, axis=0)   # (1 x m) - stabilizing term, I'm not sure about it...
    exponents = np.exp(Z + D) # (K x m)
    expsum = np.sum(exponents, axis=0, keepdims=True)
    return exponents / expsum


def batchnorm(Z, Gamma, Beta, avgZmean, avgZstd, epsilon, testtime=False):
    m = Z.shape[1]
    Zmean = avgZmean if testtime else np.mean(Z, axis=1, keepdims=True)
    Z0 = Z-Zmean
    Zvar = None if testtime else np.sum(np.square(Z0), axis=1, keepdims=True) / m
    Zstd = avgZstd if testtime else np.sqrt(Zvar+epsilon)
    Znorm = Z0 / Zstd
    Zbn = Znorm*Gamma + Beta
    return Zmean, Z0, Zvar, Zstd, Znorm, Zbn


def debatchnorm(dZbn, Gamma, Beta, Znorm, Zvar, Zstd, Z0, m, epsilon):
    dBeta = np.sum(dZbn, axis=1, keepdims=True)
    dGamma = np.sum(dZbn * Znorm, axis=1, keepdims=True)
    dZnorm = dZbn * Gamma
    dZ0 = (dZnorm/Zstd) - Z0*(np.sum(dZnorm*Z0, axis=1, keepdims=True) / ((Zvar+epsilon)*Zstd*m)) 
    dZ = dZ0 - np.sum(dZ0, axis=1, keepdims=True) / m
    return dZ, dGamma, dBeta


def desoftmax(dA):
    pass


def err_check(a, b): # to check gradient numerically
    af = a.ravel()
    bf = b.ravel()
    errors = np.abs(af - bf) / np.maximum(np.abs(af), np.abs(bf))
    return np.mean(errors)


class Batch_iterator:
    def __init__(self, Xbatches, Ybatches):
        assert len(Xbatches) == len(Ybatches), 'X, Y list of batches must have the same length' 
        self.Xbatches, self.Ybatches = Xbatches, Ybatches
        self.batches_num = len(self.Xbatches)
        self.index = -1
        self.iteration = -1
        self.epoch = 0
        self.total_iterations = 0
    
    def __call__(self, iterations):
        self.total_iterations = iterations
        return self


    def __iter__(self): 
        self.iteration = -1
        self.epoch = 0
        return self

    def __next__(self):
        self.iteration +=1
        if self.iteration >= self.total_iterations:
            raise StopIteration
        self.epoch = self.iteration // self.batches_num
        self.index = (self.index + 1) % self.batches_num 
        return (self.Xbatches[self.index], self.Ybatches[self.index], 
                self.iteration, self.epoch, self.index)


class Multiclass_NN(object):
    def __init__(self, X, Y, learning_rate=1, 
                 momentum_beta = 0.9, adam_beta = 0.99,
                 X_dev=None, Y_dev=None,
                 layers_list=[3072, 256, 128, 10],
                 keep_prob=[1.0,1.0,1.0,1.0],
                 optimizer='momentum', minibatch_size = 16,
                 preinitialized=False):
        assert isinstance(keep_prob, list), 'keep_probe is a list of probabilities of keeping neurons during dropout in each layer'
        assert len(keep_prob) == len(layers_list), 'keep_prob should have the same length as layers_list'
        assert layers_list[0] == X.shape[0], 'Number of inputs (A0) in layers_list should be equal to number of rows in X'
        assert (X_dev is None) == (Y_dev is None), 'You\'ve forgotten either X_dev or Y_dev'

        self.learning_rate = learning_rate
        self.momentum_beta = momentum_beta
        self.adam_beta = adam_beta
        self.optimizer = optimizer
        self.layers_list = layers_list
        self.KP = keep_prob
        self.L = len(layers_list)

        self.D = [None]*self.L # dropout masks

        self.K = self.layers_list[-1] # number of classes

        self.W_dims, self.Z_dims = self.calc_dims(self.layers_list, self.L)
        self.W, self.Gamma, self.Beta = self.He_init_weights(self.W_dims, self.Z_dims)

        #initializing batch norm variables for test time
        self.avgZmean = [None if dims is None else np.zeros((dims,1)) for dims in self.Z_dims]
        self.avgZstd  = [None if dims is None else np.zeros((dims,1))  for dims in self.Z_dims]
        self.epsilon = 1e-7

        self.A      = [None]*self.L
        self.Z      = [None]*self.L   
        self.Zmean  = [None]*self.L
        self.Z0     = [None]*self.L       
        self.Zvar   = [None]*self.L
        self.Zstd   = [None]*self.L
        self.Znorm  = [None]*self.L  
        self.Zbn    = [None]*self.L 

        self.dZ     = [None]*self.L
        #self.dZnorm = [None]*self.L
        # self.dZbn   = [None]*self.L
        self.dA     = [None]*self.L
        self.dBeta  = [None]*self.L
        self.dGamma = [None]*self.L
        self.dW     = [None]*self.L
        self.ndW     = [None]*self.L
        self.ndBeta  = [None]*self.L
        self.ndGamma = [None]*self.L


        self.VdW, self.VdGamma, self.VdBeta = self.init_optimizer(self.W_dims, 
                                                                  self.Z_dims,
                                                                  optimizer=self.optimizer)

        self.Xraw = X
        self.Yraw = Y

        if not preinitialized:
            self.X, self.X_mean, self.X_std = self.normalize(self.Xraw, gen_stats=True)
            self.Y = one_hot_encoding(self.Yraw, self.K)
            self.minibatch_size = minibatch_size

            # shuffle data
            self.X, self.Y = self.shuffle(self.X, self.Y)
            # Add (make if it doesn't exist) dev set
            if X_dev is None or Y_dev is None:
                self.X_dev = self.X[:, -10000:]
                self.Y_dev = self.Y[:, -10000:]
                self.X = self.X[:, :-10000]
                self.Y = self.Y[:, :-10000]
            else:
                self.X_dev = X_dev
                self.Y_dev = Y_dev
            self.Y_dev = one_hot_encoding(self.Y_dev, self.K)
            self.m_dev = self.X_dev.shape[1]

            m = self.X.shape[1]

            self.m_train = m


            self.batches_num = self.m_train  // self.minibatch_size
            last_mini_batch_idx = -(self.m_train % self.minibatch_size)
            if last_mini_batch_idx != 0:
                self.Xbatches = np.array_split(self.X[:,:last_mini_batch_idx], self.batches_num, axis=1)
                self.Ybatches = np.array_split(self.Y[:,:last_mini_batch_idx], self.batches_num, axis=1)
                self.Xbatches.append(self.X[:,last_mini_batch_idx:])
                self.Ybatches.append(self.Y[:,last_mini_batch_idx:])
                self.batches_num +=1
            else:
                self.Xbatches = np.array_split(self.X, self.batches_num, axis=1)
                self.Ybatches = np.array_split(self.Y, self.batches_num, axis=1)
            
            self.batchiterator = Batch_iterator(self.Xbatches, self.Ybatches)



    def shuffle(self, X, Y):
        p = np.random.permutation(Y.shape[1])
        X_shuffled, Y_shuffled = X[:,p], Y[:,p]
        # Y = np.array(Y, dtype=np.int)
        return X_shuffled, Y_shuffled


    def slice_to_minibatches(self, X, Y, size):
        pass


    def init_optimizer(self, W_dims, Z_dims, optimizer='momentum'):
        VdW = [None if dims is None else np.zeros(dims) for dims in W_dims]
        VdBeta = [None if dims is None else np.zeros((dims,1)) for dims in Z_dims]
        VdGamma = [None if dims is None else np.zeros((dims,1)) for dims in Z_dims] # ?
        return VdW, VdGamma, VdBeta


    def calc_dims(self, layers_list, L):
        W_dims = [None] + [ (layers_list[l+1], layers_list[l]) for l in range(L-1) ]
        Z_dims  = [None] + [ layers_list[l] for l in range(1,L) ]
        return W_dims, Z_dims


    def He_init_weights(self, W_dims, Z_dims):
        init_Wn = lambda dims: np.random.randn(*dims) / np.sqrt(dims[1]/2)
        W = [ None if dims is None else init_Wn(dims) for dims in W_dims ]
        init_Beta_n = lambda dims: np.zeros((dims, 1))
        Beta = [ None if dims is None else init_Beta_n(dims) for dims in Z_dims ]
        init_Gamma_n = lambda dims: np.ones((dims, 1))
        Gamma = [ None if dims is None else init_Gamma_n(dims) for dims in Z_dims ]
        return W, Gamma, Beta


    def normalize(self, X_unnorm, gen_stats=False):
        if gen_stats:
            X_mean = np.mean(X_unnorm, axis=1).reshape(-1,1)
            X_std  = np.std(X_unnorm, axis=1).reshape(-1,1)
        else:
            X_mean = self.X_mean
            X_std = self.X_std

        X_norm = (X_unnorm - X_mean) / X_std

        return X_norm, X_mean, X_std


    def new_dropout_mask(self, KP):
        for l in range(1,self.L):
            mask = (np.random.rand(self.layers_list[l],1) < KP[l]).astype(np.int32)
            self.D[l] = mask / KP[l] 


    def forward(self, X, W = None, Beta = None, Gamma = None, dropout=False, testtime=False):
        A        = self.A
        Z        = self.Z
        Zmean    = self.Zmean
        Z0       = self.Z0   
        Zvar     = self.Zvar 
        Zstd     = self.Zstd 
        Znorm    = self.Znorm
        Zbn      = self.Zbn  
        D        = self.D
        KP       = self.KP
        L        = self.L
        Beta     = self.Beta
        Gamma    = self.Gamma        
        avgZmean = self.avgZmean
        avgZstd  = self.avgZstd
        epsilon  = self.epsilon

        if W is None or Beta is None or Gamma is None:
            W = self.W
            Beta = self.Beta
            Gamma = self.Gamma
        if dropout:
            self.new_dropout_mask(KP)
        else:
            self.new_dropout_mask([1.0]*L)

        A[0] = X
        for l in range(1, L-1):
            Z[l] = W[l] @ A[l-1]

            (Zmean[l], Z0[l], Zvar[l], 
            Zstd[l], Znorm[l], Zbn[l]) = batchnorm(Z[l], Gamma[l], Beta[l], 
                                                   avgZmean[l], avgZstd[l],
                                                   epsilon, testtime=testtime)
            A[l] = relu(Zbn[l]) * D[l]  
            if not testtime:
                avgZmean[l] = 0.7*avgZmean[l] + 0.3*Zmean[l]
                avgZstd[l]  = 0.7*avgZstd[l]  + 0.3*Zstd[l]

        Z[-1] = W[-1] @ A[-2]


        (Zmean[-1], Z0[-1], Zvar[-1], 
        Zstd[-1], Znorm[-1], Zbn[-1]) = batchnorm(Z[-1], Gamma[-1], Beta[-1], 
                                                  avgZmean[-1], avgZstd[-1], epsilon, testtime=testtime)
        A[-1] = softmax(Zbn[-1]) * D[-1]
        if not testtime:
            avgZmean[-1] = 0.7*avgZmean[-1] + 0.3*Zmean[-1]
            avgZstd[-1]  = 0.7*avgZstd[-1]  + 0.3*Zstd[-1]


    def predict(self, X, W = None, B = None):
        X, _, _ = self.normalize(X)
        self.forward(X, testtime=True)
        return self.A[-1]


    def cost(self, AL, Y):
        #return np.mean(-np.sum(Y*np.log(AL), axis=0, keepdims=True))
        Y = Y.astype(np.bool)
        Yhj = AL[Y]
        Losses = -np.log(Yhj[np.nonzero(Yhj)])
        return np.sum(Losses) / self.minibatch_size


    def backprop(self, X, Y):
        A      = self.A
        Z      = self.Z
        Z0     = self.Z0   
        Zvar   = self.Zvar 
        Zstd   = self.Zstd 
        Znorm  = self.Znorm
        Zbn    = self.Zbn  
        D      = self.D
        L      = self.L
        m      = self.minibatch_size
        dZ     = self.dZ
        dBeta  = self.dBeta
        dGamma = self.dGamma
        dW     = self.dW
        Beta   = self.Beta
        Gamma  = self.Gamma
        W      = self.W
        epsilon = self.epsilon

        self.forward(X, dropout=True)

        dZbn  = (A[-1] - Y) / m
        dZ[-1], dGamma[-1], dBeta[-1] = debatchnorm(dZbn, Gamma[-1], Beta[-1], 
                                                    Znorm[-1], Zvar[-1], Zstd[-1], 
                                                    Z0[-1], m, epsilon)
        dW[-1] = dZ[-1] @ A[-2].T
        for l in range(L-2, 0, -1):
            dA = (W[l+1].T @ dZ[l+1]) * D[l]
            dZbn = dA * derelu(Zbn[l])
            dZ[l], dGamma[l], dBeta[l] = debatchnorm(dZbn, Gamma[l], Beta[l], 
                                                     Znorm[l], Zvar[l], Zstd[l], 
                                                     Z0[l], m, epsilon)
            dW[l] = dZ[l] @ A[l-1].T



    def sgd_train(self, iterations=1000, yld=10):
        pass


    def adam_train(self, iterations=1000, yld=10):
        pass


    def momentum_train(self, iterations=1000, yld=10, numcheck=False):
        b  = self.momentum_beta    
        nb = 1 - b
        a  = self.learning_rate
        L  = self.L
        dW     = self.dW
        dBeta  = self.dBeta
        dGamma = self.dGamma        
        VdW     = self.VdW
        VdGamma = self.VdGamma
        VdBeta  = self.VdBeta
        num_check_err = 12345
        start_train = now()

        for X, Y, i, epoch, idx in self.batchiterator(iterations):
            #momentum            
            self.backprop(X, Y)

            if numcheck:
                self.num_grad(X,Y)

                ndW     = self.ndW
                ndBeta  = self.ndBeta
                ndGamma = self.ndGamma
                
                dtheta = self.pack_weights(dW, dGamma, dBeta)
                ndtheta = self.pack_weights(ndW, ndGamma, ndBeta)

                num_check_err = err_check(dtheta, ndtheta)


            for l in range(1, L):
                self.VdW[l] = b*self.VdW[l] + nb*dW[l]
                self.VdGamma[l] = b*self.VdGamma[l] + nb*dGamma[l]
                self.VdBeta[l] = b*self.VdBeta[l] + nb*dBeta[l]
                self.W[l] -= a*self.VdW[l]
                self.Beta[l] -= a*self.VdBeta[l]
                self.Gamma[l] -= a*self.VdGamma[l]
            
            #metrics
            if not i % yld:
                train_time = now() - start_train
                start_stats = now()
                train_cost = self.cost(self.A[-1], Y)
                AL = self.predict(self.X_dev)                
                dev_cost = self.cost(AL, self.Y_dev)
                accuracy, precision, recall, f1 = self.statistics(AL, self.Y_dev)



                info = {"iteration"     : i,
                        "epoch"         : epoch,
                        "batch_idx"     : idx,
                        "train_cost"    : np.asscalar(train_cost),
                        "dev_cost"      : np.asscalar(dev_cost),
                        "dev_accuracy"  : np.asscalar(accuracy),
                        "dev_precision" : precision,
                        "dev_recall"    : recall,
                        "dev_f1"        : f1,
                        "report_time"   : now() - start_stats,
                        "train_time"    : train_time, 
                        "num_check_err" : num_check_err}
                start_train = now()
                yield info


    def pack_weights(self, W, Gamma, Beta):
        WBlob = np.concatenate([Wl.ravel() for Wl in W[1:]])
        GammaBlob = np.concatenate([Gammal.ravel() for Gammal in Gamma[1:]])
        BetaBlob = np.concatenate([Betal.ravel() for Betal in Beta[1:]])
        theta = np.concatenate((WBlob, GammaBlob, BetaBlob))
        return theta


    def unpack_weights(self, theta):
        W     = [None]
        Gamma = [None]
        Beta  = [None]
        start = 0
        for dims in self.W_dims[1:]: # for nxm matrix
            mx_size = dims[0] * dims[1]
            end = start+mx_size
            mx  = theta[start:end]
            start = end
            W.append(mx.reshape(*dims))

        for dims in self.Z_dims[1:]: # for nx1 matrix
            mx_size = dims
            end = start+mx_size
            mx  = theta[start:end]
            start = end
            Gamma.append(mx.reshape(dims,1))

        for dims in self.Z_dims[1:]: # for nx1 matrix
            mx_size = dims
            end = start+mx_size
            mx  = theta[start:end]
            start = end
            Beta.append(mx.reshape(dims,1))
        return W, Gamma, Beta


    def num_grad(self, X, Y):
        A       = self.A 
        L       = self.L
        m       = self.minibatch_size
        Beta    = self.Beta
        Gamma   = self.Gamma
        W       = self.W

        epsilon = 1e-7
        
        theta   = self.pack_weights(W, Gamma, Beta)
        dtheta  = np.zeros(theta.shape)
        

        for i in range(theta.size):
            theta[i] += epsilon
            W, Gamma, Beta = self.unpack_weights(theta)
            self.forward(X, W = W, Gamma = Gamma, Beta = Beta, dropout=False, testtime=False)
            rightloss = self.cost(A[-1], Y)

            theta[i] -= 2*epsilon
            W, Gamma, Beta = self.unpack_weights(theta)
            self.forward(X, W = W, Gamma = Gamma, Beta = Beta, dropout=False, testtime=False)
            leftloss = self.cost(A[-1], Y)

            dtheta[i]  = (rightloss - leftloss) / (2*epsilon)

        self.ndW, self.ndGamma, self.ndBeta = self.unpack_weights(dtheta)


    def statistics(self, AL, Y):
        ALraw = one_hot_decoding(AL)
        Yraw = one_hot_decoding(Y)
        eq = (ALraw == Yraw)
        accuracy = np.mean(eq)
        #accuracy, precision, recall, f1
        return accuracy,0,0,0


def list_of_mx_tostr(listmx):
    list_of_dims = [ 'None' if mx is None else mx.shape  for mx in listmx]
    return ' '.join([str(dims) for dims in list_of_dims])



if __name__ == '__main__':
    from dataprep import load_all, convert_for_nn, data_augmentation
    np.set_printoptions(threshold=np.nan)
    train_batches, test_batches = load_all()
    X, Y = convert_for_nn(train_batches)
    X, Y = data_augmentation(X, Y, flip=True, brighter=False, darker=False)
    Xtest, Ytest = convert_for_nn(test_batches)

    x = Multiclass_NN(X,Y, X_dev = Xtest, Y_dev = Ytest, learning_rate=2, keep_prob=[1.0,0.85,0.85,1.0], layers_list=[3072,128,32,10], minibatch_size=256)
    print(f'DONE!\n')
    print(f'x.learning_rate: {x.learning_rate}')
    print(f'x.optimizer: {x.optimizer}')
    print(f'x.layers_list: {x.layers_list}')
    print(f'x.L: {x.L}')
    print(f'x.K: {x.K}')
    print(f'x.W_dims: {x.W_dims}')
    print(f'x.B_dims: {x.B_dims}')

    print(f'x.W (only dims): {list_of_mx_tostr(x.W)}')
    print(f'x.B (only dims): {list_of_mx_tostr(x.B)}')
    print(f'x.VdW (only dims): {list_of_mx_tostr(x.VdW)}')
    print(f'x.VdB (only dims): {list_of_mx_tostr(x.VdB)}')

    print(f'x.Xraw (only dims) {x.Xraw.shape!s}')
    print(f'x.Yraw (only dims) {x.Yraw.shape!s}')

    print(f'x.X_mean shape {x.X_mean.shape}')
    print(f'x.X_std shape {x.X_std.shape}')


    print(f'x.minibatch_size {x.minibatch_size}')

    print(f'x.m_dev {x.m_dev}')

    print(f'x.X_dev (only dims) {x.X_dev.shape!s}')
    print(f'x.Y_dev (only dims) {x.Y_dev.shape!s}')

    print(f'x.m_train {x.m_train}')

    print(f'x.X (only dims) {x.X.shape!s}')
    print(f'x.Y (only dims) {x.Y.shape!s}')

    print(f'x.batches_num {x.batches_num}')

    print(f'x.Xbatches (only dims): {list_of_mx_tostr(x.Xbatches)}')
    print(f'x.Ybatches (only dims): {list_of_mx_tostr(x.Ybatches)}')


    print(f'x.Xbatches (only len): {len(x.Xbatches)}')
    print(f'x.Ybatches (only len): {len(x.Ybatches)}')


    x.forward(0,0,0)

