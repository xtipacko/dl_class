import numpy as np


def GCD(a, b):
    while b:
        a, b = b, a % b
    return a


def LCM(a, b):
    return a*b // GCD(a, b)


class Multiclass_NN(object):
    def __init__(self, X, Y, learning_rate=1, 
                 X_dev=None, Y_dev=None, 
                 layers_list=[3072, 256, 128, 10], 
                 keep_prob=[1.0,1.0,1.0,1.0], 
                 optimizer='momentum', minibatch_size = 16 ):
        assert isinstance(keep_prob, list), 'keep_probe is a list of probabilities of keeping neurons during dropout in each layer'
        assert len(keep_prob) == len(layers_list), 'keep_prob should have the same length as layers_list'
        assert layers_list[0] == X.shape[0], 'Number of inputs (A0) in layers_list should be equal to number of rows in X'
        assert (X_dev is None) == (Y_dev is None), 'You\'ve forgotten either X_dev or Y_dev'

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.layers_list = layers_list
        self.KP = keep_prob
        self.L = len(layers_list)  

        self.D = [None]*self.L # dropout masks

        self.K = self.layers_list[-1] # number of classes

        self.W_dims, self.B_dims = self.calc_dims(self.layers_list, self.L)
        self.W, self.B = self.He_init_weights(self.W_dims, self.B_dims)
   
        self.VdW, self.VdB = self.init_optimizer(self.W_dims, self.B_dims, optimizer=self.optimizer)

        self.Xraw = X
        self.Yraw = Y               

        self.X, self.X_mean, self.X_std = self.normalize(self.Xraw, gen_stats=True)
        self.Y = self.one_hot_encoding(self.Yraw, self.K)        
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
        self.Y_dev = self.one_hot_encoding(self.Y_dev, self.K)   
        self.m_dev = self.X_dev.shape[1]   
        # least comon multiplier between m and mini-batch size
        m = self.X.shape[1]
        # m_lcm = LCM(m, self.minibatch_size)
        # replicate X, Y (m_lcm // m) times
        # repeat_num = m_lcm // m
        # self.X = np.tile(self.X, repeat_num)
        # self.Y = np.tile(self.Y, repeat_num)
        self.m_train = m
        # self.m_train = m_lcm

        # shuffle X, Y
        # self.X, self.Y = self.shuffle(self.X, self.Y)
        # 6.slice X to  minibatches
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
        

    def shuffle(self, X, Y):
        p = np.random.permutation(Y.shape[1])
        X_shuffled, Y_shuffled = X[:,p], Y[:,p] 
        # Y = np.array(Y, dtype=np.int)
        return X_shuffled, Y_shuffled

    def slice_to_minibatches(self, X, Y, size):
        pass

    def init_optimizer(self, W_dims, B_dims, optimizer='momentum'):
        VdW = [None if dims is None else np.zeros(dims) for dims in W_dims]
        VdB = [None if dims is None else np.zeros((dims,1)) for dims in B_dims]
        return VdW, VdB


    def calc_dims(self, layers_list, L):
        W_dims = [None] + [ (layers_list[l+1], layers_list[l]) for l in range(L-1) ]
        B_dims = [None] + [ layers_list[l] for l in range(1,L) ]
        return W_dims, B_dims


    def He_init_weights(self, W_dims, B_dims):
        init_Wn = lambda dims: np.random.randn(*dims)/np.sqrt(dims[1]/2)
        W = [ None if dims is None else init_Wn(dims) for dims in W_dims ]
        init_Bn = lambda dims: np.zeros((dims[0], 1))
        B = [ None if dims is None else init_Bn(dims) for dims in W_dims ]
        return W, B


    def normalize(self, X_unnorm, gen_stats=False):
        if gen_stats:
            X_mean = np.mean(X_unnorm, axis=1).reshape(-1,1)
            X_std  = np.std(X_unnorm, axis=1).reshape(-1,1)
        else:
            X_mean = self.X_mean
            X_std = self.X_std
        
        X_norm = (X_unnorm - X_mean) / X_std

        return X_norm, X_mean, X_std


    def one_hot_encoding(self, Yraw, K):
        H = np.arange(0, K).reshape(-1,1)
        Y = (H == Yraw).astype(np.int32)
        return Y

    
    def sigmoid(self, Z):
        return np.reciprocal(1 + np.exp(-Z))


    def desigmoid(self, Z):
        sigm = self.sigmoid(Z)
        return (1-sigm)*sigm


    def relu(self, Z):
        A = np.maximum(0, Z)
        return A


    def derelu(self, Z):
        dZ = (Z >= 0).astype(np.int32)
        return dZ


    def softmax(self, Z):
        '''softmax with numerical stabilization'''
        D = -np.max(Z, axis=0)   # (1 x m) - stabilizing term, I'm not sure about it...
        exponents = np.exp(Z + D) # (K x m)
        expsum = np.sum(exponents, axis=0)
        return exponents / expsum


    def desoftmax(self, dA):
        pass  


    def new_dropout_mask(self, KP):        
        for l in range(1,self.L):
            mask = (np.random.rand(self.layers_list[l],1) < KP[l]).astype(np.int32)
            self.D[l] = mask



    def forward(self, X, W, B, dropout=True):
        if dropout:
            self.new_dropout_mask(self.KP)
        else:
            self.new_dropout_mask([1.0]*self.L)
        
        print(self.D)
        

    def predict(self, ):
        pass


    def cost(self, AL, Y):
        pass


    def backprop(self, ):
        pass


    def momentum(self, ):
        pass


    def sgd(self, ):
        pass


    def adam(self, ):
        pass


    def update(self, optimizer='momentum'):
        pass


    def train(self, epochs=10, optimizer='momentum'):
        pass


    def numcheck(self, ):
        pass


    def statistics(self, ):
        #accuracy, precision, recall, f1
        pass


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