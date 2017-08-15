from data import batch_1, batch_2, batch_3, batch_4
from data import batch_5, test_batch, batch_meta
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from linclf import LR
# dp1 = batch_1()
# dpmeta= batch_meta()
# print(len(dp1[b'data'] ))
# print(len(dp1[b'labels'] ))
# print(dpmeta[b'label_names'])

# Classes:
# [b'airplane',    #0
#  b'automobile', 
#  b'bird',
#  b'cat', 
#  b'deer', 
#  b'dog', 
#  b'frog', 
#  b'horse', 
#  b'ship', 
#  b'truck']
datapoints = batch_1()
datapoints2 =  batch_2()

if __name__ == '__main__':
    X = np.array(datapoints[b'data']).T         #(3072, 10000)
    #X = X[:128,:10]
    Y = np.array(datapoints[b'labels'])#[:10] #(10000,)
    Y = np.array(Y == 0, dtype=np.int)
    Y = Y.reshape(1, np.size(Y))

    w = np.zeros((X.shape[0],1)) #np.random.randn(X.shape[0],1)        #(3072, 1)
    b = 0
    
    Xval = np.array(datapoints2[b'data']).T
    Yval = np.array(datapoints2[b'labels'])
    Yval = np.array(Y == 0, dtype=np.int)
    Yval = Yval.reshape(1, np.size(Yval))

    model = LR(X,Y,w,b, alpha=0.001, l=0.1)
    model.backprop_numcheck(iterations=10)
    accuracy = model.accuracy(model.X, model.Y)
    valaccuracy = model.accuracy(Xval, Yval)
    print(f'Training accuracy {accuracy}')
    for i in range(1000):
        model.backprop(iterations=10)
        print(model.J(model.w, model.b))
        accuracy = model.accuracy(model.X, model.Y)
        valaccuracy = model.accuracy(Xval, yval)
        print(f'Training accuracy {accuracy} Validation accuracy {valaccuracy}')





