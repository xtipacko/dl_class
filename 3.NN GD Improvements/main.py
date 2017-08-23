from data import batch_1, batch_2, batch_3, batch_4
from data import batch_5, test_batch, batch_meta
from pprint import pprint
from prepareds import dsprep
from visds import visualize_dataset_rnd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from nnbinclf import NN
from time import sleep
import sys
# dp1 = batch_1()
# dpmeta= batch_meta()
# print(len(dp1[b'data'] ))
# print(len(dp1[b'labels'] ))
# print(dpmeta[b'label_names'])

# Classes:
# [b'airplane',    #0
#  b'automobile',  #1
#  b'bird',        #2
#  b'cat',         #3
#  b'deer',        #4
#  b'dog',         #5
#  b'frog',        #6
#  b'horse',       #7
#  b'ship',        #8
#  b'truck']       #9
# plt.ion()
def stats(tp, tn, fp, fn):
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1 = (2*precision*recall) / (precision+recall)
    return accuracy, precision, recall, f1

if __name__ == '__main__':
    np.set_printoptions(threshold=np.nan)
    datapoints = [] 
    datapoints.append(batch_1())
    datapoints.append(batch_2())
    datapoints.append(batch_3())
    datapoints.append(batch_4())
    cv_dp = batch_5()
    XX = []
    YY = []
    for dp in datapoints:
        X = np.array(dp[b'data']).T
        Y = np.array(dp[b'labels'])
        Y = Y.reshape(1, np.size(Y))
        XX.append(X) 
        YY.append(Y)
    X = np.column_stack(XX) #(3072, 40000)
    Y = np.column_stack(YY) #(40000,1)
    print(f'X {X.shape},  Y {Y.shape}')

    X,Y = dsprep(X,Y,0)
    #visualize_dataset_rnd(X,Y,5,8) 
    print(f'X {X.shape},  Y {Y.shape}')


    Xval = np.array(cv_dp[b'data']).T
    Yval = np.array(cv_dp[b'labels']).reshape(1,-1)
    print(f'Xval {Xval.shape},  Yval {Yval.shape}')
    Xval, Yval = dsprep(Xval,Yval,0)
    print(f'Xval {Xval.shape},  Yval {Yval.shape}')


    np.random.seed(122)
    alph = [16, 8, 4, 2, 1, 0.64, 0.32, 0.16, 0.08, 0.06, 0.04, 0.03, 0.02]
    alph = alph[::-1]
    stats_r = []
    from time import time as now
    for a in alph:
        model = NN(X,Y, alpha=a, l=20, layers_list=[3072, 256, 128, 128, 64, 64, 64, 1])
        # model.backprop_numcheck(iterations=1000)
        print('backprop started')
        iterations = 100        
        # accuracy, precision, recall, f1 = stats(*model.accuracy(model.X, model.Y))
        for i in range(20):
            start = now()
            cost = model.J(model.WW, model.BB)
            accuracy, precision, recall, f1 = stats(*model.accuracy(model.X, model.Y))
            cvcost = model.J(model.WW, model.BB, X = Xval, Y = Yval)
            cvaccuracy, cvprecision, cvrecall, cvf1 = stats(*model.accuracy(Xval, Yval))
            model.backprop_minibatch_rmsprop(iterations=iterations, batch_size=256, b2=0.9)
            t = now() - start
            print(f'[{(i+1)*iterations}] 100 itterations took {t:.3f}s\r', end='')
            sys.stdout.flush()

        print(f'For a = {a}')
        print(f'[{(i+1)*iterations:07}] Train: Cost {cost:.7f},  Ac {accuracy:.4f}, Pr {precision:.4f}, Re {recall:.4f}, F1 {f1:.4f};'
                  f'     CVal: Cost {cvcost:.7f}, Ac {cvaccuracy:.4f}, Pr {cvprecision:.4f}, Re {cvrecall:.4f}, F1 {cvf1:.4f}')
        s = { 'cost':cost,
              'accuracy':accuracy,
              'precision,':precision, 
              'recall':recall,
              'f1,':f1, 
              'cvcost':cvcost,
              'cvaccuracy':cvaccuracy,
              'cverr': 1 - cvaccuracy,
              'cvprecision':cvprecision,
              'cvrecall':cvrecall,
              'cvf1':cvf1 }
        stats_r.append(s)
    costs = [stat['cost'] for stat in stats_r]
    cverrs = [stat['cverr'] for stat in stats_r]
    plt.plot(costs, color='r')
    plt.plot(cverrs, color='b')
    plt.show()

    # print(model.WW, end = '\n\n\n\n\n')
    # print(model.BB)