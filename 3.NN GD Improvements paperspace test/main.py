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


    X,Y = dsprep(X,Y,0)
    #visualize_dataset_rnd(X,Y,5,8) 



    Xval = np.array(cv_dp[b'data']).T
    Yval = np.array(cv_dp[b'labels']).reshape(1,-1)

    Xval, Yval = dsprep(Xval,Yval,0)



    
    from time import time as now
    from  itertools import product as listproduct
    alph = [0.15, 0.3, 0.53, 0.76, 1, 1.22]
    lmbd = [10, 14, 17, 20, 23, 26]
    searchspace = listproduct(lmbd, alph)

    stats_r = []

    for l,a in searchspace:
        np.random.seed(122)
        model = NN(X,Y, alpha=a, l=l, layers_list=[3072, 256, 128, 128, 64, 64, 64, 64, 1])
        # model.backprop_numcheck(iterations=1000)
        print('backprop started')
        iterations = 500
        # accuracy, precision, recall, f1 = stats(*model.accuracy(model.X, model.Y))
        for i in range(10):
            start = now()
            # cost = model.J(model.WW, model.BB)
            # accuracy, precision, recall, f1 = stats(*model.accuracy(model.X, model.Y))
            # cvcost = model.J(model.WW, model.BB, X = Xval, Y = Yval)
            # cvaccuracy, cvprecision, cvrecall, cvf1 = stats(*model.accuracy(Xval, Yval))
            model.backprop_minibatch_momentum(iterations=iterations, batch_size=256, b=0.9)
            t = now() - start
            print('[{iter}] {iterations} itterations took {t:.3f}s\r'.format(iter=(i+1)*iterations, iterations=iterations, t=t), end='')
            sys.stdout.flush()

        #averaging over 5 last iterations     
        # 1
        cost = model.J(model.WW, model.BB)
        cvcost = model.J(model.WW, model.BB, X = Xval, Y = Yval)
        cvaccuracy, cvprecision, cvrecall, cvf1 = stats(*model.accuracy(Xval, Yval))

        # 2
        model.backprop_minibatch_momentum(iterations=1, batch_size=256, b=0.9)
        cost += model.J(model.WW, model.BB)
        cvcost += model.J(model.WW, model.BB, X = Xval, Y = Yval)
        ncvaccuracy, ncvprecision, ncvrecall, ncvf1 = stats(*model.accuracy(Xval, Yval))

        cvaccuracy += ncvaccuracy
        cvprecision += ncvprecision
        cvrecall += ncvrecall
        cvf1 += ncvf1

        # 3
        model.backprop_minibatch_momentum(iterations=1, batch_size=256, b=0.9)
        cost += model.J(model.WW, model.BB)
        cvcost += model.J(model.WW, model.BB, X = Xval, Y = Yval)
        ncvaccuracy, ncvprecision, ncvrecall, ncvf1 = stats(*model.accuracy(Xval, Yval))

        cvaccuracy += ncvaccuracy
        cvprecision += ncvprecision
        cvrecall += ncvrecall
        cvf1 += ncvf1

        # 4
        model.backprop_minibatch_momentum(iterations=1, batch_size=256, b=0.9)
        cost += model.J(model.WW, model.BB)
        cvcost += model.J(model.WW, model.BB, X = Xval, Y = Yval)
        ncvaccuracy, ncvprecision, ncvrecall, ncvf1 = stats(*model.accuracy(Xval, Yval))

        cvaccuracy += ncvaccuracy
        cvprecision += ncvprecision
        cvrecall += ncvrecall
        cvf1 += ncvf1

        # 5
        model.backprop_minibatch_momentum(iterations=1, batch_size=256, b=0.9)
        cost += model.J(model.WW, model.BB)
        cvcost += model.J(model.WW, model.BB, X = Xval, Y = Yval)
        ncvaccuracy, ncvprecision, ncvrecall, ncvf1 = stats(*model.accuracy(Xval, Yval))

        cvaccuracy += ncvaccuracy
        cvprecision += ncvprecision
        cvrecall += ncvrecall
        cvf1 += ncvf1

        cost /= 5
        cvcost /= 5
        cvaccuracy /= 5
        cvprecision /= 5
        cvrecall /= 5
        cvf1  /= 5
        #avereging completed

        print('\n[{iter:07}]  a = {a:>5}, l = {l:>3}, Train Cost: {cost:.7f}, CVal Accuracy: {cvaccuracy:.4f}, F1 {cvf1:.4f}, Cost: {cvcost:.7f}'.format(iter=(i+1)*iterations, iterations=iterations, a=a, l=l, cost=cost, cvaccuracy=cvaccuracy, cvf1=cvf1, cvcost=cvcost))
        cverr = 1 - cvaccuracy
        #print(f'[{(i+1)*iterations:07}] Train: Cost {cost:.7f},  Ac {accuracy:.4f}, Pr {precision:.4f}, Re {recall:.4f}, F1 {f1:.4f};'
        #          f'     CVal: Cost {cvcost:.7f}, Ac {cvaccuracy:.4f}, Pr {cvprecision:.4f}, Re {cvrecall:.4f}, F1 {cvf1:.4f}')
        s = { 'cost':cost,
              # 'accuracy':accuracy,
              # 'precision,':precision, 
              # 'recall':recall,
              # 'f1,':f1, 
              'cvcost':cvcost,
              'cvaccuracy':cvaccuracy,
              'cverr': cverr,
              'cvprecision':cvprecision,
              'cvrecall':cvrecall,
              'cvf1':cvf1,
              'a':a,
              'l':l
              }
        stats_r.append(s)

    print('Best 5:')
    kf = lambda x: x['cverr']
    best5 = sorted(stats_r, key=kf)
    for i in best5[:5]:
        a,l,cost,cvaccuracy,cvf1,cvcost = i['a'],i['l'],i['cost'],i['cvaccuracy'],i['cvf1'],i['cvcost']
        print('a = {a:>5}, l = {l:>3}, Train Cost: {cost:.7f}, CVal Accuracy: {cvaccuracy:.4f}, F1 {cvf1:.4f}, Cost: {cvcost:.7f}'.format(a=a, l=l, cost=cost, cvaccuracy=cvaccuracy, cvf1=cvf1, cvcost=cvcost))
    costs = [stat['cost'] for stat in stats_r]
    cverrs = [stat['cverr'] for stat in stats_r]
    plt.plot(costs, color='r')
    plt.plot(cverrs, color='b')
    plt.show()

    # print(model.WW, end = '\n\n\n\n\n')
    # print(model.BB)