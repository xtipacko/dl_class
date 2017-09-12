#!/usr/bin/env python
from data import batch_1, batch_2, batch_3, batch_4
from data import batch_5, test_batch, batch_meta
from pprint import pprint
from prepareds import dsprep, augmentds
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


def add_line_err_func(val, step, last_point, color='red', label='', linestyle='-'):
    if last_point is None:
        last_point = (0,val)        
    p1 = last_point
    p2 = (last_point[0]+step, val)
    plt.plot((p1[0],p2[0]), (p1[1],p2[1]), color=color, label=label, linestyle=linestyle)
    plt.pause(0.01)
    return p2

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
    print('\n\n\n\n')
    
    X,Y = augmentds(X,Y)
    X,Y = dsprep(X,Y,0)   
    #visualize_dataset_rnd(X,Y,5,8)

    print(f'X {X.shape},  Y {Y.shape}')


    Xval = np.array(cv_dp[b'data']).T
    Yval = np.array(cv_dp[b'labels']).reshape(1,-1)
    print(f'Xval {Xval.shape},  Yval {Yval.shape}')
    Xval, Yval = dsprep(Xval,Yval,0)
    print(f'Xval {Xval.shape},  Yval {Yval.shape}')
        

    plt.ion()
    plt.rcParams['toolbar'] = 'None'
    plt.rcParams['figure.facecolor'] = (0,0,0)
    plt.rcParams['axes.facecolor']   = (0,0,0)
    plt.rcParams['axes.edgecolor']   = (1,1,1)
    plt.rcParams['axes.labelcolor']  = (1,1,1)
    plt.rcParams['legend.facecolor'] = (0,0,0)      
    plt.rcParams['legend.edgecolor'] = (1,1,1)
    plt.rcParams['xtick.color']      = (1,1,1)
    plt.rcParams['ytick.color']      = (1,1,1)
    plt.rcParams['text.color']       = (1,1,1)
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()
    # cost_last_point = None
    # err_last_point =  None


    from time import time as now

    rand_probab = lambda: np.random.rand()*0.7 + 0.3
    rand_alpha = lambda: 5**(1-4*np.random.rand())



    # searchspace = [(rand_probab(), rand_alpha()) for i in range(100)]
    searchspace = [ (2, 0.8), (1, 0.9), (0.5, 0.7),     (0.5, 0.8), (0.5, 0.7), (0.2, 0.8) ]
    colors      = ['violet', 'skyblue', 'springgreen',  'aqua',     'salmon',   'whitesmoke' ]
    iterations = 100
    cycles = 1000

    stats_r = []
    legend = False
       
    np.random.seed(122)
    start_probe = now()
    #[3072, 3072, 6144, 1024, 1024, 256, 256, 64, 64, 1]
    models_data = []
    for (a, p), color in zip(searchspace, colors):
        model = NN(X,Y, alpha=a, layers_list=[3072, 512, 256, 128, 128, 64, 64, 1], keep_prob=[1, p, p, p, p, p, p, 1])
        model_data = [model, a, p, None, None, color]
        models_data.append(model_data)
    # model.backprop_numcheck(iterations=1000)
    print('backprop started')
    
    
    for i in range(cycles):
        start_cycle = now()
        #model.backprop_minibatch_adam(iterations=iterations, realiter=i*iterations, batch_size=256, b1=0.9, b2=0.999)
        for j, (model, a, p, cost_last_point, err_last_point, color)  in enumerate(models_data):     
            model.backprop_minibatch_momentum(iterations=iterations, batch_size=128, b=0.9)
            t_cycle = now() - start_cycle
            print(f'[{(i+1)*iterations}][{t_cycle:.3f}s]\r', end='')
            sys.stdout.flush()
    
            cost = model.J(model.WW, model.BB)
            cvcost = model.J(model.WW, model.BB, X = Xval, Y = Yval)
            cvaccuracy, cvprecision, cvrecall, cvf1 = stats(*model.accuracy(Xval, Yval))
        
            t_probe = now() - start_probe
            report = (f'a: {f"{a:.7f}":<12} p: {f"{p:.7f}":<12} Train Cost: {f"{cost:.7f}":<12}'
                      f'| CVal Accuracy: {f"{cvaccuracy:.4f}":<12} F1: {f"{cvf1:.4f}":<12} Cost: {f"{cvcost:.7f}":<12}')
            print(f'[{(i+1)*iterations}][{t_probe:.3f}s] {report}')
            cverr = 1 - cvaccuracy
    
            # models_data[j][3] = add_line_err_func(cost, 1, cost_last_point, color=color, linestyle='-', label='Train cost a={a:.2f} p={p:.2f}')
            models_data[j][4] = add_line_err_func(cverr, 1, err_last_point, color=color, linestyle='--', label=f'CV errors a={a:.2f} p={p:.2f}')
        if not legend:
            plt.legend()
            legend = True
        plt.pause(0.01)

    #     s = { 'cost':cost,
    #           # 'accuracy':accuracy,
    #           # 'precision,':precision, 
    #           # 'recall':recall,
    #           # 'f1,':f1, p: {f"{p:.7f}":<12} 
    #           'cvcost':cvcost,
    #           'cvaccuracy':cvaccuracy,
    #           'cverr': cverr,
    #           'cvprecision':cvprecision,
    #           'cvrecall':cvrecall,
    #           'cvf1':cvf1,
    #           'a':a,
    #           'p':p
    #           }
    #     stats_r.append(s)
    #     cost_last_point = add_line_err_func(cost, 1, cost_last_point, color='violet', label='Train Costs')
    #     err_last_point = add_line_err_func(cverr, 1, err_last_point, color='skyblue', label='CV errors')
    #     if not legend:
    #         plt.legend()
    #         legend = True
    #     plt.pause(0.01)
    
    
    # print('\n\n\nSorted by accuracy:')
    # kf_err = lambda x: x['cverr']
    # best_accuracies = sorted(stats_r, key=kf_err)
    # with open('accuracy.txt', 'a+') as f:
    #     for stat in best_accuracies:
    #         (cost,cvcost,cvaccuracy,cverr,
    #          cvprecision,cvrecall,cvf1,a,p) = (stat['cost'],stat['cvcost'],stat['cvaccuracy'],
    #                                            stat['cverr'],stat['cvprecision'],stat['cvrecall'],
    #                                            stat['cvf1'],stat['a'],stat['p'])
    #         report = (f'a: {f"{a:.7f}":<12} p: {f"{p:.7f}":<12} Train Cost: {f"{cost:.7f}":<12}'
    #                   f'| CVal Accuracy: {f"{cvaccuracy:.4f}":<12} F1: {f"{cvf1:.4f}":<12} Cost: {f"{cvcost:.7f}":<12}')
    #         print(report, file=f)
    #         print(report)
    #     cverrs = [stat['cverr'] for stat in best_accuracies]


    # print('\n\n\nSorted by training cost:')
    # kf_cost = lambda x: x['cost']
    # best_costs = sorted(stats_r, key=kf_cost)[::-1]
    # with open('costs.txt', 'a+') as f:
    #     for stat in best_costs:
    #         (cost,cvcost,cvaccuracy,cverr,
    #          cvprecision,cvrecall,cvf1,a,p) = (stat['cost'],stat['cvcost'],stat['cvaccuracy'],
    #                                            stat['cverr'],stat['cvprecision'],stat['cvrecall'],
    #                                            stat['cvf1'],stat['a'],stat['p'])
    #         report = (f'a: {f"{a:.7f}":<12} p: {f"{p:.7f}":<12} Train Cost: {f"{cost:.7f}":<12}'
    #                   f'| CVal Accuracy: {f"{cvaccuracy:.4f}":<12} F1: {f"{cvf1:.4f}":<12} Cost: {f"{cvcost:.7f}":<12}')
    #         print(report, file=f)
    #         print(report)
    #     costs = [stat['cost'] for stat in best_costs]


    # plt.close()
    # plt.ioff()    
    # plt.plot(costs, color='violet', label='Train costs')
    # plt.plot(cverrs, color='skyblue', label='CV errors')
    # plt.legend()
    # plt.show()
