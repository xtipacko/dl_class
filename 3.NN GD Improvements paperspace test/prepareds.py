#!/usr/bin/env python3
import numpy as np


def dsprep(X,Y,class_num, seed=1):
    '''1. binarizes dataset by class_num, class_num becomes class 1, other 0
       2. equalizes 1 and 0 class amount of training examples
       3. shuffles
    '''    
    assert Y.shape[1] == X.shape[1], 'Y and X shapes are not equal'
    np.random.seed(seed)
    #binarization
    Y = np.array(Y == class_num, dtype=np.int)

    #equalization
    Y_fmask = np.broadcast_to(Y, (X.shape[0],Y.shape[1]))
    X_false = np.ma.masked_array(X,mask=Y_fmask)
    X_false = np.ma.compress_cols(X_false)
    Y_false = np.ma.compress_cols(Y)
    Y = 1 - Y
    Y_tmask = np.broadcast_to(Y, (X.shape[0],Y.shape[1]))
    X_true = np.ma.masked_array(X,mask=Y_tmask)
    X_true = np.ma.compress_cols(X_true)
    Y_true = np.ma.compress_cols(Y)  
    X = np.column_stack((X_true, X_false[:,0:X_true.shape[1]]))
    shape = (1,X_true.shape[1])
    Y = np.column_stack((np.ones(shape), np.zeros(shape)))

    #shuffling
    p = np.random.permutation(Y.shape[1])
    X,Y = X[:,p],Y[:,p] 
    Y = np.array(Y, dtype=np.int)
    
    return X,Y