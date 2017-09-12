from visds import visualize_formated_img
import numpy as np
import matplotlib.pyplot as plt


def dsprep(X,Y,class_num, seed=2):
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
    X = np.array(X)
    shape = (1,X_true.shape[1])
    Y = np.column_stack((np.ones(shape), np.zeros(shape)))

    #shuffling
    p = np.random.permutation(Y.shape[1])
    X,Y = X[:,p],Y[:,p] 
    Y = np.array(Y, dtype=np.int)
    
    return X,Y


def vischeck(*args, idx=0):
    datasets = args
    img = []
    for ds in datasets:
        ds = ds.reshape(3,32,32,-1)
        ds = np.swapaxes(ds,0,2)
        ds = np.swapaxes(ds,0,1)
        img.append(ds[:,:,:,idx])

    visualize_formated_img(*img)


def flipimg_ds(ds):
    ds = ds.reshape(3,32,32,-1)
    # ds = np.swapaxes(ds,0,2)
    # ds = np.swapaxes(ds,0,1)
    ds = np.flip(ds, axis=2)
    # ds = np.swapaxes(ds,0,1)
    # ds = np.swapaxes(ds,0,2)
    ds = ds.reshape(3072,-1)
    return ds


def brightness_shift(ds):
    pass


def augmentds(X,Y):
    '''accepts and returns dataset in standard for NN format '''
    idx=7
    # 1. add flipped images     
    m = X.shape[1]
    flipped_X = flipimg_ds(X)    
    X = np.column_stack([X, flipped_X])
    Y = np.column_stack([Y, Y])
    # print(X.shape)
    # print(Y.shape)
    # print(Y[0,19], Y[0,m+19])
    # vischeck(X,X[:,m:], idx=19)
    
    # visualize_formated_img(img, flipped_img)

    return X, Y
