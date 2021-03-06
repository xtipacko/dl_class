#!/usr/bin/python3.6
from dataprep import load_all, convert_for_nn, data_augmentation
from nnmclf import Multiclass_NN
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pickle 
import os.path

def cv(X, Y, Xtest, Ytest):    
    lr_gen = lambda: 3**(np.random.rand()*6-4)
    kp_gen = lambda: np.random.rand()*0.6+0.4
    kpl_gen = lambda layers: [1.0]+[kp_gen()]*(len(layers)-2)+[1.0]
    layouts = [[3072,  1024,  1024,   512,   512,   512,   256,   256,  128, 128,  10],
               [3072,   512,   512,   256,   256,   256,   512,   512,   10],
               [3072,   256,   512,   256,   128,   64,    32,     16,   10],]
               #[3072,  3072,   512,   128,    10]]

    for j, layers_list in enumerate(layouts):
        for i in range(60):            
            learning_rate = lr_gen() #0.03
            keep_prob     = kpl_gen(layers_list) #[1.0,    0.8,   0.8,   0.8,   0.8,   0.8,   0.8,   0.8,  1.0]
           # layers_list = #[3072,   512,   512,   256,   256,   256,   512,   512,   10]    
            #fname = f'weights_a={learning_rate}, Li={str(layers_list)}, KP={str(keep_prob)}_.pickle'
            
            print(f'[{j:>02}, {i:>03}] MODEL DESCRIPTION: a={learning_rate}, KP={keep_prob[1]}, Li={str(layers_list)}')
            model = Multiclass_NN(X,Y, 
                                  X_dev = Xtest, Y_dev = Ytest, 
                                  learning_rate=learning_rate, 
                                  keep_prob=keep_prob, 
                                  layers_list=layers_list, 
                                  minibatch_size=256,
                                  preinitialized=bool(i))
            if i == 0:
                tmpX, tmpX_mean, tmpX_std  = model.X, model.X_mean, model.X_std
                tmpY                           = model.Y
                tmpminibatch_size              = model.minibatch_size
                tmpX_dev                       = model.X_dev
                tmpY_dev                       = model.Y_dev
                tmpm_dev                       = model.m_dev
                tmpm_train                     = model.m_train
                tmpbatches_num                 = model.batches_num
                tmpXbatches                    = model.Xbatches
                tmpYbatches                    = model.Ybatches
                tmpbatches_num                 = model.batches_num
                tmpbatchiterator               = model.batchiterator
            else:
                model.X, model.X_mean, model.X_std  = tmpX, tmpX_mean, tmpX_std  
                model.Y                           = tmpY                           
                model.minibatch_size              = tmpminibatch_size              
                model.X_dev                       = tmpX_dev                       
                model.Y_dev                       = tmpY_dev                       
                model.m_dev                       = tmpm_dev                       
                model.m_train                     = tmpm_train                     
                model.batches_num                 = tmpbatches_num                 
                model.Xbatches                    = tmpXbatches                    
                model.Ybatches                    = tmpYbatches                    
                model.batches_num                 = tmpbatches_num                 
                model.batchiterator               = tmpbatchiterator       
            template = '[{iteration:06}, epoch:{epoch}, {train_time:.3f}/{report_time:.3f} ms] Train cost: {train_cost:.7f};   Dev cost: {dev_cost:.7f}, accuracy: {dev_accuracy:.4f}'
    
            print('backprop started')
            try:
                for info in model.momentum_train(iterations=10000, yld=1000):        
                    print(template.format(**info))
                    if (np.isnan(info["train_cost"]) or info['dev_accuracy'] < 0.44 ) and  info["iteration"] >= 1000:
                        break
            finally:    
                if info["dev_accuracy"] > 0.55:
                    with open('costs.txt', 'a+') as f:
                        print(f'[{j:>02}, {i:>03}][Cost: {info["train_cost"]:.7f}; Dev cost: {info["dev_cost"]:.7f}, accuracy: {info["dev_accuracy"]:.4f}]: a={learning_rate:.5f}, KP={keep_prob[1]}, Li={str(layers_list)}', file=f)




def train(X, Y, Xtest, Ytest, store_result=True, use_stored_result=True, numcheck=False):
    learning_rate = 2

    # keep_prob =   [ 1.0,  0.6,   0.6,   0.6,   0.6,  0.6,  0.6,  0.6,  0.6,  0.6,  0.6,    1] 
    # layers_list = [3072,  1024,  512,   512,   512,  256,  256,  256,  128,  128,  128,  10]

    keep_prob   = [ 1.0,   1.0,   1.0,    1.0, 1.0] 
    layers_list = [3072,  16,    16,     16,  10]
    fname = f'weights_a={learning_rate}, Li={str(layers_list)}, KP={str(keep_prob)}_.pickle'
    print('model created')
    model = Multiclass_NN(X,Y, 
                          X_dev = Xtest, Y_dev = Ytest, 
                          learning_rate=learning_rate, 
                          keep_prob=keep_prob, 
                          layers_list=layers_list, 
                          minibatch_size=4)

    template = '[{iteration:06}, epoch:{epoch}, {train_time:.3f}/{report_time:.3f} ms] Train cost: {train_cost:.7f};   Dev cost: {dev_cost:.7f}, accuracy: {dev_accuracy:.4f}'
    if numcheck:
        template += ', {num_check_err:.12f}'
    if os.path.isfile(fname) and use_stored_result:
        with open(fname, 'rb') as f:
            W, Beta, Gamma = pickle.load(f)
        model.W = W
        model.Beta = Beta
        model.Gamma = Gamma
        print('weights loaded')
    print('backprop started')
    try:
        for info in model.momentum_train(iterations=10, yld=1, numcheck=True):        
            print(template.format(**info))
            # if np.isnan(info["train_cost"]) and  info["iteration"] >= 80:
            #     break
    finally:    
        if store_result:
            weights = model.W, model.Beta, model.Gamma        
            with open(fname, 'wb+') as f:
                pickle.dump(weights, f)
    


    
def main():    
    np.set_printoptions(threshold=np.nan)

    train_batches, test_batches = load_all()
    X, Y = convert_for_nn(train_batches)
    X, Y = data_augmentation(X, Y, flip=True, brighter=True, darker=True)
    Xtest, Ytest = convert_for_nn(test_batches)

    #cv(X, Y, Xtest, Ytest)
    train(X[:, :10000], Y[:, :10000], Xtest[:, :1000], Ytest[:, :1000], store_result=False, use_stored_result=False, numcheck=True)


if __name__ == '__main__':
    main()
