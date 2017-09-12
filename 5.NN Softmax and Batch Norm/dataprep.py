import numpy as np
import pickle
from visds import visualize_dataset_rnd

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

def load_all():
    train_batches_num = 5
    train_batches = [None]*(train_batches_num)
    for i, _ in enumerate(train_batches):
        train_batches[i] = unpickle(f'../CIFAR-10/data_batch_{i+1}')
    #batch_meta = unpickle('../CIFAR-10/batches.meta')
    test_batches = [ unpickle('../CIFAR-10/test_batch') ]
    return train_batches, test_batches

def convert_for_nn(batches):
    X_lst = []
    Y_lst = []
    for batch in batches:
        X = np.array(batch[b'data']).T
        Y = np.array(batch[b'labels'])
        Y = Y.reshape(1,-1)
        X_lst.append(X)
        Y_lst.append(Y)
    X = np.column_stack(X_lst)
    Y = np.column_stack(Y_lst)
    return X, Y
    
def flip_images(X):
    X = X.reshape(3,32,32,-1)
    # X = np.swapaxes(X,0,2)
    # X = np.swapaxes(X,0,1)
    X = np.flip(X, axis=2)
    # X = np.swapaxes(X,0,1)
    # X = np.swapaxes(X,0,2)
    X = X.reshape(3072,-1)
    return X

def brightness_shift(X, brightness=1.15):
    # it is better to scale components, not just add/substract
    assert 0 <= brightness <= 2, 'brightness should be between 0 and 2'
    X = X.astype(np.int16)
    shift = np.array(np.rint(255*(brightness-1)), dtype=np.int16)
    X = X+shift 
    X = np.clip(X, 0, 255)
    return X.astype(np.uint8)



def slight_noise(X):
    pass


def data_augmentation(X, Y, flip=True, brighter=True, darker=True):
    m = X.shape[1]
    # flipping image
    if flip:
        flipped_X = flip_images(X)
        X = np.column_stack([X, flipped_X])
        Y = np.column_stack([Y, Y])
    # adding brighter and darker image
    X_flipped = X
    Y_flipped = Y
    if brighter:
        X_brighter = brightness_shift(X_flipped, brightness=1.15)
        X = np.column_stack([X, X_brighter])
        Y = np.column_stack([Y, Y_flipped])
    if darker:
        X_darker = brightness_shift(X_flipped, brightness=0.85)
        X = np.column_stack([X, X_darker])
        Y = np.column_stack([Y, Y_flipped])
    return X, Y


if __name__ == '__main__':
    np.set_printoptions(threshold=np.nan)
    train_batches, test_batches = load_all()
    X, Y = convert_for_nn(train_batches)
    print(Y.dtype)
    X, Y = data_augmentation(X, Y)
    print(f'Xtrain shape: {np.shape(X)}, Ytrain shape: {np.shape(Y)}')
    Xtest, Ytest = convert_for_nn(test_batches)
    print(X.dtype)
    #print(f'Xtest shape: {np.shape(Xtest)}, Ytest shape: {np.shape(Ytest)}')
    visualize_dataset_rnd(X,Y,5,5)
    #visualize_dataset_rnd(Xtest, Ytest,5,5)


