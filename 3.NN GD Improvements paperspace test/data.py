#!/usr/bin/env python3
import pickle


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict


def batch_1():
    return unpickle('../CIFAR-10/data_batch_1')

def batch_2():
    return unpickle('../CIFAR-10/data_batch_2')

def batch_3():
    return unpickle('../CIFAR-10/data_batch_3')

def batch_4():
    return unpickle('../CIFAR-10/data_batch_4')

def batch_5():
    return unpickle('../CIFAR-10/data_batch_5')

def batch_meta():
    return unpickle('../CIFAR-10/batches.meta')

def test_batch():
    return unpickle('../CIFAR-10/test_batch')