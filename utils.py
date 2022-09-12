import matplotlib.pyplot as plt
import numpy as np


def visualize(data,trend,threshold = None, fig_size = (30,20)):
    '''
    data: input data : numpy array
    trend: result of apply trend filter : numpy array
    threshold: for residual : scalar
    fig_size: figure size 
    '''
    fig = plt.figure(figsize=fig_size)
    #TREND & DATA
    # plt.subplot(2,1,1)
    residual = data - trend
    plt.plot(np.arange(data.shape[0]),data,label = 'data')
    plt.plot(np.arange(data.shape[0]),trend, label = 'trend')
    if threshold:
        idx = np.where(np.abs(residual) > threshold)[0]
        plt.scatter(idx, data[idx],c = 'red')

    #RESIDUAL
    plt.subplot(2,1,2)
    plt.plot(np.arange(data.shape[0]),data - trend)

    plt.legend(loc = 'upper left')
    plt.show()