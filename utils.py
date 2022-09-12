import matplotlib.pyplot as plt
import numpy as np

def visualize(data,trend,threshold = None, fig_size = (30,10)):
    '''
    data: input data : numpy array
    trend: result of apply trend filter : numpy array
    threshold: for residual : scalar
    fig_size: figure size 
    '''
    fig = plt.figure(figsize=fig_size)
    residual = data - trend
    plt.plot(np.arange(data.shape[0]),data,label = 'data')
    plt.plot(np.arange(data.shape[0]),trend, label = 'trend')
    if threshold:
        idx = np.where(np.abs(residual) > threshold)[0]
        plt.scatter(idx, data[idx],c = 'red')

    plt.legend(loc = 'upper left')
    plt.show()