import numpy as np
from skimage.metrics import structural_similarity
import sklearn.metrics as met


def SSIM(x,y):
    ssim = []
    for i in range(x.shape[0]):
        data_range = np.amax([x[i],y[i]]) - np.amin([x[i],y[i]])
        ssim.append(structural_similarity(x[i], y[i], data_range=data_range))
    return np.asarray(ssim)

def RMS(x,y):
    return met.mean_squared_error(y, x, squared=False, multioutput='raw_values')

def PSNR(x, y):
    mse = np.mean(np.square(np.subtract(x,y)), axis=1)
    if mse.any == 0:
        return np.Inf
    return 20 * np.log10(np.amax([x,y], axis=(0,2))) - 10 * np.log10(mse)

def correlation(x, y):
    corr = []
    for i in range(x.shape[0]):
        corr.append(np.correlate(x[i],y[i]))
    return np.asarray(corr)