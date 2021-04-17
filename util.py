import numpy as np

def add_noise(data, SNs=None, seed=42):
    ''' 
    return np array with specified noise 
    If S/N not specified - draw S/N from uniform(0,100)
    '''
    np.random.seed(42) # Reproducible result
    if SNs is None: # draw S/N
        SNs = np.random.uniform(low=1, high=100, size=data.shape[0])
        # 2d array: SNs[realization][BVRI]
        SNs = np.repeat(SNs, 4, axis=0).reshape(data.shape[0], 4)
    X_noisy = data + np.random.normal(loc=0, scale=data/SNs)
    while np.any(X_noisy < 0): # Avoid negative flux
        idx = np.argwhere(X_noisy < 0)
        X_noisy[idx] = data[idx] + np.random.normal(loc=0, scale=(data/SNs)[idx])
    return X_noisy
