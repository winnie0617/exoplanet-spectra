import numpy as np

def add_noise(data):
    ''' return np array with specified noise '''
    np.random.seed(42) # Reproducible result
    SNs = np.random.uniform(low=1, high=100, size=data.shape[0])
    # 2d array: SNs[realization][BVRI]
    SNs = np.repeat(SNs, 4, axis=0).reshape(data.shape[0], 4)
    X_noisy = data+ np.random.normal(loc=0, scale=data/SNs)
    while np.any(X_noisy < 0): # Avoid negative flux
        X_noisy[X_noisy < 0] = data[X_noisy < 0] + np.random.normal(loc=0, scale=data/SNs)
    return X_noisy