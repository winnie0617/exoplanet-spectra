import numpy as np

def add_noise(data, SNs=None, seed=42):
    ''' 
    Returns np array with specified noise 
    If S/N not specified - draw S/N from uniform(0,100)
    Input should be np arrays
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

def upsample_minority(X, y):
    ''' 
    Returns np arrays X, y with balanced positive and negative samples 
    '''
    # Add noise and aggregate negative samples
    num_pos = y.sum()
    factor = num_pos // (len(y)-num_pos)

    X_neg = X[y == False].to_numpy()
    X_upsampled = np.append(X.to_numpy(), np.repeat(X_neg, factor-1, axis=0), axis=0)
    y_upsampled = np.append(y.to_numpy(), np.repeat(False, len(X_neg)*(factor-1)))
    n = len(y_upsampled)
    idx = np.arange(n)
    np.random.shuffle(idx)
    X = X_upsampled[idx,:]
    y = y_upsampled[idx]

    return X, y