# ===YOU SHOULD EDIT THIS FUNCTION===
def mean_naive(X):
    """Compute the mean for a dataset by iterating over the dataset
    
    Arguments
    ---------
    X: (N, D) ndarray representing the dataset.
    
    Returns
    -------
    mean: (D, ) ndarray which is the mean of the dataset.
    """
    N, D = X.shape
    mean = np.zeros(D)
    for m in range(D):
        k=0
        for n in range(N):
            k+=X[n,m]
        smean=k/N
        mean[m]=smean
    return mean

# ===YOU SHOULD EDIT THIS FUNCTION===
def cov_naive(X):
    N, D = X.shape
    covariance = np.zeros((D, D))
    for i in range (D):
        eDi=sum(X[:,i])/N
        for j in range (D):
            eDj=sum(X[:,j])/N
            m=0
            for k in range(N):
                m+=(X[k,i]-eDi)*(X[k,j]-eDj)
            co=m/N
            covariance[i,j]=co
    return covariance

# GRADED FUNCTION: DO NOT EDIT THIS LINE

# ===YOU SHOULD EDIT THIS FUNCTION===
def mean(X):
    """Compute the mean for a dataset
    
    Arguments
    ---------
    X: (N, D) ndarray representing the dataset.
    
    Returns
    -------
    mean: (D, ) ndarray which is the mean of the dataset.
    """
    mean = np.zeros(X.shape[1]) # EDIT THIS
    mean = np.mean(X, axis=0) 
    return mean
 
# ===YOU SHOULD EDIT THIS FUNCTION===
def cov(X):
    """Compute the covariance for a dataset
    Arguments
    ---------
    X: (N, D) ndarray representing the dataset.
    
    Returns
    -------
    covariance_matrix: (D, D) ndarray which is the covariance matrix of the dataset.
    
    """
    # It is possible to vectorize our code for computing the covariance, i.e. we do not need to explicitly
    # iterate over the entire dataset as looping in Python tends to be slow
    N, D = X.shape
    covariance_matrix = np.zeros((D, D)) # EDIT THIS
    Y = np.transpose(X) 
    covariance_matrix = np.cov(Y) 
    return covariance_matrix

# GRADED FUNCTION: DO NOT EDIT THIS LINE

# ===YOU SHOULD EDIT THIS FUNCTION===
def affine_mean(mean, A, b):
    """Compute the mean after affine transformation
    Args:
        mean: ndarray, the mean vector
        A, b: affine transformation applied to x
    Returns:
        mean vector after affine transformation
    """
    affine_m = A@mean+b
    return affine_m

# ===YOU SHOULD EDIT THIS FUNCTION===
def affine_covariance(S, A, b):
    v=A@S
    y=np.transpose(A)
    affine_cov=v@y
    return affine_cov