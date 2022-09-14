# Define function:
def simple_normalize(data):
    """Input numpy array of data sets, X. Output numpy array of: zero mean and divided by stdev Xs."""
    data = np.array(data) # make sure data is np array.
    for col in list(range(0, data.shape[1])):
        old_col = data[:, col]
        new_col = np.array([(x - old_col.mean())/old_col.std() for x in old_col])
        data[:, col] = new_col
    return data

# Define whitening function:
def whitener(data):
    """Input 2D data set. Output whiten & inv. whiten functions. Taken from colab week 2."""
    sigma = np.cov(data, rowvar=False)
    mu = np.mean(data, axis=0)
    values, vectors = np.linalg.eig(sigma)
    l = np.diag(values ** -0.5)

    def func(datapoints):
        return np.array( [ l.dot(vectors.T.dot(d - mu)) for d in datapoints ])

    def inv_func(datapoints):
        return np.array( [ np.linalg.inv(vectors.T).dot(np.linalg.inv(l).dot(d)) + mu for d in datapoints ] )
  
    return func, inv_func


# Define PCA function:
def pca(X, n_components):
    """Input 2D np.array of data sets. Output np.array of PCA data sets."""
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_pca = pca.transform(X)
    return X_pca

def pca_transform(data):
    """Input 2D np.array of data sets. Output np.array of PCA data sets. Taken for colab week 2."""
    mu = np.mean(data, axis=0)
    sigma = np.cov(data, rowvar=False)
    values, vectors = np.linalg.eig(sigma)
    
    components = sorted( zip(values, vectors.T), key = lambda vv: vv[0], reverse=True )
    
    def func(datapoints):
        result = []
        for d in datapoints:
            t = [ vector.dot(d - mu) for (value, vector) in components ]
            result.append(t)
        return np.array(result)
    
    return func