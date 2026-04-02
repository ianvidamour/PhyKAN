import numpy as np

def intrinsic_dimension(points,diagnostics=0):
    epsilon=1e-16
    n_points = points.shape[0]
    mean = points.mean(axis=0) 
    centred = points - mean
    products = centred @ centred.T
    norms = np.diag(products)
    # Compute the pairwise projections (x – y, y - c)
    projections = products - norms
    # Compute the probability of separating pairs of points
    p = (np.count_nonzero(projections >= 0) - n_points) / (n_points * (n_points - 1))
    # Convert the probability into a dimensionality
    dimensionality=-1 - np.log2(p+epsilon)
    # Print the computed and max dimensionality
    
    if diagnostics == 1:
        # Print the computed and max dimensionality
        print("Computed dimensionality:", dimensionality)
        print("Maximum dimensionality (when p=0):", -1 - np.log2(epsilon))

    return dimensionality