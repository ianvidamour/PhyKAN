import numpy as np
import torch  
from torch import linalg

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

def intrinsic_dimension2(points2, points1, flag=0, diagnostics=0):
    epsilon = 1e-10
    
    n_points1 = points1.shape[0]
    n_points2 = points2.shape[0] 
    
    # Compute the means and the central point
    mean1 = points1.mean(axis=0)
    mean2 = points2.mean(axis=0)

    if flag == 1:
        centre = mean1
    elif flag == 2:
        centre = mean2
    else:
        centre = (mean1 + mean2) * 0.5
    
    if diagnostics == 1:
        print("Centre shape:", centre.shape)
        print("Centre mean value:", np.mean(centre))

    points1c = points1 - centre
    points2c = points2 - centre    
    
    products = points2c @ points1c.T
    norms = np.sum(points2c * points2c, axis=1)[:, np.newaxis]
    # Compute the pairwise projections (x – y, y - c) as (xy-\y|**2)
    projections = products - norms
    # Compute the probability of separating pairs of points
    p = np.count_nonzero(projections >= 0) / (n_points1 * n_points2)
    # Convert the probability into a dimensionality
    dimensionality= -1 - np.log2(p + epsilon)
    
    
    if diagnostics == 1:
        # Print the computed and max dimensionality
        print("Computed dimensionality:", dimensionality)
        print("Maximum dimensionality (when p=0):", -1 - np.log2(epsilon))
    
    # Convert the probability into a dimensionality
    return dimensionality

    """
    Calculate the intrinsic dimension matrix for all pairs of classes.

    Parameters:
    - Y_tr: PyTorch tensor of shape (n_samples, n_classes), class labels in one-hot encoded format.
    - Data_s_w_np: NumPy array of shape (n_samples, n_features), the data instances.
    - intrinsic_dimension: Function to calculate intrinsic dimension for a single class.
    - intrinsic_dimension2: Function to calculate intrinsic dimension between two different classes.

    Returns:
    - intrinsic_dim_matrix: NumPy array of shape (n_classes, n_classes), intrinsic dimensions for all pairs of classes.
    """

    def get_class_indices(Y, class_label):
        return (Y[:, class_label] == 1).nonzero(as_tuple=True)[0]

    # Number of classes
    N_classes = Y_tr.shape[1]

    # Pre-allocate the matrix
    intrinsic_dim_matrix = np.zeros((N_classes, N_classes))

    # Iterate through all pairs of classes
    for i in range(N_classes):
        for j in range(N_classes):

            # Handle the diagonal separately
            if i == j:
                # Get the indices for the class
                indices_class_i = get_class_indices(Y_tr, i)

                # Extract the instances for this class from Data_s_w_np
                Z_class_i = Data_s_w_np[indices_class_i]

                # Call your intrinsic_dimension function
                dim = intrinsic_dimension(Z_class_i)  # Adjust parameters as needed

            else:
                # Get the indices for each class
                indices_class_i = get_class_indices(Y_tr, i)
                indices_class_j = get_class_indices(Y_tr, j)

                # Extract the instances for these classes from Data_s_w_np
                Z_class_i = Data_s_w_np[indices_class_i]
                Z_class_j = Data_s_w_np[indices_class_j]

                # Call your intrinsic_dimension2 function
                dim = intrinsic_dimension2(Z_class_i, Z_class_j)  # Adjust parameters as needed

            # Store the result in the matrix
            intrinsic_dim_matrix[i, j] = dim

    return intrinsic_dim_matrix

def whiten_torch(Z,covariance_bias=False,variance_explained=1):
    
    mean = torch.mean(Z)
    Z_centered = Z - mean

    cov_matrix = torch.cov(Z_centered.T) #, rowvar=0, bias=covariance_bias
    eigenvalues, eigenvectors = linalg.eigh(cov_matrix)

    # Sort eigenvectors/values in order of which explaines variance the most
    idx = torch.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    # Ensure no negative eigenvalues
    eigenvalues = torch.clip(eigenvalues, min=0, max=None)

    # Calculates the percentage of variance explained by each eigenvector
    cumsum_eigenvalues = torch.cumsum(eigenvalues,axis=0) / torch.sum(eigenvalues, axis=0)
    number_of_components=min((torch.searchsorted(cumsum_eigenvalues,variance_explained)+1), eigenvalues.shape[0])

    principle_components=eigenvectors[:,: number_of_components]
    principle_components_eigenvalues=eigenvalues[: number_of_components]

    epsilon = 1e-12
    
    whiten_matrix = principle_components @  torch.diag(1.0 / torch.sqrt( principle_components_eigenvalues + epsilon))
    print("components: ", principle_components.shape) 
    print("pca eigenvalues:§ ", principle_components_eigenvalues.shape)
    print("whiten_matrix: ", whiten_matrix.shape)
    # Whitening: decorrelate and scale features
    Z_whitened = Z_centered @ whiten_matrix

    return Z_whitened

def whiten_pca_np(Z,covariance_bias=False,variance_explained=0.95):
    
    # Check the type of Z and convert it to a numpy array if necessary
    if isinstance(Z, torch.Tensor):
        Z_np = Z.numpy()
    elif isinstance(Z, np.matrix):
        Z_np = np.array(Z)
    else:  # assuming Z is a numpy array
        Z_np = Z

    # Calculate the mean and subtract it
    mean = np.mean(Z_np, axis=0)
    Z_centered = Z_np - mean

    # Compute the covariance matrix
    cov_matrix = np.cov(Z_centered, rowvar=0, bias=covariance_bias) # should be the same   

    # Perform eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    # Clip the eigenvalues to be non-negative
    eigenvalues = np.clip(eigenvalues, a_min=0, a_max=None)

    # Add debugging information
    #print("Shape of eigenvalues:", eigenvalues.shape)
    #print("Any NaN in eigenvalues:", np.any(np.isnan(eigenvalues)))
    #print("Any zeros in sum:", np.any(np.sum(eigenvalues, axis=0) == 0))
    #print("Min eigenvalue:", np.min(eigenvalues))
    #print("Max eigenvalue:", np.max(eigenvalues))

    # Add safety check for division
    cumsum_eigenvalues = np.zeros_like(eigenvalues)
    sum_eigenvalues = np.sum(eigenvalues, axis=0)

    # Handle cases where sum is zero or contains NaN
    mask = sum_eigenvalues != 0
    if np.any(mask):
        cumsum_eigenvalues[:,mask] = np.cumsum(eigenvalues[:,mask], axis=0)/sum_eigenvalues[mask]

    # Finde the principle components that explain variance in the data equal to variance_explained
    cumsum_eigenvalues=np.cumsum(eigenvalues,axis=0)/np.sum(eigenvalues, axis=0)

    number_of_components=min((np.searchsorted(cumsum_eigenvalues,variance_explained)+1),eigenvalues.shape[0])

    principle_components=eigenvectors[:,: number_of_components]
    principle_components_eigenvalues=eigenvalues[: number_of_components]

    # Compute the diagonal matrix of inverse square roots of eigenvalues of the principle components
    epsilon = 1e-10
    whiten_matrix = principle_components @  np.diag(1.0 / np.sqrt( principle_components_eigenvalues + epsilon))

    # Whitening: decorrelate and scale features
    Z_whitened = Z_centered @ whiten_matrix
    return Z_whitened
