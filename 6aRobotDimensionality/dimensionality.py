import numpy as np
import torch


def select_device(dev):
    """
    Get an appropriate pytorch device.
    The key feature is that 
        select_device('gpu')
    will automatically return either cuda or MPS
    depending on what's available.
    """
    if type(dev) is torch.device:
        return dev
    elif dev == 'gpu':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device('cpu')
    else:
        return torch.device(dev)


def clear_cuda_memory():
    """
    Clear cached memory from CUDA. This cannot release
    memory which is still referencable from the code.
    """
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def ensure_torch_tensor(data):
    """
    Make sure that the input is a pytorch tensor. If it
    already is a pytorch tensor, nothing will be changed.
    If it is a numpy array, it will be converted to a
    pytorch tensor.
    """
    if type(data) is torch.Tensor:
        return data
    else:
        return torch.from_numpy(data)


def intrinsic_dimension(points, *, centre='mean', eps=0, device='cpu'):
    """
    Compute the separability-based intrinsic dimension of a dataset.
    
    `points` should be a dataset with one point in each column.
    """
    device = select_device(device)
    points = ensure_torch_tensor(points)

    n_points = points.shape[1]

    # Work out how to centre the data
    if type(centre) is str:
        if centre == 'mean':
            centre = points.mean(axis=1)
        elif centre == 'zero':
            centre = None
        else:
            raise ValueError(f'Unrecognised centre specification "{centre}"')

    # Centre the data and move it to the correct device
    if centre is None:
        centred = points.to(device)
    else:
        centre = ensure_torch_tensor(centre)
        centred = (points - centre[:, None]).to(device)
        
    # Compute the pairwise projections (x – y, y) as (x, y) - |y|^2
    projections = centred.T @ centred
    norms_sq = torch.diag(projections)
    projections -= norms_sq

    # Clear the GPU memory if necessary
    if centred.is_cuda:
        points = centre = centred = norms_sq = None
        projections = projections.cpu()
        clear_cuda_memory()

    # Compute the probability of separating pairs of points
    # Note: the CUDA implementation of count_nonzero seems to
    #       use too much memory to be practical
    p = (torch.count_nonzero(projections >= 0) - n_points) / (n_points * (n_points - 1))

    # Convert the probability into a dimensionality
    dim = -1 - torch.log2(p + eps)
    if type(dim) is torch.Tensor:
        dim = dim.item()
    return dim


def relative_intrinsic_dimension(target, nontarget, *, centre='nontarget_mean', eps=0, device='cpu'):
    """
    Compute the separability-based relative intrinsic dimension of a target 
    dataset with respect to a background universe of data.
    
    `target` and `nontarget` should be datasets with one point in each column.

    Values for centre:
    'combined_mean': data is centred on the combined mean of all data points
    'target_mean': data is centred on the mean of the target dataset
    'nontarget_mean': data is centred on the mean of the nontarget dataset
    'zero': no centering is applied to the data
    """
    device = select_device(device)
    target = ensure_torch_tensor(target)
    nontarget = ensure_torch_tensor(nontarget)

    n_target_points, n_nontarget_points = target.shape[1], nontarget.shape[1]

    # Work out how to centre the data
    if type(centre) is str:
        if centre == 'combined_mean':
            centre = (target.sum(axis=1) + nontarget.sum(axis=1)) / (n_target_points + n_nontarget_points)
        elif centre == 'target_mean':
            centre = target.mean(axis=1)
        elif centre == 'nontarget_mean':
            centre = nontarget.mean(axis=1)
        elif centre == 'zero':
            centre = None
        else:
            raise ValueError(f'Unrecognised centre specification "{centre}"')

    # Centre the data and move it to the correct device
    target_c = target
    nontarget_c = nontarget
    if centre is None:
        target_c = target.to(device)
        nontarget_c = nontarget.to(device)
    else:
        centre = ensure_torch_tensor(centre)
        target_c = (target - centre[:, None]).to(device)
        nontarget_c = (nontarget - centre[:, None]).to(device)
    
    # Compute the pairwise projections (x – y, y) as (x, y) - |y|^2
    projections = nontarget_c.T @ target_c
    target_norms_sq = (target_c**2).sum(axis=0)
    projections -= target_norms_sq

    # Clear the GPU memory if necessary
    if target_c.is_cuda:
        target_c = nontarget_c = centre = target_norms_sq = None
        projections = projections.cpu()
        clear_cuda_memory()

    # Compute the probability of separating pairs of points
    # Note: the CUDA implementation of count_nonzero seems to
    #       use too much memory to be practical
    p = torch.count_nonzero(projections >= 0) / (n_target_points * n_nontarget_points)

    # Convert the probability into a dimensionality
    dim = -1 - torch.log2(p + eps)
    if type(dim) is torch.Tensor:
        dim = dim.item()
    return dim


def principal_components(dataset, *, device='cpu'):
    """
    Compute the principal components of a dataset 
    (assumed to be a matrix with one data point per column).

    Returns the eigenvalues and eigenvectors of the
    data covariance matrix, both sorted by descending eigenvalue.
    """
    dataset = ensure_torch_tensor(dataset)
    data = dataset.to(select_device(device))

    cov = torch.cov(data)
    evals, evecs = torch.linalg.eigh(cov)
    evals, perm = torch.sort(evals, descending=True)
    evecs = evecs[:, perm]
    evals[evals < 0] = 0  # Remove any (incorrect) negative eigenvalues

    # Clean up the GPU memory if required
    if data.is_cuda:
        data = cov = perm = None
        evals, evecs = evals.cpu(), evecs.cpu()
        clear_cuda_memory()

    return evals, evecs


def pca_dimension(eigenvalues, threshold, threshold_type='relative'):
    """
    Compute the PCA dimensionality of a dataset from the
    eigenvalues of its covariance matrix, using one of
    several different algorithms.

    `threshold_type`:
    `'explained_variance'` (default): the dimensionality is
        computed as the number n such that the sum of the 
        first n eigenvalues is at least `threshold` times the 
        total sum of the eigenvalues.
    `'relative'`: the dimensionality is computed as the number
        of eigenvalues which are at least `threshold` times
        the largest eigenvalue.
    `'absolute'`: the dimensionality is computed as the number
        of eigenvalues which are at least `threshold` in size.
    """
    if threshold_type == 'explained_variance':
        # Then the decision needs to be based on finding
        # the number of eigenvalues such that their sum
        # is greater than the given fraction of the total 
        # sum of the eigenvalues
        eigenvalues = np.cumsum(eigenvalues)
        dim = np.searchsorted(eigenvalues, threshold * eigenvalues[-1]) + 1
    else:
        if threshold_type == 'relative':
            # Scale the threshold to be a proportion of 
            # the largest eigenvalue
            threshold = threshold * eigenvalues.max()
        elif threshold_type == 'absolute':
            # Then we don't need to modify the threshold
            pass
        else:
            raise ValueError(f'Unrecognised value "{threshold_type}" specified for "threshold_type"')
        dim = np.count_nonzero(eigenvalues > threshold)

    # Unwrap the pytorch tensor if needed
    if type(dim) is torch.Tensor:
        return dim.item()
    return dim


def principal_components_projector(eigenvectors, subspace_dimension, preserve_dims=False):
    """
    Project a dataset onto a set of principal component eigenvectors.
    
    `eigenvectors` should be a dataset with one point in each column.
    """
    projector = eigenvectors[:, :subspace_dimension].T
    if preserve_dims:
        projector = projector.T @ projector
    return projector


def whitening_matrix(pca_evals, pca_evecs, *, output_dim=None, eps=1e-12):
    """
    Construct the projection matrix which whitens the
    data, i.e. such that the mapped data has an identity
    covariance matrix.
    
    `pca_evecs` should be a dataset with one eigenvector in each column.
    """
    if output_dim is None:
        output_dim = pca_evals.shape[0]
    pca_evals = pca_evals[:output_dim]
    pca_evecs = pca_evecs[:, :output_dim]
    evals_invsqrt = 1.0 / np.sqrt(pca_evals + eps)
    scaled_evecs = pca_evecs * evals_invsqrt
    return scaled_evecs.T


def whiten_data(data, pca_evals, pca_evecs, *, output_dim=None, eps=1e-12):
    """
    Whiten a dataset (i.e. map it so that its mean is 
    zero and covariance matrix is the identity matrix) 
    using pre-computed principal components.
    
    `data` and `pca_evecs` should be datasets with one point in each column.
    """
    data = data - data.mean(axis=1)[:, None]
    whitener = whitening_matrix(pca_evals, pca_evecs, output_dim=output_dim, eps=eps)
    return whitener @ data


def sparsify_data(data, quantile, *, n_chunks=10, device='cpu'):
    """
    Implementation of the sparsification operation for a 
    specified quantile level. 
    
    `data` should be a dataset with one point in each column.

    The parameter `n_chunks` is used to break the quantile
    operation up to fit into GPU memory. Since the quantile 
    is computed independently for each row of the matrix, 
    this does not affect the output of the computation. If
    struggling with GPU memory size, try increasing the
    `n_chunks`.
    """
    device = select_device(device)
    data = ensure_torch_tensor(data).to(device)
    abs_data = torch.abs(data)
    n_chunks = min(n_chunks, data.shape[0])

    # Break the data into chunks to make the quantile
    # computation easier on the GPU memory.
    rows_per_part = data.shape[0] // n_chunks
    thresholds = []
    for i in range(n_chunks):
        print(rows_per_part, rows_per_part*i, rows_per_part*(i+1))
        chunk_abs = abs_data[rows_per_part*i:rows_per_part*(i+1), :]
        thresholds.append(torch.quantile(chunk_abs, quantile, axis=1))
    thresholds = torch.hstack(thresholds)

    # Sparsify the data using the computed thresholds
    sparsified = torch.sign(data) * torch.relu(abs_data - thresholds[:, None])

    # Clean up the CUDA memory if necessary
    if abs_data.is_cuda:
        sparsified = sparsified.cpu()
        data = abs_data = chunk_abs = thresholds = None
        clear_cuda_memory()

    return sparsified
