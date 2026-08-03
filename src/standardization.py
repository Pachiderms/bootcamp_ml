import numpy as np

def zscore(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    x' as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
    This function shouldn't raise any Exception.
    """
    
    if not isinstance(x, np.ndarray) or x.size == 0:
        print(f"zscore type/empty err: {type(x)=} {x.size=}")
        return None
    
    m = x.shape[0]
    mean = 0
    for i in range(m):
        mean += x[i]
        
    mean = mean / m if m > 0 else 0
        
    std = 0
    for i in range(m):
        std += (x[i] - mean) ** 2
    std /= m if m > 0 else 0
    std = np.sqrt(std)
    
    x_normalized = (x - mean) / np.where(std == 0, 1, std)
    
    return x_normalized

def minmax(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    x' as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
    This function shouldn't raise any Exception.
    """
    
    if not isinstance(x, np.ndarray) or x.size == 0:
        print(f"minmax type/empty err: {type(x)=} {x.size=}")
        return None
    
    min_val = np.min(x)
    max_val = np.max(x)
    
    if max_val == min_val:
        return np.zeros(x.shape)
    
    x_normalized = (x - min_val) / (max_val - min_val)
    
    return x_normalized