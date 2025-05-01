

def json_to_ndarray(data):
    """
    Convert JSON data to a NumPy array.
    """
    import numpy as np
    return np.array([[data['x1'], data['x2'], data['x3'], data['x4']]])

