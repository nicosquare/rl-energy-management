def normalize(value, min_value, max_value):
    """
        Normalize a value given its max and min value from the distribution it came from.
    Parameters
    ----------
    value: float
        Value to be normalized.
    min_value:
        Maximum value in the distribution.
    max_value:
        Minimum value in the distribution.

    Returns
    -------
        Normalized value
    """
    return (value - min_value) / (max_value - min_value)