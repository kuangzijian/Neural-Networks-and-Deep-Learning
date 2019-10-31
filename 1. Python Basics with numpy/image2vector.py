# GRADED FUNCTION: image2vector
def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    length = image.shape[0]
    height = image.shape[1]
    depth = image.shape[2]
    v = image.reshape(length * height * depth, 1)
    ### END CODE HERE ###

    return v