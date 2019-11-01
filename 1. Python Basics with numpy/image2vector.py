# GRADED FUNCTION: image2vector
def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """

    length = image.shape[0]
    height = image.shape[1]
    depth = image.shape[2]
    v = image.reshape(length * height * depth, 1)

    return v