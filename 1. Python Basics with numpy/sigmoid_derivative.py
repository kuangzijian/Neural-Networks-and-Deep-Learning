# GRADED FUNCTION: sigmoid_derivative
import numpy as np


def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.

    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """

    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    return ds