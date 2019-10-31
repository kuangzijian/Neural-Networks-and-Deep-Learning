import math
import numpy as np
from sigmoid_derivative import sigmoid_derivative
from sigmoid import sigmoid
from basic_sigmoid import basic_sigmoid

print("basic_sigmoid(3): " + str(basic_sigmoid(3)))
print("Expected Output: 0.9525741268224334")

print("========================================")
x = np.array([1, 2, 3])
print("sigmoid(3): " + str(sigmoid(x)))
print("Expected Output: [ 0.73105858, 0.88079708, 0.95257413]")

print("========================================")
x = np.array([1, 2, 3])
print("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))
print("Expected Output: [ 0.19661193 0.10499359 0.04517666]")



