import numpy as np
from numpy.linalg import norm

def align(a, b):
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-8)

def divergence(a, b):
    return 0.38 - align(a, b)

def tension(a, b):
    return abs(divergence(a, b))

def basin_direction(states, window=7):
    return states[-window:].mean(axis=0)
