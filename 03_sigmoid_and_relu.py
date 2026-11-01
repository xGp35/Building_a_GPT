import numpy as np
from numpy.typing import NDArray

class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        ans = 1 / (1 + np.e**(-z))
        return np.round(ans, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, ) element-wise
        ans = np.maximum(0,z) # Numpy version of max in python.
        return np.round(ans,5) 

