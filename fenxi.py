import numpy as np
import matplotlib.pyplot as plt
arr = np.loadtxt("cmake-build-debug/conv_output.txt")
arr = np.maximum(arr, 0)
plt.imshow(arr, cmap='gray')
plt.title("Conv Output (with ReLU)")
plt.colorbar()
plt.show()