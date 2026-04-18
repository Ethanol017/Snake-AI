import numpy as np
import matplotlib.pyplot as plt

state = np.load('./max_size_state.npy')
# print(state)
plt.imshow(state)
plt.show()