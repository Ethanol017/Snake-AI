import numpy as np
import matplotlib.pyplot as plt

state = np.load("./max_size_state.npy")
palette = np.asarray(
    [
        [22, 22, 22],
        [45, 160, 65],
        [230, 70, 70],
        [30, 130, 210],
    ],
    dtype=np.uint8,
)

if state.ndim == 2:
    plt.imshow(palette[state])
else:
    plt.imshow(state)
plt.show()
