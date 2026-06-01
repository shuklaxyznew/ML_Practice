import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, gaussian_filter, sobel

im = np.zeros((300, 300))
im[64:-64, 64:-64] = 1

im = rotate(im, 30, mode="constant")
im = gaussian_filter(im, sigma=7)

plt.imshow(im, cmap="gray")
plt.axis("off")
plt.title("Original Synthetic Image")
plt.show()

dx = sobel(im, axis=0, mode="constant")
dy = sobel(im, axis=1, mode="constant")
sobel_edges = np.hypot(dx, dy)

plt.imshow(sobel_edges, cmap="gray")
plt.axis("off")
plt.title("Sobel Edge Detection")
plt.show()
