import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('./image.png')

plt.imshow(img)
plt.show()