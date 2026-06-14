import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

x_train, t_train, x_test, t_test = load_mnist(normalize=False, flatten=True)

# 이미지 1장 표시
img = x_train[0]
label = t_train[0]

print("이미지 shape:", img.shape)   # (784,)
print("정답 레이블:", label)

img = img.reshape(28, 28)           # 784 → 28x28 로 복원

plt.imshow(img, cmap='gray')
plt.title(f'label: {label}')
plt.axis('off')
plt.show()
