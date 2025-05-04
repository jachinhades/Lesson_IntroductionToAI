'''import numpy as np
import matplotlib.pyplot as plt

def load_results(filename):
    images = []
    preds = []
    trues = []
    with open(filename, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            pred, true = int(parts[0]), int(parts[1])
            pixels = np.array(parts[2:]).reshape(28, 28)
            images.append(pixels)
            preds.append(pred)
            trues.append(true)
    return images, preds, trues

images, preds, trues = load_results("cmake-build-debug/results1.txt")

# 显示前10张图像的识别效果
for i in range(100):
    plt.imshow(images[i], cmap='gray')
    plt.title(f"Predicted: {preds[i]}, Label: {trues[i]}")
    plt.axis('off')
    plt.show()
'''
import numpy as np
import matplotlib.pyplot as plt

def load_results(filename):
    images = []
    preds = []
    trues = []
    with open(filename, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            pred, true = int(parts[0]), int(parts[1])
            pixels = np.array(parts[2:]).reshape(28, 28)
            images.append(pixels)
            preds.append(pred)
            trues.append(true)
    return images, preds, trues

images, preds, trues = load_results("cmake-build-debug/results1.txt")

# 创建 2 行 5 列的子图布局
plt.figure(figsize=(25, 20))
for i in range(500):
    plt.subplot(25, 20, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"P:{preds[i]} T:{trues[i]}", fontsize=10)
    plt.axis('off')

plt.suptitle("MNIST Prediction Results (First 100 Images)", fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.85)  # 为标题腾点空间
plt.show()
