from scipy.stats import multivariate_normal
import numpy as np
import scipy.io
mat = scipy.io.loadmat('array_sample.mat')
rgb = mat['array_sample']
rgb = np.delete(rgb, 0, axis=1)
rgb = rgb.tolist()

class1 = [rgb[i] for i in range(len(rgb)) if rgb[i][3] == 1]
class2 = [rgb[i] for i in range(len(rgb)) if rgb[i][3] == -1]

# compute the prior probability
prior1 = len(class1) / (len(class1) + len(class2))
prior2 = len(class2) / (len(class1) + len(class2))

# compute the mean and covariance for 2 classes
class1 = np.asarray(class1)
class2 = np.asarray(class2)
mu1 = np.mean(np.delete(class1, 3, axis=1), axis=0)
mu2 = np.mean(np.delete(class2, 3, axis=1), axis=0)
Sigma1 = np.cov(np.delete(class1, 3, axis=1), rowvar=False)
Sigma2 = np.cov(np.delete(class2, 3, axis=1), rowvar=False)

import cv2
img = cv2.imread('309.bmp')
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mat2 = scipy.io.loadmat('Mask.mat')
mask = mat2['Mask']
img = cv2.bitwise_and(img, img, mask=mask) / 255
cv2.imshow('rgb image after being masked', img)
height = img.shape[0]
width = img.shape[1]
# print([prior1, prior2])
for row in range(height):     # 遍历每一行
    for col in range(width):  # 遍历每一列
        if img[row][col].all() == 0: continue
        if prior1 * multivariate_normal.pdf(img[row][col], mu1, Sigma1) > prior2 * multivariate_normal.pdf(img[row][col], mu2, Sigma2):
            img[row][col] = 255
        else:
            img[row][col] = 0

cv2.imshow('segmented gray image', img)
cv2.waitKey(0)
