import numpy as np
from scipy.stats import norm
import scipy.io
mat = scipy.io.loadmat('gray.mat')
gray = mat['gray']
gray = gray.tolist()

class1 = [gray[i] for i in range(len(gray)) if gray[i][1] == 1]
class2 = [gray[i] for i in range(len(gray)) if gray[i][1] == -1]

# compute the prior probability
prior1 = len(class1) / (len(class1) + len(class2))
prior2 = len(class2) / (len(class1) + len(class2))

# compute the mean and variance for 2 classes
class1 = np.asarray(class1)
class2 = np.asarray(class2)
mu1 = np.mean(np.delete(class1, 1, axis=1))
mu2 = np.mean(np.delete(class2, 1, axis=1))
var1 = np.var(np.delete(class1, 1, axis=1))
var2 = np.var(np.delete(class2, 1, axis=1))


import cv2
img = cv2.imread('309.bmp')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mat2 = scipy.io.loadmat('Mask.mat')
mask = mat2['Mask']
gray_img = gray_img * mask / 255
cv2.imshow('gray image after being masked', gray_img)
# cv2.waitKey(0)
height = gray_img.shape[0]
width = gray_img.shape[1]
# print([prior1, prior2])
# print([[mu1, var1], [mu2, var2]])
for row in range(height):     # 遍历每一行
    for col in range(width):  # 遍历每一列
        if gray_img[row][col] == 0: continue
        if prior1 * norm.pdf(gray_img[row][col], mu1, var1) > prior2 * norm.pdf(gray_img[row][col], mu2, var2):
            gray_img[row][col] = 255
        else:
            gray_img[row][col] = 0

cv2.imshow('segmented gray image', gray_img)
cv2.waitKey(0)