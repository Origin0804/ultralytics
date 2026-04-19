import cv2
import numpy as np

img = np.random.rand(100, 100, 6).astype(np.float32)
try:
    res = cv2.resize(img, (50, 50))
    print("cv2.resize success! shape:", res.shape)
except Exception as e:
    print("cv2.resize failed:", e)

try:
    M = np.float32([[1, 0, 10], [0, 1, 10]])
    res = cv2.warpAffine(img, M, (100, 100))
    print("cv2.warpAffine success! shape:", res.shape)
except Exception as e:
    print("cv2.warpAffine failed:", e)
