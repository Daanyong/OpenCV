# 모폴로지 연산 적용하기
# 주어진 이진화된 이미지에 대해 다음 모폴로지 연산을 적용
# 팽창(Dilation), 침식(Erosion), 열림(Open), 닫힘(Close)

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('JohnHancocksSignature.png', cv2.IMREAD_UNCHANGED)
image = img[img.shape[0]//2:img.shape[0], 0:img.shape[0]//2+1]

# 요구사항 1: cv.getStructuringElement()를 사용하여 사각형 커널(5x5)을 만드세요
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 요구사항 2: cv.morphologyEx()를 사용하여 각 모폴로지 연산을 적용하세요
dilation = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)
erosion = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# 요구사항 3: 원본 이미지와 모폴로지 연산 결과를 한 화면에 출력하세요
result = np.hstack((image, dilation, erosion, opening, closing))

plt.imshow(result)
plt.show()