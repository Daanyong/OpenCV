# 이진화 및 히스토그램 구하기
# 주어진 이미지를 불러와서 다음을 수행
# 이미지를 그레이스케일로 변환
# 특정 임계값을 설정하여 이진화
# 이진화된 이미지의 히스토그램을 계산하고 시각화

import cv2 as cv
from matplotlib import pyplot as plt

# 요구사항 1: cv.imread()를 사용하여 이미지 로드
img = cv.imread('soccer.jpg')

# 요구사항 2: cv.cvtColor()를 사용해 이미지를 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 요구사항 3: cv.threshold()를 사용해 이진화, 임계값: 127
retval, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

# 요구사항 4: cv.calcHist()를 사용해 히스토그램을 계산하고, matplotlib으로 시각화
hist_gray = cv.calcHist([gray], [0], None, [256], [0, 256])
hist_binary = cv.calcHist([binary], [0], None, [256], [0, 256])

plt.figure(figsize=(12, 5))

# 원본 그레이스케일 이미지의 히스토그램
plt.subplot(1, 2, 1)
plt.plot(hist_gray, color='b', linewidth=1)
plt.title('Histogram of Grayscale Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

# 이진화된 이미지의 히스토그램 (막대그래프로 표현)
plt.subplot(1, 2, 2)
plt.plot(hist_binary, color='b', linewidth=1)
plt.title('Histogram of Binary Image')
plt.xlabel('Pixel Value (0 = Black, 255 = White)')
plt.ylabel('Frequency')

plt.show()
