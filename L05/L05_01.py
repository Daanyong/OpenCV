# 주어진 이미지를 불러와서 다음을 수행하세요
# 이미지를 그레이스케일로 변환합니다
# 소벨(Sobel) 필터를 사용하여 X축과 Y축의 방향의 에지를 검출합니다
# 검출된 에지 강도(edge strength) 이미지를 시각화합니다

import cv2 as cv
import matplotlib.pyplot as plt

# 요구사항1: cv.imread()를 사용하여 이미지를 불러옵니다
img = cv.imread('soccer.jpg')

# 요구사항2: cv.cvtColor()를 사용하여 그레이스케일로 변환합니다
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 요구사항3: cv.Sobel()을 사용하여 X축(cv.CV_64F, 1, 0)과 Y축(cv.CV_64F, 0, 1) 방향의 에지를 검출합니다
grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
grad_Y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

# 요구사항4: cv.magnitude()를 사용하여 에지강도를 계산합니다
edge_strength = cv.magnitude(grad_x, grad_Y)
edge_display = cv.convertScaleAbs(edge_strength)

# 요구사항5: matplotlib를 사용하여 원본 이미지와 에지 강도 이미지를 나란히 시각화합니다
# 힌트3: plt.imshow()에서 cmap='gray'를 사용하여 흑백으로시각화합니다
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edge_display, cmap='gray')
plt.title('Edge Strength Image')
plt.axis('off')

plt.tight_layout()
plt.show()