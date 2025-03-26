# 캐니 에지 및 허프 변환을 이용한 직선 검출
# 주어진 이미지를 다음과 같이 처리하세요
# 캐니 에지 검출을 사용하여 에지 맵을 생성합니다
# 허프 변환을 사용하여 이미지에서 직선을 검출합니다
# 검출된 직선을 원본 이미지에 빨간색으로 표시합니다

import cv2 as cv
import matplotlib.pyplot as plt

# 요구사항1: cv.imread()를 사용하여 이미지를 불러옵니다
img = cv.imread('soccer.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

canny = cv.Canny(gray, 100, 200)

# 요구사항2: cv.HoughLinesP()를 사용하여 직선을 검출합니다
# 힌트: cv.HoughLinesP()에서 rho, theta, threshold, minLineLength, maxLineGap 값을 조정하여 직선 검출 성능을 개선할 수 있습니다.
lines = cv.HoughLinesP(canny, rho=1, theta=3.14/180, threshold=100, minLineLength=50, maxLineGap=10)

# 요구사항3: cv.line()을 사용하여 검출된 직선을 원본 이미지에 표시합니다
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 요구사항4: matplotlib를 사용하여 원본 이미지와 검출된 직선을 나란히 시각화합니다
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(canny, cmap='gray')
plt.title('Canny Edge')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Detected Lines')
plt.axis('off')

plt.tight_layout()
plt.show()