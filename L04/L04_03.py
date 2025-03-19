# 기하 연산 및 선형 보간 적용하기
# 주어진 이미지를 다음과 같이 변환하세요
# 이미지를 45도 회전시킵니다
# 회전된 이미지를 1.5배 확대합니다
# 회전 및 확대된 이미지에 선형 보간(Bilinear Interpolation)을 적용하여 부드럽게 표현하세요

import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "tree.png"
image = cv2.imread(image_path)

rows, cols = image.shape[:2]

# 요구사항 1: getRotationMatrix2D()를 사용하여 회전 변환 행렬을 생성하세요
rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1.5)

new_cols, new_rows = int(cols * 1.5), int(rows * 1.5)

# 요구사항 2: warpAffine()를 사용하여 이미지를 회전 및 확대하세요
# 요구사항 3: INTER_LINEAR을 사용하여 선형 보간을 적용하세요
rotated_scaled_image = cv2.warpAffine(image, rotation_matrix, (new_cols, new_rows), flags=cv2.INTER_LINEAR)

# 원본 이미지와 회전 및 확대된 이미지를 한 화면에 비교하세요
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(rotated_scaled_image, cv2.COLOR_BGR2RGB))
plt.title("Rotated & Scaled Image (45° & 1.5x)")
plt.axis("off")

plt.show()
