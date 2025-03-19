import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
image_path = "tree.png"
image = cv2.imread(image_path)

# 이미지 크기 가져오기
rows, cols = image.shape[:2]

# 회전 변환 행렬 생성 (중심: (cols/2, rows/2), 회전 각도: 45도, 배율: 1.5배)
rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1.5)

# 회전 및 확대된 이미지 크기 설정
new_cols, new_rows = int(cols * 1.5), int(rows * 1.5)

# Affine 변환 적용 (선형 보간 적용)
rotated_scaled_image = cv2.warpAffine(image, rotation_matrix, (new_cols, new_rows), flags=cv2.INTER_LINEAR)

# 원본과 변환된 이미지 비교 출력
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
