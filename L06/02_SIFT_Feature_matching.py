# SIFT를 이용한 두 영상 간 특징점 매칭
# 두 개의 이미지를 입력받아 SIFT 특징점 기반으로 매칭을 수행하고 결과를 시각화하세요

import cv2 as cv
import matplotlib.pyplot as plt

# cv.imread()를 사용하여두개의이미지를불러옵니다.
img1 = cv.imread('mot_color70.jpg')
img2 = cv.imread('mot_color83.jpg')

# cv.SIFT_create()를 사용하여 특징점을추출합니다.
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# cv.BFMatcher() 또는 cv.FlannBasedMatcher()를 사용하여 두 영상 간 특징점을 매칭합니다.
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.knnMatch(des1, des2, k=2)
matches = sorted(matches, key=lambda x: x.distance)

# cv.drawMatches()를 사용하여 매칭결과를시각화합니다.
matched_img = cv.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# matplotlib을 이용하여 매칭 결과를 출력하세요
plt.figure(figsize=(12, 6))
plt.imshow(cv.cvtColor(matched_img, cv.COLOR_BGR2RGB))
plt.title('SIFT Feature Matching')
plt.axis('off')
plt.show()
