# L06 Local Feature

## 01. SIFT를 이용한 특징점 검출 및 시각화
cv.SIFT_create()를 사용하여 SIFT 객체를 생성합니다.

detectAndCompute()를 사용하여 특징점을 검출합니다.

cv.drawKeypoints()를 사용하여 특징점을 이미지에 시각화합니다.

matplotlib을 이용하여 원본 이미지와 특징점이 시각화된 이미지를 나란히 출력하세요.

#### 요구사항 1: cv.imread()를 사용하여 이미지를 불러옵니다.
    img = cv.imread('mot_color70.jpg')
#### 요구사항 2: cv.SIFT_create()를 사용하여 SIFT 객체를 생성합니다.
    sift = cv.SIFT_create()
#### 요구사항 3: detectAndCompute()를 사용하여 특징점을 검출합니다.
    kp, des = sift.detectAndCompute(img, None)
#### 요구사항 4: cv.drawKeypoints()를 사용하여 특징점을 이미지에 시각화합니다.
    img_keypoints = cv.drawKeypoints(img, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#### 요구사항 5: matplotlib을 이용하여 원본 이미지와 특징점이 시각화된 이미지를 나란히 출력하세요.
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(img_keypoints, cv.COLOR_BGR2RGB))
    plt.title('SIFT Keypoints')
    plt.axis('off')
    
    plt.show()
#### 결과화면
<img width="709" alt="image" src="https://github.com/user-attachments/assets/1f98387c-af99-40e1-a277-bd08c9c858b4" />

이미지에서 보이는 원의 크기는 SIFT에서 해당 키포인트가 감지된 스케일(Scale)을 의미함

큰 원은 큰 영역에서 감지된 키포인트, 작은 원은 작은 디테일에서 감지된 키포인트라는 뜻

---

## 02. SIFT를 이용한 두 영상 간 특징점 매칭
두 개의 이미지를 입력받아 SIFT 특징점 기반으로 매칭을 수행하고 결과를 시각화하세요.

#### 요구사항 1: cv.imread()를 사용하여두개의이미지를불러옵니다.
    img1 = cv.imread('mot_color70.jpg')
    img2 = cv.imread('mot_color83.jpg')
#### 요구사항 2: cv.SIFT_create()를 사용하여 특징점을추출합니다.
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
#### 요구사항 3: cv.BFMatcher() 또는 cv.FlannBasedMatcher()를 사용하여 두 영상 간 특징점을 매칭합니다.
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance) # 매칭 결과 거리순 정렬
#### 요구사항 4: cv.drawMatches()를 사용하여 매칭결과를시각화합니다.
    matched_img = cv.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#### 요구사항 5: matplotlib을 이용하여 매칭 결과를 출력하세요
    plt.figure(figsize=(12, 6))
    plt.imshow(cv.cvtColor(matched_img, cv.COLOR_BGR2RGB))
    plt.title('SIFT Feature Matching')
    plt.axis('off')
    plt.show()
#### 결과화면 
<img width="709" alt="image" src="https://github.com/user-attachments/assets/7865ffb6-9578-4b74-8b3e-24208bad8942" />

---

## 03. 호모그래피를 이용한 이미지 정합
SIFT 특징점을 사용하여 두 이미지 간 대응점을 찾고, 이를 바탕으로 호모그래피를 계산하여 하나의 이미지 위에 정렬하세요

#### 요구사항 1: cv.imread()를 사용하여 두 개의 이미지를 불러옵니다.
    img1 = cv.imread('img1.jpg')
    img2 = cv.imread('img2.jpg')
#### 요구사항 2: cv.SIFT_create()를 사용하여 특징점을 검출합니다.
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
#### 요구사항 3: cv.BFMatcher()를 사용하여 특징점을 매칭합니다.
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    
    matched_img = cv.drawMatches(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#### 요구사항 4: cv.findHomography()를 사용하여 호모그래피 행렬을 계산합니다.
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

#### 요구사항 5: cv.warpPerspective()를 사용하여 한 이미지를 변환하여 다른 이미지와 정렬합니다.
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    corners1 = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
    corners2 = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
    transformed_corners1 = cv.perspectiveTransform(corners1, M)
    all_corners = np.concatenate((transformed_corners1, corners2), axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1,0,translation_dist[0]],[0,1,translation_dist[1]],[0,0,1]], dtype=np.float32)
    stitched_width = x_max - x_min
    stitched_height = y_max - y_min
    
    result = cv.warpPerspective(img1, H_translation.dot(M), (stitched_width, stitched_height))
    result[translation_dist[1]:translation_dist[1]+h2, translation_dist[0]:translation_dist[0]+w2] = img2
#### 요구사항 6: 변환된 이미지를 원본 이미지와 비교하여 출력하세요.
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1)
    plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    plt.title("Target Image")
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
    plt.title("Aligned Image")
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(cv.cvtColor(matched_img, cv.COLOR_BGR2RGB))
    plt.title('Matched Points')
    plt.axis('off')
    
    plt.show()
#### 결과화면
<img width="701" alt="image" src="https://github.com/user-attachments/assets/28d82d9d-d80c-4cd7-a87f-e0a253c6ab12" />
