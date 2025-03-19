# L04 Vision Processing Basic

## 01. 이진화 및 히스토그램 구하기
이미지를 그레이스케일로 변환
특정 임계값을 설정하여 이진화
이진화된 이미지의 히스토그램을 계산하고 시각화
#### 요구사항 1: cv.imread()를 사용하여 이미지 로드
    img = cv.imread('soccer.jpg')
#### 요구사항 2: cv.cvtColor()를 사용해 이미지를 그레이스케일로 변환
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#### 3: cv.threshold()를 사용해 이진화, 임계값: 127
    retval, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
#### 요구사항 4: cv.calcHist()를 사용해 히스토그램을 계산하고, matplotlib으로 시각화
    hist = cv.calcHist([gray], [0], None, [256], [0, 256])
    plt.plot(hist, color='r', linewidth=1)
    plt.show()
#### 결과화면
<img width="447" alt="image" src="https://github.com/user-attachments/assets/828c0c06-b0d5-46a2-804f-f159fb466816" />

---

## 02. 모폴로지 연산 적용하기
주어진 이진화된 이미지에 대해 다음 모폴로지 연산을 적용
팽창(Dilation), 침식(Erosion), 열림(Open), 닫힘(Close)
#### 요구사항 1: cv.getStructuringElement()를 사용하여 사각형 커널(5x5)을 만드세요
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#### 요구사항 2: cv.morphologyEx()를 사용하여 각 모폴로지 연산을 적용하세요
    dilation = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)
    erosion = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
#### 요구사항 3: 원본 이미지와 모폴로지 연산 결과를 한 화면에 출력하세요
    result = np.hstack((image, dilation, erosion, opening, closing))
    plt.imshow(result)
    plt.show()
#### 결과화면
<img width="444" alt="image" src="https://github.com/user-attachments/assets/a6defb45-4c3e-4929-90fd-3b115d21d8af" />

---

## 03. 기하 연산 및선형 보간 적용하기
주어진 이미지를 45도 회전시킵니다
회전된 이미지를 1.5배 확대합니다
회전 및 확대된 이미지에 선형 보간(Bilinear Interpolation)을 적용하여 부드럽게 표현하세요
#### 요구사항 1: getRotationMatrix2D()를 사용하여 회전 변환 행렬을 생성하세요
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1.5)
#### 요구사항 2: warpAffine()를 사용하여 이미지를 회전 및 확대하세요
#### 요구사항 3: INTER_LINEAR을 사용하여 선형 보간을 적용하세요
    rotated_scaled_image = cv2.warpAffine(image, rotation_matrix, (new_cols, new_rows), flags=cv2.INTER_LINEAR)
#### 요구사항 4: 원본 이미지와 회전 및 확대된 이미지를 한 화면에 비교하세요
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
#### 결과화면
<img width="607" alt="image" src="https://github.com/user-attachments/assets/eafcfbce-9836-4d83-9d3f-67b486bd06f4" />
