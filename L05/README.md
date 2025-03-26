# L05 Edge and Region

## 01. 소벨 에지 검출 및 결과 시각화
이미지를 그레이스케일로 변환합니다

소벨(Sobel) 필터를 사용하여 X축과 Y축의 방향의 에지를 검출합니다

검출된 에지 강도(edge strength) 이미지를 시각화합니다
#### 요구사항1: cv.imread()를 사용하여 이미지를 불러옵니다
    img = cv.imread('soccer.jpg')
#### 요구사항2: cv.cvtColor()를 사용하여 그레이스케일로 변환합니다
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#### 요구사항3: cv.Sobel()을 사용하여 X축(cv.CV_64F, 1, 0)과 Y축(cv.CV_64F, 0, 1) 방향의 에지를 검출합니다
    grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    grad_Y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
#### 요구사항4: cv.magnitude()를 사용하여 에지강도를 계산합니다
    edge_strength = cv.magnitude(grad_x, grad_Y)
    edge_display = cv.convertScaleAbs(edge_strength)
#### 요구사항5: matplotlib를 사용하여 원본 이미지와 에지 강도 이미지를 나란히 시각화합니다
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
#### 결과화면
<img width="736" alt="image" src="https://github.com/user-attachments/assets/b94901f7-a4f2-4463-80b4-7303f6003545" />

---
## 02. 캐니 에지 및 허프 변환을 이용한 직선 검출
주어진 이미지를 다음과 같이 처리하세요

캐니 에지 검출을 사용하여 에지 맵을 생성합니다

허프 변환을 사용하여 이미지에서 직선을 검출합니다

검출된 직선을 원본 이미지에 빨간색으로 표시합니다
#### 요구사항1: cv.imread()를 사용하여 이미지를 불러옵니다
    img = cv.imread('soccer.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(gray, 100, 200)
#### 요구사항2: cv.HoughLinesP()를 사용하여 직선을 검출합니다
    lines = cv.HoughLinesP(canny, rho=1, theta=3.14/180, threshold=100, minLineLength=50, maxLineGap=10)
#### 요구사항3: cv.line()을 사용하여 검출된 직선을 원본 이미지에 표시합니다
    if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#### 요구사항4: matplotlib를 사용하여 원본 이미지와 검출된 직선을 나란히 시각화합니다
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
#### 결과화면
<img width="737" alt="image" src="https://github.com/user-attachments/assets/b86c2849-8679-4e0b-8e6e-f418485da7b2" />

---
## 03. GrabCut을 이용한 대화식 영역 분할 및 객체 추출
주어진 이미지를 다음과 같이 처리하세요

사용자가 지정한 사각형 영역을 바탕으로 GrabCut 알고리즘을 사용하여 객체를 추출합니다

객체 추출 결과를 마스크 형태로 시각화합니다

원본 이미지에서 배경을 제거하고 객체만 남은 이미지를 출력합니다
#### 요구사항1: cv.grabCut()를 사용하여대화식분할을수행합니다
#### 요구사항2: 초기 사각형 영역은 (x, y, width, height) 형식으로 설정하세요
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (500, 550, 300, 300)

    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
#### 요구사항3: 마스크를 사용하여 원본 이미지에서 배경을 제거합니다
    mask2 = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 1, 0).astype('uint8')
    result = img * mask2[:, :, np.newaxis]
#### 요구사항4: matplotlib를 사용하여 원본이미지, 마스크이미지, 배경 제거 이미지 세 개를 나란히 시각화합니다
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
    plt.title('Foreground Only')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
#### 결과화면
<img width="1110" alt="image" src="https://github.com/user-attachments/assets/ac0b9cad-33f8-4f39-8227-fa0f6c0c6cfc" />
