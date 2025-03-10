# L01. OpenCV - Homework

  ## 01. 이미지 불러오기 및 그레이스케일 변환
  OpenCV를 사용하여 이미지를 불러오고 화면에 출력
  
  원본 이미지와 그레이스케일로 변환된 이미지를 나란히 표시
  #### 요구사항 1: cv.imread()를 사용하여 이미지 로드
    img = cv.imread('soccer.jpg')
  #### 요구사항 2: cv.cvtColor() 함수를 사용해 이미지를 그레이스케일로 변환
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  #### 요구사항 3: np.hstack() 함수를 이용해 원본 이미지와 그레이스케일 이미지를 가로로 연결하여 출력
    imgs = np.hstack((img, cv.cvtColor(gray, cv.COLOR_GRAY2BGR)))
  #### 요구사항 4: cv.imshow()와 cv.waitKey()를 사용해 결과를 화면에 표시하고, 아무 키나 누르면 창이 닫히도록 할 것
    cv.imshow('imgs', imgs)
    cv.waitKey(0)
    cv.destroyAllWindows()
---
  ## 02. 웹캠 영상에서 에지 검출
  웹캠을 사용하여 실시간 비디오 스트림을 가져온다
  
  각 프레임에서 Canny Edge Detection을 적용하여 에지를 검출하고 원본 영상과 함께 출력
  #### 요구사항 1: cv.VideoCapture()를 사용해 웹캠 영상을 로드
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit('카메라 연결 실패')
  #### 요구사항 2: 각 프레임을 그레이스케일로 변환한 후, cv.Canny() 함수를 사용해 에지 검출 수행
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(gray, 50, 150)
  #### 요구사항 3: 원본 영상과 에지 검출 영상 가로로 연결하여 화면에 출력
    canny = canny[:, :, np.newaxis]
    imgs = np.hstack((frame, np.repeat(canny, 3, axis=2)))
    cv.imshow('imgs', imgs)
  #### 요구사항 4: q키를 누르면 영상 창이 종료
    key = cv.waitKey(1)
    if key == ord('q'):
        break 

  
---
  ## 03. 마우스로 영역 선택 및 ROI(관심영역) 추출
  이미지를 불러오고 사용자가 마우스로 클릭하고 드래그하여 관심영역(ROI)을 선택

  선택한 영역만 따로 저장하거나 표시
  #### 요구사항 1: 이미지를 불러오고 화면에 출력
    img = cv.imread('soccer.jpg')
    if img is None:
        sys.exit('파일이 존재하지 않습니다.')
  #### 요구사항 2: cv.setMouseCallback()을 사용하여 마우스 이벤트를 처리
    cv.setMouseCallback('Drawing', draw)
  #### 요구사항 3: 사용자가 클릭한 시작점에서 드래그하여 사각형을 그리며 영역을 선택
    
  #### 요구사항 4: 마우스를 놓으면 해당 영역을 잘라내서 별도의 창에 출력
  #### 요구사항 5: r키를 누르면 영역 선택을 리셋하고 처음부터 다시 선택
  #### 요구사항 6: s키를 누르면 선택한 영역을 이미지 파일로 저장
  
