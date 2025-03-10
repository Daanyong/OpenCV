# 웹캠 영상에서 에지 검출
# 웹캠을 사용하여 실시간 비디오 스트림을 가져온다
# 각 프레임에서 Canny Edge Detection을 적용하여 에지를 검출하고 원본 영상과 함께 출력

import cv2 as cv
import sys
import numpy as np

# 요구사항 1: cv.VideoCapture()를 사용해 웹캠 영상을 로드
cap = cv.VideoCapture(0, cv.CAP_DSHOW) # 카메라와 연결 시도

if not cap.isOpened():
    sys.exit('카메라 연결 실패')

while True:
    ret, frame = cap.read() # 비디오를 구성하는 프레임 획득

    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    cv.imshow('Video display', frame)

    # 요구사항 2: 각 프레임을 그레이스케일로 변환한 후, cv.Canny() 함수를 사용해 에지 검출 수행
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(gray, 50, 150)

    # 요구사항 3: 원본 영상과 에지 검출 영상 가로로 연결하여 화면에 출력
    canny = canny[:, :, np.newaxis]
    imgs = np.hstack((frame, np.repeat(canny, 3, axis=2)))
    cv.imshow('imgs', imgs)
    
    # 요구사항 4: q키를 누르면 영상 창이 종료
    key = cv.waitKey(1) # 1밀리초 동안 키보드 입력 기다림
    if key == ord('q'): # 'q' 키가 들어오면 루프를 빠져나감
        break

cap.release() # 카메라와 연결을 끊음
cv.destroyAllWindows()