# 07 Recognition

## 01. 간단한 이미지 분류기 구현
손글씨 숫자 이미지(MNIST 데이터셋)를 이용하여 간단한 이미지 분류기를 구현합니다

#### 요구사항1: MNIST 데이터셋을 로드합니다
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
#### 요구사항2: 데이터를 훈련 세트와 테스트 세트로 분할합니다
    x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
#### 요구사항3: 간단한 신경망 모델을 구축합니다
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(784,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
#### 요구사항4: 모델을 훈련시키고 정확도를 평가합니다
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'테스트 정확도: {acc}')
#### 실행화면
정확도: 0.9717000126838684
![스크린샷 2025-04-09 110847](https://github.com/user-attachments/assets/69515e1f-4a5d-4c60-a717-7a86e99bd171)
<img width="437" alt="image" src="https://github.com/user-attachments/assets/658668c8-d4c9-4853-b9f1-3e214bde8d18" />


## 02. CIFAR-10 데이터셋을 활용한 CNN 모델 구축
CIFAR-10 데이터셋을 활용하여 합성곱 신경망(CNN)을 구축하고, 이미지 분류를 수행합니다

#### 요구사항1: CIFAR-10 데이터셋을 로드합니다
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#### 요구사항2: 데이터 전처리(정규화 등)를 수행합니다
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
#### 요구사항3: CNN 모델을 설계하고 훈련시킵니다
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
#### 요구사항4: 모델의 성능을 평가하고, 테스트 이미지에 대한 예측을 수행합니다
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'테스트 정확도: {acc}')
    
    predictions = model.predict(x_test[:5])
    print('예측 결과:', predictions.argmax(axis=1))
    print('실제 정답:', y_test[:5].argmax(axis=1))
#### 실행화면
테스트 정확도: 0.7045999765396118
<img width="890" alt="image" src="https://github.com/user-attachments/assets/cca1fe94-0895-4e63-a4a4-ec3acd405518" />
