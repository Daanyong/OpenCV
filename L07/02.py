# CIFAR-10 데이터셋을 활용한 CNN 모델 구축

# CIFAR-10 데이터셋을 활용하여 합성곱 신경망(CNN)을 구축하고, 이미지 분류를 수행합니다

# 힌트: tensorflow.keras.datasets에서 CIFAR-10 데이터셋을 불러올 수 있습니다
# 힌트: Conv2D, MaxPooling2D, Flatten, Dense 레이어를 활용하여 CNN 모델을 구성해보세요
# 힌트: 데이터 전처리 시 픽셀 값을 0~1 범위로 정규화하면 모델의 수렴이 빨라질 수 있습니다

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# 요구사항1: CIFAR-10 데이터셋을 로드합니다
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 요구사항2: 데이터 전처리(정규화 등)를 수행합니다
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 요구사항3: CNN 모델을 설계하고 훈련시킵니다
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

# 요구사항4: 모델의 성능을 평가하고, 테스트 이미지에 대한 예측을 수행합니다
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f'테스트 정확도: {acc}')

predictions = model.predict(x_test[:5])
print('예측 결과:', predictions.argmax(axis=1))
print('실제 정답:', y_test[:5].argmax(axis=1))