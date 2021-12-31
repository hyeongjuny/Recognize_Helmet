# 전동 킥보드 헬멧 작용여부 판단 모델

## Data Collect

> <img width="659" alt="image" src="https://user-images.githubusercontent.com/96864406/147825672-6e7d9171-f29f-4024-9b96-a195b399336c.png">

> <img width="199" alt="image" src="https://user-images.githubusercontent.com/96864406/147825724-d752a22e-76a5-4f0c-80eb-03bc567ec854.png">

> 친구 및 지인들에게 셀프 동영상을 헬멧 인증할 수 있는 각도를 고려하려 수집.
> 동영상을 프레임 단위로 읽어와 Dataset을 완성.



## Model Set

```
def build_model():
    model = models.Sequential()
    conv_base = VGG16(weights='imagenet', include_top = False, input_shape = input_shape)
    conv_base.trainable=False

    model.add(conv_base)

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    # compile
    model.compile(optimizer='RMSprop',
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# main loop without cross-validation
import time
starttime=time.time();
num_epochs = 150
model = build_model()
history = model.fit_generator(train_generator,
                    epochs=num_epochs, steps_per_epoch=100,
                    validation_data=validation_generator, validation_steps=50)

```

> Model Layer설정 및 훈련
