# 전동 킥보드 헬멧 작용여부 판단 모델

## 1. Data Collect

> <img width="659" alt="image" src="https://user-images.githubusercontent.com/96864406/147825672-6e7d9171-f29f-4024-9b96-a195b399336c.png">

> <img width="199" alt="image" src="https://user-images.githubusercontent.com/96864406/147825724-d752a22e-76a5-4f0c-80eb-03bc567ec854.png">

```
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20, shear_range=0.1,
                                   width_shift_range=0.1, height_shift_range=0.1,
                                   zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
```

> 친구 및 지인들에게 셀프 동영상을 헬멧 인증할 수 있는 각도를 고려하려 수집.
> 동영상을 프레임 단위로 읽어와 Dataset을 완성.
> 절대적인 Data의 수가 부족하여 Argument.


## 2. Model Set

> <img width="1049" alt="image" src="https://user-images.githubusercontent.com/96864406/147826237-295b3aad-0501-4da8-a74f-b27ccb1912bc.png">

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

> Model Layer 설정 및 훈련.
> VGG16이 Overfitting이 Inception V3보다 더 늦게 일어나므로 VGG16을 사용!


## 3. Visualization

```
# visualization
def plot_acc(h, title="accuracy"):
    plt.plot(h.history['acc'])
    plt.plot(h.history['val_acc'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)

def plot_loss(h, title="loss"):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)

plot_loss(history)
plt.savefig('loss_FT_re.png')
plt.clf()
plot_acc(history)
plt.savefig('accuracy_FT_re.png')
```

> ![image](https://user-images.githubusercontent.com/96864406/147826624-39c8fa2b-5906-453f-9136-daaea5f315ec.png)

> Finetuning을 통해 최종 Model 결과 Plot 및 Save






