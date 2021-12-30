from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# set image generators
train_dir='/home/boamike/PycharmProjects/test1/Final_Project/Img_Data/train'
test_dir='/home/boamike/PycharmProjects/test1/Final_Project/Img_Data/test'
validation_dir='/home/boamike/PycharmProjects/test1/Final_Project/Img_Data/val'

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20, shear_range=0.1,
                                   width_shift_range=0.1, height_shift_range=0.1,
                                   zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(400, 400),
        batch_size=30,
        class_mode='binary')
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(400, 400),
        batch_size=30,
        class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(400, 400),
        batch_size=30,
        class_mode='binary')

# model definition
input_shape = [400, 400, 3] # as a shape of image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import BatchNormalization, Dropout, Activation
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

# saving the model
model.save('FinalProject_ep150.h5')

# evaluation
train_loss, train_acc = model.evaluate_generator(train_generator)
test_loss, test_acc = model.evaluate_generator(test_generator)
print('train_acc:', train_acc)
print('test_acc:', test_acc)
print("elapsed time (in sec): ", time.time()-starttime)

# visualization
def plot_acc(h, title="accuracy"):
    plt.plot(h.history['acc'])
    plt.plot(h.history ['val_acc'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)

def plot_loss(h, title="loss"):
    plt.plot(h.history ['loss'])
    plt.plot(h.history ['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)

plot_loss(history)
plt.savefig('Final.loss.png')
plt.clf()
plot_acc(history)
plt.savefig('Final.accuracy.png')

